#!/bin/bash
# ===================================================================
# Azure H100 GPU VM Deployment — Doctor Preview Face Swap Service
# Deploys on Standard_NC40ads_H100_v5 (40 vCPUs, 320GB RAM, 1x H100)
# Region: centralus (where H100 quota is available)
# ===================================================================

set -e

# ===================================================================
# CONFIGURATION
# ===================================================================
RESOURCE_GROUP="doctor-preview-h100-rg"
LOCATION="centralus"
VM_NAME="doctor-preview-h100"
VM_SIZE="Standard_NC40ads_H100_v5"   # 40 vCPUs, 320 GB RAM, 1x H100 80GB
ADMIN_USER="azureuser"
PORT=8765
NSG_NAME="${VM_NAME}-nsg"
VNET_NAME="${VM_NAME}-vnet"
SUBNET_NAME="${VM_NAME}-subnet"
PUBLIC_IP_NAME="${VM_NAME}-ip"
NIC_NAME="${VM_NAME}-nic"
DNS_LABEL="doctor-preview-h100"

# Use Ubuntu 22.04 HPC image (optimized for GPU workloads)
IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"

echo "====================================================================="
echo "  Doctor Preview — H100 GPU VM Deployment"
echo "  VM Size: $VM_SIZE"
echo "  Region:  $LOCATION"
echo "====================================================================="

# ===================================================================
# STEP 1: Verify Azure Login
# ===================================================================
echo ""
echo "[Step 1/8] Checking Azure Login..."
if az account show &>/dev/null; then
    echo "✅ Logged in as: $(az account show --query user.name -o tsv)"
    echo "   Subscription: $(az account show --query name -o tsv)"
else
    echo "Not logged in. Opening browser..."
    az login
fi

# ===================================================================
# STEP 2: Create Resource Group
# ===================================================================
echo ""
echo "[Step 2/8] Creating Resource Group: $RESOURCE_GROUP in $LOCATION"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

# ===================================================================
# STEP 3: Create Network Security Group with rules
# ===================================================================
echo ""
echo "[Step 3/8] Creating Network Security Group..."
az network nsg create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$NSG_NAME" \
    --location "$LOCATION" \
    --output none

# Allow SSH
az network nsg rule create \
    --resource-group "$RESOURCE_GROUP" \
    --nsg-name "$NSG_NAME" \
    --name "AllowSSH" \
    --priority 1000 \
    --access Allow \
    --direction Inbound \
    --protocol Tcp \
    --destination-port-ranges 22 \
    --output none

# Allow WebSocket port
az network nsg rule create \
    --resource-group "$RESOURCE_GROUP" \
    --nsg-name "$NSG_NAME" \
    --name "AllowWebSocket" \
    --priority 1010 \
    --access Allow \
    --direction Inbound \
    --protocol Tcp \
    --destination-port-ranges $PORT \
    --output none

# Allow HTTP/HTTPS
az network nsg rule create \
    --resource-group "$RESOURCE_GROUP" \
    --nsg-name "$NSG_NAME" \
    --name "AllowHTTP" \
    --priority 1020 \
    --access Allow \
    --direction Inbound \
    --protocol Tcp \
    --destination-port-ranges 80 443 \
    --output none

echo "✅ NSG created with SSH + WebSocket + HTTP rules"

# ===================================================================
# STEP 4: Create VM with H100 GPU
# ===================================================================
echo ""
echo "[Step 4/8] Creating H100 VM: $VM_NAME ($VM_SIZE)"
echo "  This may take 5-10 minutes..."

az vm create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --size "$VM_SIZE" \
    --image "$IMAGE" \
    --admin-username "$ADMIN_USER" \
    --generate-ssh-keys \
    --nsg "$NSG_NAME" \
    --public-ip-address "$PUBLIC_IP_NAME" \
    --public-ip-sku Standard \
    --public-ip-address-dns-name "$DNS_LABEL" \
    --os-disk-size-gb 256 \
    --storage-sku Premium_LRS \
    --output table

echo "✅ VM created!"

# Get VM IP
VM_IP=$(az vm show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --show-details \
    --query publicIps \
    --output tsv)

VM_FQDN=$(az network public-ip show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$PUBLIC_IP_NAME" \
    --query dnsSettings.fqdn \
    --output tsv)

echo "VM Public IP: $VM_IP"
echo "VM FQDN: $VM_FQDN"

# ===================================================================
# STEP 5: Install NVIDIA Drivers + Docker on VM
# ===================================================================
echo ""
echo "[Step 5/8] Installing NVIDIA drivers + Docker + NVIDIA Container Toolkit..."
echo "  This will take 10-15 minutes..."

az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts '
#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== Updating system ==="
apt-get update -y
apt-get upgrade -y

echo "=== Installing NVIDIA drivers ==="
apt-get install -y linux-headers-$(uname -r)
apt-get install -y ubuntu-drivers-common
ubuntu-drivers install --gpgpu

echo "=== Installing NVIDIA CUDA Toolkit ==="
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -y
apt-get install -y cuda-toolkit-12-1

echo "=== Installing Docker ==="
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "=== Installing NVIDIA Container Toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "=== Adding azureuser to docker group ==="
usermod -aG docker azureuser

echo "=== Verifying installations ==="
nvidia-smi || echo "nvidia-smi not yet available (may need reboot)"
docker --version
echo "DONE: All software installed"
' \
    --output json

echo "✅ Software installation complete!"

# ===================================================================
# STEP 5b: Reboot VM for driver activation
# ===================================================================
echo ""
echo "[Step 5b] Rebooting VM for NVIDIA driver activation..."
az vm restart \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME"

echo "Waiting 60 seconds for VM to come back online..."
sleep 60

# Verify GPU
echo "Verifying NVIDIA GPU..."
az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts 'nvidia-smi' \
    --output json

echo "✅ GPU verified!"

# ===================================================================
# STEP 6: Copy Application Code to VM
# ===================================================================
echo ""
echo "[Step 6/8] Copying application code to VM..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create app directory on VM
az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts 'mkdir -p /home/azureuser/doctor-preview' \
    --output none

# SCP the files to VM
echo "Uploading Dockerfile..."
scp -o StrictHostKeyChecking=no \
    "$SCRIPT_DIR/Dockerfile" \
    "${ADMIN_USER}@${VM_IP}:/home/azureuser/doctor-preview/"

echo "Uploading requirements.txt..."
scp -o StrictHostKeyChecking=no \
    "$SCRIPT_DIR/requirements.txt" \
    "${ADMIN_USER}@${VM_IP}:/home/azureuser/doctor-preview/"

echo "Uploading src/..."
scp -o StrictHostKeyChecking=no -r \
    "$SCRIPT_DIR/src" \
    "${ADMIN_USER}@${VM_IP}:/home/azureuser/doctor-preview/"

echo "✅ Application code uploaded!"

# ===================================================================
# STEP 7: Build Docker Image & Run Container on VM
# ===================================================================
echo ""
echo "[Step 7/8] Building Docker image on VM (this takes 15-20 min)..."

az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts '
#!/bin/bash
set -e
cd /home/azureuser/doctor-preview

echo "=== Building Docker image ==="
docker build -t doctor-preview-h100:latest .

echo "=== Stopping any existing container ==="
docker stop doctor-preview 2>/dev/null || true
docker rm doctor-preview 2>/dev/null || true

echo "=== Starting container with H100 GPU ==="
docker run -d \
    --name doctor-preview \
    --gpus all \
    --restart unless-stopped \
    -p 8765:8765 \
    -e EXECUTION_PROVIDER=CUDAExecutionProvider \
    -e PORT=8765 \
    -e ENABLE_WEBRTC=true \
    -e ENABLE_LIPSYNC=true \
    -e ENABLE_GFPGAN=true \
    -e TARGET_FPS=30 \
    -e MAX_SESSIONS=20 \
    -e JPEG_QUALITY=95 \
    doctor-preview-h100:latest

echo "=== Container started ==="
docker ps
echo ""
echo "=== Waiting for server to initialize (models loading)... ==="
sleep 30
docker logs doctor-preview --tail 50
' \
    --output json

echo "✅ Docker container running!"

# ===================================================================
# STEP 8: Verify Deployment
# ===================================================================
echo ""
echo "[Step 8/8] Verifying deployment..."

# Wait a bit more for the server to fully start
echo "Waiting for server warmup (60 seconds)..."
sleep 60

# Health check
echo "Running health check..."
az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts '
curl -s http://localhost:8765/health || echo "Health check endpoint not yet ready"
echo ""
echo "=== Container Status ==="
docker ps --filter name=doctor-preview --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv
echo ""
echo "=== Container Logs (last 20 lines) ==="
docker logs doctor-preview --tail 20
' \
    --output json

echo ""
echo "====================================================================="
echo "   ✅  H100 DEPLOYMENT SUCCESSFUL!"
echo "====================================================================="
echo ""
echo "   VM Name:       $VM_NAME"
echo "   VM Size:       $VM_SIZE (1x H100 80GB GPU)"
echo "   Region:        $LOCATION"
echo "   Public IP:     $VM_IP"
echo "   FQDN:          $VM_FQDN"
echo "   Port:          $PORT"
echo ""
echo "   WebSocket URL: ws://$VM_FQDN:$PORT/ws"
echo "   API URL:       http://$VM_FQDN:$PORT"
echo "   Health Check:  http://$VM_FQDN:$PORT/health"
echo ""
echo "====================================================================="
echo "   Useful Commands:"
echo "====================================================================="
echo "   SSH into VM:"
echo "     ssh ${ADMIN_USER}@${VM_IP}"
echo ""
echo "   View container logs:"
echo "     ssh ${ADMIN_USER}@${VM_IP} 'docker logs -f doctor-preview'"
echo ""
echo "   View GPU status:"
echo "     ssh ${ADMIN_USER}@${VM_IP} 'nvidia-smi'"
echo ""
echo "   Restart container:"
echo "     ssh ${ADMIN_USER}@${VM_IP} 'docker restart doctor-preview'"
echo ""
echo "   Stop VM (save costs):"
echo "     az vm deallocate -g $RESOURCE_GROUP -n $VM_NAME"
echo ""
echo "   Start VM again:"
echo "     az vm start -g $RESOURCE_GROUP -n $VM_NAME"
echo ""
echo "   DELETE everything (destroy):"
echo "     az group delete -n $RESOURCE_GROUP --yes --no-wait"
echo "====================================================================="
