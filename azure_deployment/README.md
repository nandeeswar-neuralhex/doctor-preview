# 🚀 Azure GPU Deployment Guide
## Doctor Preview Face Swap Service

Complete guide to deploy the Doctor Preview face swap service to Azure GPU instances.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start (5 minutes)](#quick-start)
4. [Detailed Deployment Steps](#detailed-deployment-steps)
5. [GPU Options & Pricing](#gpu-options--pricing)
6. [Management & Monitoring](#management--monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Cost Optimization](#cost-optimization)

---

## 🎯 Overview

This deployment uses **Azure Container Instances (ACI)** with GPU support to run your AI face swap service. 

### Architecture:
```
Internet → Azure Container Instance (GPU) → Your Container
          ├── NVIDIA T4/V100/A100 GPU
          ├── CUDA 12.1 + cuDNN 8
          ├── FastAPI Server (Port 8765)
          └── WebSocket Support
```

### What You Get:
- ✅ GPU-accelerated face swapping (24+ FPS)
- ✅ Public WebSocket endpoint
- ✅ Auto-scaling ready
- ✅ Pay-per-second billing
- ✅ Fully managed (no VM maintenance)
- ✅ Global availability (15+ regions)

---

## 📦 Prerequisites

### 1. Azure Account
- Active Azure subscription
- Credit card or billing account
- Free tier OK for testing (then pay-as-you-go)

**Don't have an account?** Get $200 free credit:
👉 https://azure.microsoft.com/free/

### 2. Install Azure CLI

**macOS:**
```bash
brew update && brew install azure-cli
```

**Linux:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

**Windows:**
Download from: https://aka.ms/installazurecliwindows

**Verify Installation:**
```bash
az --version
# Should show: azure-cli 2.x.x or higher
```

### 3. Required Files (Already Included)
```
azure_deployment/
├── Dockerfile              # GPU-optimized container image
├── requirements.txt        # Python dependencies
├── deploy.sh              # Main deployment script
├── update.sh              # Update deployed container
├── monitor.sh             # Monitor & view logs
├── cleanup.sh             # Delete all resources
└── src/                   # Application source code
    ├── server.py
    ├── face_swapper.py
    ├── lip_syncer.py
    └── download_models.py
```

---

## ⚡ Quick Start

### 🚀 Deploy in 3 Commands:

```bash
# 1. Navigate to deployment folder
cd azure_deployment

# 2. Login to Azure (opens browser)
az login

# 3. Deploy everything
./deploy.sh
```

**That's it!** The script will:
1. ✅ Authenticate you via browser
2. ✅ Create resource group
3. ✅ Create container registry
4. ✅ Build Docker image (10-15 min)
5. ✅ Deploy GPU container (5-10 min)
6. ✅ Provide WebSocket URL

**Total time:** ~20-25 minutes (mostly downloading AI models)

### Expected Output:
```
✅ DEPLOYMENT SUCCESSFUL!

Container FQDN: doctor-preview-12345.eastus.azurecontainer.io
WebSocket URL:  ws://doctor-preview-12345.eastus.azurecontainer.io:8765/ws
API URL:        http://doctor-preview-12345.eastus.azurecontainer.io:8765
```

---

## 🔐 Azure Authentication

### Browser-Based Login (Recommended)

When you run `az login`, Azure CLI will:
1. Open your default browser automatically
2. Redirect to: https://login.microsoftonline.com/
3. Ask you to sign in with your Microsoft account
4. Show confirmation page
5. Auto-redirect back to CLI

**If browser doesn't open automatically:**
```bash
# Use device code flow
az login --use-device-code
# Follow the instructions and visit: https://microsoft.com/devicelogin
```

### Select Subscription (If You Have Multiple)

```bash
# List all subscriptions
az account list --output table

# Set specific subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

---

## 📖 Detailed Deployment Steps

### Step 1: Configure Your Deployment

Edit `deploy.sh` and update these variables:

```bash
# Resource names (must be unique)
RESOURCE_GROUP="doctor-preview-rg"
ACR_NAME="doctorpreviewacr123"  # Change this! (alphanumeric only)
CONTAINER_NAME="doctor-preview-gpu-instance"

# Azure region (choose closest to your users)
LOCATION="eastus"  # or: westus2, northeurope, southcentralus

# GPU type (see pricing section)
GPU_SKU="Standard_NC4as_T4_v3"  # T4 GPU (recommended)
GPU_COUNT=1
```

### Step 2: Run Deployment

```bash
./deploy.sh
```

The script will prompt you for:
- **Subscription ID** (if you have multiple subscriptions)

### Step 3: Test Your Deployment

```bash
# Get your endpoint
ENDPOINT="<your-fqdn-from-deploy-output>"

# Health check
curl http://$ENDPOINT:8765/health

# Expected response:
# {"status":"healthy","gpu":"CUDA available"}
```

### Step 4: Update Desktop App

Edit your desktop app configuration to use the new WebSocket URL:

```javascript
// In your desktop app (e.g., desktop_app/src/components/Settings.jsx)
const WS_URL = "ws://<your-fqdn>:8765/ws";
```

---

## 💰 GPU Options & Pricing

### Available GPU SKUs on Azure:

| GPU SKU | GPU Model | vCPUs | RAM | GPU Memory | Price/Hour* | Best For |
|---------|-----------|-------|-----|------------|-------------|----------|
| **Standard_NC4as_T4_v3** | NVIDIA T4 | 4 | 28 GB | 16 GB | ~$0.53 | **Recommended** - Inference |
| Standard_NC6s_v3 | NVIDIA V100 | 6 | 112 GB | 16 GB | ~$3.06 | High performance |
| Standard_NC8as_T4_v3 | NVIDIA T4 | 8 | 56 GB | 16 GB | ~$1.06 | Multiple streams |
| Standard_NC24ads_A100_v4 | NVIDIA A100 | 24 | 220 GB | 80 GB | ~$3.67 | Maximum performance |

*Prices are approximate and vary by region. Check current pricing: https://azure.microsoft.com/pricing/details/container-instances/

### Region Availability

Not all regions support GPU SKUs. **Recommended regions:**
- **eastus** - East US (Virginia)
- **southcentralus** - South Central US (Texas)
- **westus2** - West US 2 (Washington)
- **northeurope** - North Europe (Ireland)
- **westeurope** - West Europe (Netherlands)

Check availability:
```bash
az container list-skus --location eastus --output table
```

### 💡 Cost Optimization Tips

**For Development/Testing:**
- Use **T4 GPU** (Standard_NC4as_T4_v3) - cheapest option
- Delete container when not in use: `./cleanup.sh`
- Use Azure's $200 free credit for new accounts

**For Production:**
- Use **Azure Spot Instances** (up to 90% cheaper)
- Set up auto-scaling based on demand
- Use reserved instances for 24/7 workloads (save 30-50%)

---

## 🛠️ Management & Monitoring

### View Container Status
```bash
./monitor.sh
```

### View Logs
```bash
# Last 50 lines
./monitor.sh logs

# Stream in real-time
./monitor.sh stream
```

### Restart Container
```bash
./monitor.sh restart
```

### Update Container (New Code)
```bash
# Build new image and redeploy
./update.sh
```

### Manual Commands

```bash
# List all resources
az resource list --resource-group doctor-preview-rg --output table

# Get container details
az container show \
    --resource-group doctor-preview-rg \
    --name doctor-preview-gpu-instance \
    --output json

# Execute command in container
az container exec \
    --resource-group doctor-preview-rg \
    --name doctor-preview-gpu-instance \
    --exec-command "/bin/bash"

# Check GPU utilization
az container logs \
    --resource-group doctor-preview-rg \
    --name doctor-preview-gpu-instance \
    | grep -i "gpu\|cuda"
```

### Azure Portal (Web UI)

View and manage your deployment: https://portal.azure.com

1. Navigate to **Resource Groups** → `doctor-preview-rg`
2. Click on your container instance
3. View metrics, logs, and settings

---

## 🐛 Troubleshooting

### Problem: Container fails to start

**Check logs:**
```bash
./monitor.sh logs
```

**Common issues:**
- GPU quota not available in region → Try different region
- Model download failed → Check internet connectivity in container
- Out of memory → Increase memory allocation

### Problem: "GPU not found" in logs

**Verify GPU SKU:**
```bash
az container show \
    --resource-group doctor-preview-rg \
    --name doctor-preview-gpu-instance \
    --query "containers[0].resources.requests"
```

**Solution:** Ensure `--gpu-count` and `--gpu-sku` are set correctly

### Problem: Can't connect to WebSocket

**Check public IP:**
```bash
./monitor.sh
```

**Verify port is open:**
```bash
az container show \
    --resource-group doctor-preview-rg \
    --name doctor-preview-gpu-instance \
    --query "ipAddress.ports"
```

**Test connection:**
```bash
FQDN="<your-fqdn>"
curl -i -N \
    -H "Connection: Upgrade" \
    -H "Upgrade: websocket" \
    -H "Host: $FQDN:8765" \
    http://$FQDN:8765/ws
```

### Problem: High latency

**Solutions:**
- Deploy in region closer to your users
- Use Azure Front Door for global load balancing
- Enable CDN for static assets

### Problem: Out of GPU quota

**Check quotas:**
```bash
az vm list-usage --location eastus --output table | grep -i "Standard NC"
```

**Request quota increase:**
1. Go to: https://portal.azure.com
2. Navigate to **Subscriptions** → **Usage + quotas**
3. Search for "NC" (GPU SKUs)
4. Click **Request Increase**

---

## 🔄 Updating Your Service

### Update Application Code

1. Modify files in `azure_deployment/src/`
2. Run: `./update.sh`
3. Wait ~10-15 minutes for rebuild and redeploy

### Update Dependencies

1. Edit `azure_deployment/requirements.txt`
2. Run: `./update.sh`

### Change Environment Variables

```bash
# Redeploy with new env vars
az container create \
    --resource-group doctor-preview-rg \
    --name doctor-preview-gpu-instance \
    --image <your-acr>.azurecr.io/doctor-preview-gpu:latest \
    --environment-variables \
        EXECUTION_PROVIDER=CUDAExecutionProvider \
        PORT=8765 \
        ENABLE_WEBRTC=true \
        ENABLE_LIPSYNC=true \
        JPEG_QUALITY=90 \
        MAX_SESSIONS=20 \
    ...
```

---

## 🧹 Cleanup & Delete Resources

### Delete Container Only
```bash
az container delete \
    --resource-group doctor-preview-rg \
    --name doctor-preview-gpu-instance \
    --yes
```

### Delete Everything (Container + Registry + Resource Group)
```bash
./cleanup.sh
```

**⚠️ Warning:** This deletes ALL resources and is irreversible!

---

## 🚀 Advanced: Production Deployment

### Option 1: Azure Kubernetes Service (AKS)

For auto-scaling and high availability:

```bash
# Create AKS cluster with GPU nodes
az aks create \
    --resource-group doctor-preview-rg \
    --name doctor-preview-aks \
    --node-count 1 \
    --node-vm-size Standard_NC4as_T4_v3 \
    --enable-cluster-autoscaler \
    --min-count 1 \
    --max-count 5
```

See: `AKS_DEPLOYMENT.md` (coming soon)

### Option 2: Azure Virtual Machine

For full control:

```bash
# Create GPU VM
az vm create \
    --resource-group doctor-preview-rg \
    --name doctor-preview-vm \
    --size Standard_NC4as_T4_v3 \
    --image nvidia:ngc-base-version-22-04:nvidia-ngc-base-22-04-gen1:latest \
    --admin-username azureuser \
    --generate-ssh-keys
```

### Option 3: Azure App Service (Containers)

Managed platform with auto-scaling:

```bash
# Create App Service Plan with GPU (if available)
az appservice plan create \
    --name doctor-preview-plan \
    --resource-group doctor-preview-rg \
    --is-linux \
    --sku P1V2

# Deploy container
az webapp create \
    --resource-group doctor-preview-rg \
    --plan doctor-preview-plan \
    --name doctor-preview-app \
    --deployment-container-image-name <your-acr>.azurecr.io/doctor-preview-gpu:latest
```

---

## 📞 Support

### Azure Documentation
- Container Instances: https://docs.microsoft.com/azure/container-instances/
- GPU Support: https://docs.microsoft.com/azure/container-instances/container-instances-gpu
- CLI Reference: https://docs.microsoft.com/cli/azure/container

### Azure Support
- Community: https://techcommunity.microsoft.com/
- Support Portal: https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade

### Billing
- Cost Management: https://portal.azure.com/#view/Microsoft_Azure_CostManagement/
- Pricing Calculator: https://azure.microsoft.com/pricing/calculator/

---

## ✅ Deployment Checklist

Before going to production:

- [ ] Test WebSocket connectivity from desktop app
- [ ] Verify GPU is being used (check logs for CUDA)
- [ ] Load test with multiple concurrent users
- [ ] Set up monitoring/alerts in Azure Monitor
- [ ] Configure backup/disaster recovery
- [ ] Enable SSL/TLS for WebSocket (use Azure Front Door)
- [ ] Set up CI/CD pipeline (Azure DevOps or GitHub Actions)
- [ ] Configure auto-scaling policies
- [ ] Set up log aggregation (Azure Log Analytics)
- [ ] Create runbook for common issues

---

## 🎉 Next Steps

1. **Update Desktop App** - Change WebSocket URL to Azure endpoint
2. **Test Face Swap** - Upload a selfie and test the service
3. **Monitor Performance** - Watch logs and GPU utilization
4. **Optimize Costs** - Delete resources when not in use
5. **Scale Up** - Add more instances or upgrade GPU for production

---

**Happy Deploying! 🚀**

If you need help, open an issue or contact support.
