#!/usr/bin/env python3
"""Deploy face-analysis to Azure Container Instance (CPU — MediaPipe is CPU-based)."""
import subprocess, yaml, sys

RG = "doctor-preview-rg"
ACR = "doctorpreviewacr"
IMAGE = "doctorpreviewacr.azurecr.io/face-analysis-gpu:latest"
CONTAINER = "face-analysis-instance"
PORT = 8766

# Get ACR password
pw = subprocess.check_output(
    ["az", "acr", "credential", "show", "--name", ACR, "--resource-group", RG,
     "--query", "passwords[0].value", "--output", "tsv"],
    text=True
).strip()
print(f"✓ ACR password obtained ({len(pw)} chars)")

# Build YAML — CPU-optimized (MediaPipe/OpenCV are CPU-based analyzers)
data = {
    "apiVersion": "2021-10-01",
    "location": "eastus",
    "name": CONTAINER,
    "type": "Microsoft.ContainerInstance/containerGroups",
    "properties": {
        "containers": [{
            "name": "face-analysis",
            "properties": {
                "image": IMAGE,
                "resources": {
                    "requests": {"cpu": 4.0, "memoryInGB": 8.0}
                },
                "ports": [{"port": PORT, "protocol": "TCP"}],
                "environmentVariables": [{"name": "PORT", "value": str(PORT)}],
            },
        }],
        "imageRegistryCredentials": [{
            "server": "doctorpreviewacr.azurecr.io",
            "username": ACR,
            "password": pw,
        }],
        "osType": "Linux",
        "ipAddress": {
            "type": "Public",
            "ports": [{"port": PORT, "protocol": "TCP"}],
            "dnsNameLabel": "face-analysis-svc",
        },
        "restartPolicy": "Always",
    },
}

yaml_path = "/tmp/face-analysis-deploy.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data, f, default_flow_style=False)
print(f"✓ YAML written to {yaml_path}")

# Deploy
print(f"⏳ Deploying {CONTAINER} to Azure (typically 2-5 minutes)...")
result = subprocess.run(
    ["az", "container", "create", "--resource-group", RG, "--file", yaml_path, "--output", "table"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(f"⚠️  stderr: {result.stderr}")
    sys.exit(result.returncode)

# Get IP
print("✓ Getting connection details...")
ip_result = subprocess.check_output(
    ["az", "container", "show", "--resource-group", RG, "--name", CONTAINER,
     "--query", "ipAddress.ip", "--output", "tsv"],
    text=True
).strip()

fqdn_result = subprocess.check_output(
    ["az", "container", "show", "--resource-group", RG, "--name", CONTAINER,
     "--query", "ipAddress.fqdn", "--output", "tsv"],
    text=True
).strip()

print(f"""
{'='*60}
✅ FACE ANALYSIS SERVICE DEPLOYED!
{'='*60}

  IP:        {ip_result}
  FQDN:      {fqdn_result}
  Port:      {PORT}

  API URL:        http://{ip_result}:{PORT}
  Health Check:   http://{ip_result}:{PORT}/health
  Validate:       POST http://{ip_result}:{PORT}/validate
  Analyze:        POST http://{ip_result}:{PORT}/analyze

{'='*60}
  Logs:   az container logs -g {RG} -n {CONTAINER}
  Status: az container show -g {RG} -n {CONTAINER} -o table
  Delete: az container delete -g {RG} -n {CONTAINER} --yes
{'='*60}
""")
