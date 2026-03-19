#!/bin/bash
# ===================================================================
# Azure VM Configuration
# ===================================================================
# Edit this file to reflect your actual VM names, resource groups,
# and deployed branches. Run `./status.sh` to verify connectivity.
#
# VM_LIST format (one entry per line):
#   "<vm-name>|<resource-group>|<branch>|<gpu-type>|<port>"
#
# Fields:
#   vm-name        : Azure VM name (az vm show --name)
#   resource-group : Azure resource group the VM belongs to
#   branch         : Git branch deployed on this VM
#   gpu-type       : GPU type for informational display (T4 / H100)
#   port           : Application port (default 8765)
# ===================================================================

VM_LIST=(
    "doctor-preview-vm-main|doctor-preview-rg|main|T4|8765"
    "doctor-preview-vm-dev|doctor-preview-rg|dev|T4|8765"
    "doctor-preview-vm-staging|doctor-preview-rg|staging|T4|8765"
    "doctor-preview-vm-feature1|doctor-preview-rg|feature/face-enhance|H100|8765"
    "doctor-preview-vm-feature2|doctor-preview-rg|feature/lipsync-v2|H100|8765"
)

# Default Azure region (used for display purposes only)
DEFAULT_LOCATION="eastus"

# How long to wait (seconds) between status poll attempts when starting a VM
START_POLL_INTERVAL=15

# Maximum total seconds to wait for a VM to reach "running" state
START_TIMEOUT=300

# How long to wait for the application health endpoint to respond (seconds)
APP_HEALTH_TIMEOUT=180

# ===================================================================
# Helper: parse a VM_LIST entry into named variables
#   Usage: parse_vm_entry "$entry"
#   Sets: VM_NAME, VM_RG, VM_BRANCH, VM_GPU, VM_PORT
# ===================================================================
parse_vm_entry() {
    local entry="$1"
    VM_NAME=$(echo "$entry"   | cut -d'|' -f1)
    VM_RG=$(echo "$entry"     | cut -d'|' -f2)
    VM_BRANCH=$(echo "$entry" | cut -d'|' -f3)
    VM_GPU=$(echo "$entry"    | cut -d'|' -f4)
    VM_PORT=$(echo "$entry"   | cut -d'|' -f5)
}
