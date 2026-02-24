#!/bin/bash
# ===================================================================
# Cleanup Azure Resources
# ===================================================================

set -e

RESOURCE_GROUP="doctor-preview-rg"

echo "====================================================================="
echo "⚠️  WARNING: This will delete ALL resources in:"
echo "    Resource Group: $RESOURCE_GROUP"
echo "====================================================================="
echo ""
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Deleting resource group and all resources..."
az group delete \
    --name "$RESOURCE_GROUP" \
    --yes \
    --no-wait

echo ""
echo "✅ Cleanup initiated (running in background)"
echo "Check status with: az group show --name $RESOURCE_GROUP"
