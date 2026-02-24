#!/bin/bash
# ===================================================================
# Monitor Azure Container Instance
# ===================================================================

RESOURCE_GROUP="doctor-preview-rg"
CONTAINER_NAME="doctor-preview-gpu-instance"

# Function to show container info
show_info() {
    echo "====================================================================="
    echo "Container Status"
    echo "====================================================================="
    az container show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --output table
    
    echo ""
    echo "====================================================================="
    echo "Connection Details"
    echo "====================================================================="
    FQDN=$(az container show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --query "ipAddress.fqdn" \
        --output tsv 2>/dev/null || echo "N/A")
    
    IP=$(az container show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --query "ipAddress.ip" \
        --output tsv 2>/dev/null || echo "N/A")
    
    echo "FQDN: $FQDN"
    echo "IP:   $IP"
    echo "WebSocket URL: ws://$FQDN:8765/ws"
}

# Function to show logs
show_logs() {
    echo ""
    echo "====================================================================="
    echo "Container Logs (last 50 lines)"
    echo "====================================================================="
    az container logs \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --tail 50
}

# Function to stream logs
stream_logs() {
    echo ""
    echo "====================================================================="
    echo "Streaming Container Logs (Ctrl+C to stop)"
    echo "====================================================================="
    az container attach \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME"
}

# Main menu
if [ "$1" == "logs" ]; then
    show_logs
elif [ "$1" == "stream" ]; then
    stream_logs
elif [ "$1" == "restart" ]; then
    echo "Restarting container..."
    az container restart \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME"
    echo "Container restarted!"
else
    show_info
    echo ""
    echo "Usage:"
    echo "  ./monitor.sh          - Show container info"
    echo "  ./monitor.sh logs     - Show recent logs"
    echo "  ./monitor.sh stream   - Stream logs in real-time"
    echo "  ./monitor.sh restart  - Restart container"
fi
