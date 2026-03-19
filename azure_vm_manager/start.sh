#!/bin/bash
# ===================================================================
# Start Azure VMs
#
# Usage:
#   ./start.sh                  - Start ALL VMs defined in vm_config.sh
#   ./start.sh <vm-name>        - Start a single VM by name
#   ./start.sh --branch <name>  - Start the VM assigned to a branch
#   ./start.sh --no-wait        - Start VMs without waiting for ready
#   ./start.sh --parallel       - Start all VMs in parallel
#
# Startup time estimates (after deallocated state):
#   Azure VM boot         : ~1–2 minutes
#   Application start     : ~2–5 minutes (model loading)
#   Total (T4 / H100)     : ~3–7 minutes
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vm_config.sh"

# ===================================================================
# Defaults
# ===================================================================
TARGET_VM=""
TARGET_BRANCH=""
NO_WAIT=false
PARALLEL=false

# ===================================================================
# Parse arguments
# ===================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-wait)    NO_WAIT=true;          shift ;;
        --parallel)   PARALLEL=true;         shift ;;
        --branch)     TARGET_BRANCH="$2";    shift 2 ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 [<vm-name>] [--branch <name>] [--no-wait] [--parallel]"
            exit 1
            ;;
        *)
            TARGET_VM="$1"
            shift
            ;;
    esac
done

# ===================================================================
# Check Azure CLI login
# ===================================================================
check_login() {
    if ! az account show &>/dev/null; then
        echo "❌ Not logged in to Azure CLI."
        echo "   Run: az login"
        exit 1
    fi
}

# ===================================================================
# Wait for VM to reach "running" power state
# Returns 0 on success, 1 on timeout
# ===================================================================
wait_for_vm_running() {
    local vm_name="$1"
    local rg="$2"
    local elapsed=0

    echo "   ⏳ Waiting for VM to reach 'running' state..."
    while [[ $elapsed -lt $START_TIMEOUT ]]; do
        local power_state
        power_state=$(az vm show \
            --resource-group "$rg" \
            --name "$vm_name" \
            --show-details \
            --query "powerState" \
            --output tsv 2>/dev/null || echo "unknown")

        if [[ "$power_state" == "VM running" ]]; then
            echo "   ✅ VM is running (${elapsed}s elapsed)"
            return 0
        fi

        echo "   ⌛ State: $power_state — retrying in ${START_POLL_INTERVAL}s..."
        sleep "$START_POLL_INTERVAL"
        elapsed=$((elapsed + START_POLL_INTERVAL))
    done

    echo "   ⚠️  Timed out waiting for VM after ${START_TIMEOUT}s"
    return 1
}

# ===================================================================
# Wait for application health endpoint
# Returns 0 when healthy, 1 on timeout
# ===================================================================
wait_for_health() {
    local vm_name="$1"
    local rg="$2"
    local port="$3"
    local elapsed=0

    # Get the public IP of the VM
    local ip
    ip=$(az vm show \
        --resource-group "$rg" \
        --name "$vm_name" \
        --show-details \
        --query "publicIps" \
        --output tsv 2>/dev/null || echo "")

    if [[ -z "$ip" ]]; then
        echo "   ⚠️  Could not retrieve public IP — skipping health check"
        return 0
    fi

    local health_url="http://${ip}:${port}/health"
    echo "   🔍 Waiting for app to be healthy at $health_url"

    while [[ $elapsed -lt $APP_HEALTH_TIMEOUT ]]; do
        local http_code
        http_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
            --max-time 5 "$health_url" 2>/dev/null || echo "000")

        if [[ "$http_code" == "200" ]]; then
            echo "   ✅ Application is healthy (${elapsed}s elapsed)"
            echo "   🌐 API URL:       http://${ip}:${port}"
            echo "   🔌 WebSocket URL: ws://${ip}:${port}/ws"
            return 0
        fi

        echo "   ⌛ HTTP $http_code — retrying in ${START_POLL_INTERVAL}s..."
        sleep "$START_POLL_INTERVAL"
        elapsed=$((elapsed + START_POLL_INTERVAL))
    done

    echo "   ⚠️  App health check timed out after ${APP_HEALTH_TIMEOUT}s"
    echo "   🌐 API URL (may still be starting): http://${ip}:${port}"
    echo "   🔌 WebSocket URL:                  ws://${ip}:${port}/ws"
    return 0
}

# ===================================================================
# Start a single VM
# ===================================================================
start_vm() {
    local vm_name="$1"
    local rg="$2"
    local branch="$3"
    local gpu="$4"
    local port="$5"

    echo ""
    echo "====================================================================="
    echo "  Starting: $vm_name"
    echo "  Branch:   $branch  |  GPU: $gpu  |  Resource Group: $rg"
    echo "====================================================================="

    # Check current state to avoid redundant work
    local current_state
    current_state=$(az vm show \
        --resource-group "$rg" \
        --name "$vm_name" \
        --show-details \
        --query "powerState" \
        --output tsv 2>/dev/null || echo "unknown")

    if [[ "$current_state" == "VM running" ]]; then
        echo "  ℹ️  VM is already running — skipping start."
        if [[ "$NO_WAIT" == false ]]; then
            wait_for_health "$vm_name" "$rg" "$port"
        fi
        return 0
    fi

    echo "  ▶️  Current state: $current_state"
    echo "  ▶️  Sending start command..."

    az vm start \
        --resource-group "$rg" \
        --name "$vm_name" \
        --no-wait \
        --output none

    if [[ "$NO_WAIT" == true ]]; then
        echo "  ℹ️  Start command sent (--no-wait). VM will come up shortly."
        return 0
    fi

    wait_for_vm_running "$vm_name" "$rg" || return 1
    wait_for_health "$vm_name" "$rg" "$port"
}

# ===================================================================
# Start multiple VMs in parallel (background jobs)
# ===================================================================
start_all_parallel() {
    local pids=()
    local names=()

    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"
        start_vm "$VM_NAME" "$VM_RG" "$VM_BRANCH" "$VM_GPU" "$VM_PORT" &
        pids+=($!)
        names+=("$VM_NAME")
    done

    echo ""
    echo "====================================================================="
    echo "  Waiting for all parallel start jobs to complete..."
    echo "====================================================================="

    local all_ok=true
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "  ✅ ${names[$i]} — done"
        else
            echo "  ❌ ${names[$i]} — failed"
            all_ok=false
        fi
    done

    if [[ "$all_ok" == true ]]; then
        echo ""
        echo "✅ All VMs started successfully."
    else
        echo ""
        echo "⚠️  Some VMs failed to start. Run ./status.sh for details."
        exit 1
    fi
}

# ===================================================================
# Main
# ===================================================================
echo "====================================================================="
echo "  Doctor Preview — Azure VM Start"
echo "====================================================================="

check_login

# Resolve target list
if [[ -n "$TARGET_VM" ]]; then
    # Start a single named VM
    found=false
    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"
        if [[ "$VM_NAME" == "$TARGET_VM" ]]; then
            start_vm "$VM_NAME" "$VM_RG" "$VM_BRANCH" "$VM_GPU" "$VM_PORT"
            found=true
            break
        fi
    done
    if [[ "$found" == false ]]; then
        echo "❌ VM '$TARGET_VM' not found in vm_config.sh"
        exit 1
    fi

elif [[ -n "$TARGET_BRANCH" ]]; then
    # Start the VM for a given branch
    found=false
    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"
        if [[ "$VM_BRANCH" == "$TARGET_BRANCH" ]]; then
            start_vm "$VM_NAME" "$VM_RG" "$VM_BRANCH" "$VM_GPU" "$VM_PORT"
            found=true
            break
        fi
    done
    if [[ "$found" == false ]]; then
        echo "❌ No VM configured for branch '$TARGET_BRANCH' in vm_config.sh"
        exit 1
    fi

elif [[ "$PARALLEL" == true ]]; then
    start_all_parallel

else
    # Start all VMs sequentially
    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"
        start_vm "$VM_NAME" "$VM_RG" "$VM_BRANCH" "$VM_GPU" "$VM_PORT"
    done
    echo ""
    echo "✅ All VMs started."
fi
