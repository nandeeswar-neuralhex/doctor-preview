#!/bin/bash
# ===================================================================
# Stop (deallocate) Azure VMs
#
# Deallocating a VM stops billing for compute. The OS disk and data
# disks are retained — the VM can be restarted at any time.
#
# Usage:
#   ./stop.sh                  - Stop ALL VMs defined in vm_config.sh
#   ./stop.sh <vm-name>        - Stop a single VM by name
#   ./stop.sh --branch <name>  - Stop the VM assigned to a branch
#   ./stop.sh --parallel       - Stop all VMs simultaneously
#   ./stop.sh --no-confirm     - Skip confirmation prompt
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vm_config.sh"

# ===================================================================
# Defaults
# ===================================================================
TARGET_VM=""
TARGET_BRANCH=""
PARALLEL=false
NO_CONFIRM=false

# ===================================================================
# Parse arguments
# ===================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)    PARALLEL=true;       shift ;;
        --no-confirm)  NO_CONFIRM=true;     shift ;;
        --branch)      TARGET_BRANCH="$2";  shift 2 ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 [<vm-name>] [--branch <name>] [--parallel] [--no-confirm]"
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
# Stop (deallocate) a single VM
# ===================================================================
stop_vm() {
    local vm_name="$1"
    local rg="$2"
    local branch="$3"
    local gpu="$4"

    echo ""
    echo "====================================================================="
    echo "  Stopping: $vm_name"
    echo "  Branch:   $branch  |  GPU: $gpu  |  Resource Group: $rg"
    echo "====================================================================="

    # Check current state
    local current_state
    current_state=$(az vm show \
        --resource-group "$rg" \
        --name "$vm_name" \
        --show-details \
        --query "powerState" \
        --output tsv 2>/dev/null || echo "unknown")

    if [[ "$current_state" == "VM deallocated" ]]; then
        echo "  ℹ️  VM is already deallocated — no action needed."
        return 0
    fi

    echo "  🔴 Current state: $current_state"
    echo "  🛑 Deallocating VM (this stops all compute billing)..."

    az vm deallocate \
        --resource-group "$rg" \
        --name "$vm_name" \
        --no-wait \
        --output none

    echo "  ✅ Deallocate command sent. VM will stop within 1–2 minutes."
    echo "     Verify with: ./status.sh"
}

# ===================================================================
# Stop multiple VMs in parallel (background jobs)
# ===================================================================
stop_all_parallel() {
    local pids=()
    local names=()

    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"
        stop_vm "$VM_NAME" "$VM_RG" "$VM_BRANCH" "$VM_GPU" &
        pids+=($!)
        names+=("$VM_NAME")
    done

    echo ""
    echo "====================================================================="
    echo "  Waiting for all parallel stop jobs to complete..."
    echo "====================================================================="

    local all_ok=true
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "  ✅ ${names[$i]} — deallocate command sent"
        else
            echo "  ❌ ${names[$i]} — failed"
            all_ok=false
        fi
    done

    if [[ "$all_ok" == true ]]; then
        echo ""
        echo "✅ All stop commands sent. VMs will deallocate within 1–2 minutes."
        echo "   Run ./status.sh to verify."
    else
        echo ""
        echo "⚠️  Some VMs failed to stop. Run ./status.sh for details."
        exit 1
    fi
}

# ===================================================================
# Confirmation prompt (for "stop all" operations)
# ===================================================================
confirm_stop_all() {
    if [[ "$NO_CONFIRM" == true ]]; then
        return 0
    fi
    echo ""
    echo "⚠️  This will deallocate ALL ${#VM_LIST[@]} VMs."
    echo "   Compute billing will stop, but disk storage charges remain."
    read -rp "   Continue? (yes/no): " ANSWER
    if [[ "$ANSWER" != "yes" ]]; then
        echo "Cancelled."
        exit 0
    fi
}

# ===================================================================
# Main
# ===================================================================
echo "====================================================================="
echo "  Doctor Preview — Azure VM Stop"
echo "====================================================================="

check_login

if [[ -n "$TARGET_VM" ]]; then
    # Stop a single named VM
    found=false
    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"
        if [[ "$VM_NAME" == "$TARGET_VM" ]]; then
            stop_vm "$VM_NAME" "$VM_RG" "$VM_BRANCH" "$VM_GPU"
            found=true
            break
        fi
    done
    if [[ "$found" == false ]]; then
        echo "❌ VM '$TARGET_VM' not found in vm_config.sh"
        exit 1
    fi

elif [[ -n "$TARGET_BRANCH" ]]; then
    # Stop the VM for a given branch
    found=false
    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"
        if [[ "$VM_BRANCH" == "$TARGET_BRANCH" ]]; then
            stop_vm "$VM_NAME" "$VM_RG" "$VM_BRANCH" "$VM_GPU"
            found=true
            break
        fi
    done
    if [[ "$found" == false ]]; then
        echo "❌ No VM configured for branch '$TARGET_BRANCH' in vm_config.sh"
        exit 1
    fi

elif [[ "$PARALLEL" == true ]]; then
    confirm_stop_all
    stop_all_parallel

else
    # Stop all VMs sequentially
    confirm_stop_all
    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"
        stop_vm "$VM_NAME" "$VM_RG" "$VM_BRANCH" "$VM_GPU"
    done
    echo ""
    echo "✅ All stop commands sent."
fi
