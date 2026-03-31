#!/bin/bash
# ===================================================================
# Azure VM Status Overview
#
# Displays the current power state, public IP, and application health
# for every VM defined in vm_config.sh.
#
# Usage:
#   ./status.sh           - Show status table for all VMs
#   ./status.sh <vm-name> - Show detailed status for one VM
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vm_config.sh"

# ANSI colour codes (disabled when not writing to a terminal)
if [[ -t 1 ]]; then
    C_GREEN="\033[0;32m"
    C_RED="\033[0;31m"
    C_YELLOW="\033[0;33m"
    C_CYAN="\033[0;36m"
    C_RESET="\033[0m"
    C_BOLD="\033[1m"
else
    C_GREEN=""
    C_RED=""
    C_YELLOW=""
    C_CYAN=""
    C_RESET=""
    C_BOLD=""
fi

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
# Check if the application health endpoint is responding
# ===================================================================
check_app_health() {
    local ip="$1"
    local port="$2"
    if [[ -z "$ip" ]]; then
        echo "no-ip"
        return
    fi
    local http_code
    http_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
        --max-time 5 "http://${ip}:${port}/health" 2>/dev/null || echo "000")
    echo "$http_code"
}

# ===================================================================
# Print coloured power state
# ===================================================================
format_state() {
    local state="$1"
    case "$state" in
        "VM running")      echo -e "${C_GREEN}running${C_RESET}" ;;
        "VM deallocated")  echo -e "${C_RED}stopped${C_RESET}" ;;
        "VM stopped")      echo -e "${C_YELLOW}stopped(not deallocated)${C_RESET}" ;;
        "VM starting")     echo -e "${C_YELLOW}starting…${C_RESET}" ;;
        "VM deallocating") echo -e "${C_YELLOW}stopping…${C_RESET}" ;;
        *)                 echo -e "${C_CYAN}${state}${C_RESET}" ;;
    esac
}

# ===================================================================
# Show detailed status for a single VM
# ===================================================================
show_vm_detail() {
    local vm_name="$1"
    local rg="$2"
    local branch="$3"
    local gpu="$4"
    local port="$5"

    echo ""
    echo "====================================================================="
    printf "  VM:     ${C_BOLD}%s${C_RESET}\n" "$vm_name"
    printf "  Branch: %s  |  GPU: %s  |  RG: %s\n" "$branch" "$gpu" "$rg"
    echo "====================================================================="

    local details
    details=$(az vm show \
        --resource-group "$rg" \
        --name "$vm_name" \
        --show-details \
        --output json 2>/dev/null || echo "{}")

    local power_state ip
    power_state=$(echo "$details" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('powerState','unknown'))" 2>/dev/null || echo "unknown")
    ip=$(echo "$details" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('publicIps',''))" 2>/dev/null || echo "")

    printf "  Power state : "
    format_state "$power_state"

    if [[ -n "$ip" ]]; then
        printf "  Public IP   : %s\n" "$ip"
        printf "  API URL     : http://%s:%s\n" "$ip" "$port"
        printf "  WebSocket   : ws://%s:%s/ws\n" "$ip" "$port"

        local health_code
        health_code=$(check_app_health "$ip" "$port")
        if [[ "$health_code" == "200" ]]; then
            printf "  App health  : ${C_GREEN}healthy (HTTP 200)${C_RESET}\n"
        elif [[ "$health_code" == "000" ]]; then
            printf "  App health  : ${C_YELLOW}unreachable (starting or stopped)${C_RESET}\n"
        else
            printf "  App health  : ${C_RED}HTTP %s${C_RESET}\n" "$health_code"
        fi
    else
        printf "  Public IP   : (no public IP — VM may be stopped)\n"
    fi
}

# ===================================================================
# Print summary table header
# ===================================================================
print_table_header() {
    printf "\n"
    printf "${C_BOLD}%-32s %-18s %-22s %-8s %-12s %-20s${C_RESET}\n" \
        "VM Name" "Branch" "Power State" "GPU" "Public IP" "App Health"
    printf "%s\n" "$(printf '%.0s-' {1..114})"
}

# ===================================================================
# Print one summary table row
# ===================================================================
print_table_row() {
    local vm_name="$1"
    local branch="$2"
    local power_state="$3"
    local gpu="$4"
    local ip="$5"
    local health="$6"

    # Colour the state cell
    local state_str
    case "$power_state" in
        "VM running")      state_str="${C_GREEN}running${C_RESET}" ;;
        "VM deallocated")  state_str="${C_RED}stopped${C_RESET}" ;;
        "VM stopped")      state_str="${C_YELLOW}stopped(ND)${C_RESET}" ;;
        "VM starting")     state_str="${C_YELLOW}starting…${C_RESET}" ;;
        "VM deallocating") state_str="${C_YELLOW}stopping…${C_RESET}" ;;
        *)                 state_str="${C_CYAN}${power_state}${C_RESET}" ;;
    esac

    local health_str
    case "$health" in
        "200")    health_str="${C_GREEN}healthy${C_RESET}" ;;
        "no-ip")  health_str="${C_YELLOW}no-ip${C_RESET}" ;;
        "000")    health_str="${C_YELLOW}unreachable${C_RESET}" ;;
        *)        health_str="${C_RED}HTTP $health${C_RESET}" ;;
    esac

    printf "%-32s %-18s %-22b %-8s %-12s %-20b\n" \
        "$vm_name" "$branch" "$state_str" "$gpu" "${ip:-(none)}" "$health_str"
}

# ===================================================================
# Show status table for all VMs
# ===================================================================
show_all_status() {
    print_table_header

    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"

        local details power_state ip health_code
        details=$(az vm show \
            --resource-group "$VM_RG" \
            --name "$VM_NAME" \
            --show-details \
            --output json 2>/dev/null || echo "{}")

        power_state=$(echo "$details" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('powerState','unknown'))" \
            2>/dev/null || echo "unknown")

        ip=$(echo "$details" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('publicIps',''))" \
            2>/dev/null || echo "")

        if [[ "$power_state" == "VM running" && -n "$ip" ]]; then
            health_code=$(check_app_health "$ip" "$VM_PORT")
        else
            health_code="no-ip"
        fi

        print_table_row "$VM_NAME" "$VM_BRANCH" "$power_state" "$VM_GPU" "$ip" "$health_code"
    done

    echo ""
    echo "  Tip: ./start.sh                 — start all VMs"
    echo "  Tip: ./stop.sh                  — stop all VMs"
    echo "  Tip: ./start.sh <vm-name>       — start a specific VM"
    echo "  Tip: ./stop.sh  <vm-name>       — stop  a specific VM"
    echo "  Tip: ./start.sh --branch <name> — start VM for a branch"
    echo ""
}

# ===================================================================
# Main
# ===================================================================
echo "====================================================================="
echo "  Doctor Preview — Azure VM Status"
echo "====================================================================="

check_login

TARGET_VM="${1:-}"

if [[ -n "$TARGET_VM" ]]; then
    found=false
    for entry in "${VM_LIST[@]}"; do
        parse_vm_entry "$entry"
        if [[ "$VM_NAME" == "$TARGET_VM" ]]; then
            show_vm_detail "$VM_NAME" "$VM_RG" "$VM_BRANCH" "$VM_GPU" "$VM_PORT"
            found=true
            break
        fi
    done
    if [[ "$found" == false ]]; then
        echo "❌ VM '$TARGET_VM' not found in vm_config.sh"
        exit 1
    fi
else
    show_all_status
fi
