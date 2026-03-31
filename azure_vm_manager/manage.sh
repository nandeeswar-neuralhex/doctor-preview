#!/bin/bash
# ===================================================================
# Doctor Preview — Azure VM Manager (unified entry point)
#
# A single script that delegates to start.sh / stop.sh / status.sh.
#
# Usage:
#   ./manage.sh start                    - Start all VMs
#   ./manage.sh start <vm-name>          - Start a single VM
#   ./manage.sh start --branch <name>    - Start VM for a branch
#   ./manage.sh start --parallel         - Start all VMs in parallel
#   ./manage.sh start --no-wait          - Fire-and-forget start
#
#   ./manage.sh stop                     - Stop all VMs (with prompt)
#   ./manage.sh stop <vm-name>           - Stop a single VM
#   ./manage.sh stop --branch <name>     - Stop VM for a branch
#   ./manage.sh stop --parallel          - Stop all VMs in parallel
#   ./manage.sh stop --no-confirm        - Skip confirmation
#
#   ./manage.sh status                   - Show status table
#   ./manage.sh status <vm-name>         - Detailed status for one VM
#
#   ./manage.sh help                     - Show this help message
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===================================================================
# Usage / help
# ===================================================================
print_help() {
    cat <<'EOF'

  ┌──────────────────────────────────────────────────────────────────┐
  │         Doctor Preview — Azure VM Manager                        │
  └──────────────────────────────────────────────────────────────────┘

  COMMANDS
  ────────
  start                      Start all VMs (sequential, waits for ready)
  start <vm-name>            Start a single VM by Azure VM name
  start --branch <branch>    Start the VM for a given git branch
  start --parallel           Start all VMs at the same time
  start --no-wait            Send start command without waiting

  stop                       Deallocate all VMs (prompts for confirmation)
  stop <vm-name>             Deallocate a single VM
  stop --branch <branch>     Deallocate the VM for a given git branch
  stop --parallel            Deallocate all VMs at the same time
  stop --no-confirm          Skip the confirmation prompt

  status                     Print a summary table of all VMs
  status <vm-name>           Show detailed info for a single VM

  help                       Show this help message

  STARTUP TIME ESTIMATES
  ──────────────────────
  When a VM is deallocated (fully stopped), restarting it takes:

    VM boot (Azure infrastructure)   ~1–2 minutes
    Application start (model load)   ~2–5 minutes
    ─────────────────────────────────────────────
    Total (T4 or H100)               ~3–7 minutes

  The ./start.sh script polls until the app /health endpoint returns
  HTTP 200, so you know exactly when the VM is ready to serve traffic.

  CONFIGURATION
  ─────────────
  Edit vm_config.sh to add, remove, or rename VMs.
  Each entry is a pipe-separated string:
    "<vm-name>|<resource-group>|<branch>|<gpu-type>|<port>"

  EXAMPLES
  ────────
  # Start the VM that runs the 'dev' branch
  ./manage.sh start --branch dev

  # Stop everything at end of day (no prompt)
  ./manage.sh stop --parallel --no-confirm

  # Check which VMs are running
  ./manage.sh status

  # Quick restart of a single VM
  ./manage.sh stop doctor-preview-vm-main --no-confirm
  ./manage.sh start doctor-preview-vm-main

EOF
}

# ===================================================================
# Dispatch
# ===================================================================
COMMAND="${1:-help}"
shift || true   # remaining args are passed to sub-script

case "$COMMAND" in
    start)
        exec "$SCRIPT_DIR/start.sh" "$@"
        ;;
    stop)
        exec "$SCRIPT_DIR/stop.sh" "$@"
        ;;
    status)
        exec "$SCRIPT_DIR/status.sh" "$@"
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo "❌ Unknown command: '$COMMAND'"
        print_help
        exit 1
        ;;
esac
