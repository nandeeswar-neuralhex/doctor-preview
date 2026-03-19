# Azure VM Manager

Scripts to **start**, **stop**, and check the **status** of the Azure Virtual Machines
running the Doctor Preview GPU service.

## Why these scripts?

Each branch is deployed on its own Azure VM (T4 or H100 GPU).
Keeping all 5 VMs running 24 / 7 is expensive — deallocating them when
they are not needed stops compute billing entirely while retaining the
disk and application state.

## Directory layout

```
azure_vm_manager/
├── manage.sh      ← Unified entry point (recommended)
├── start.sh       ← Start one or all VMs
├── stop.sh        ← Stop (deallocate) one or all VMs
├── status.sh      ← Status table for all VMs
└── vm_config.sh   ← VM definitions — edit this first
```

## Quick-start

```bash
# 1. Edit vm_config.sh with your actual VM names and resource groups
vim azure_vm_manager/vm_config.sh

# 2. Make scripts executable
chmod +x azure_vm_manager/*.sh

# 3. Log in to Azure (only needed once per session)
az login

# 4. Check the current status of all VMs
./azure_vm_manager/manage.sh status

# 5. Start all VMs
./azure_vm_manager/manage.sh start

# 6. Stop all VMs when done
./azure_vm_manager/manage.sh stop
```

---

## Configuration — `vm_config.sh`

Edit `VM_LIST` to match your Azure infrastructure.
Each entry is a pipe-separated string:

```
"<vm-name>|<resource-group>|<branch>|<gpu-type>|<port>"
```

Example (5 VMs matching the current setup):

```bash
VM_LIST=(
    "doctor-preview-vm-main|doctor-preview-rg|main|T4|8765"
    "doctor-preview-vm-dev|doctor-preview-rg|dev|T4|8765"
    "doctor-preview-vm-staging|doctor-preview-rg|staging|T4|8765"
    "doctor-preview-vm-feature1|doctor-preview-rg|feature/face-enhance|H100|8765"
    "doctor-preview-vm-feature2|doctor-preview-rg|feature/lipsync-v2|H100|8765"
)
```

---

## Commands

### `manage.sh` — Unified entry point

```bash
# Start
./manage.sh start                      # Start all VMs (sequential)
./manage.sh start --parallel           # Start all VMs at the same time
./manage.sh start <vm-name>            # Start one VM by name
./manage.sh start --branch dev         # Start the VM for the 'dev' branch
./manage.sh start --no-wait            # Fire-and-forget (don't wait for ready)

# Stop
./manage.sh stop                       # Stop all VMs (prompts for confirmation)
./manage.sh stop --parallel --no-confirm # Stop all at once, no prompt
./manage.sh stop <vm-name>             # Stop one VM by name
./manage.sh stop --branch staging      # Stop the VM for the 'staging' branch

# Status
./manage.sh status                     # Summary table for all VMs
./manage.sh status <vm-name>           # Detailed info for one VM

# Help
./manage.sh help
```

---

## Startup time estimates

When a VM is **deallocated** (fully stopped), starting it again takes:

| Phase | Duration |
|---|---|
| Azure infrastructure boot | ~1–2 minutes |
| Application start (model loading) | ~2–5 minutes |
| **Total (T4 or H100)** | **~3–7 minutes** |

The `start.sh` script polls the `/health` endpoint automatically and
prints the WebSocket URL as soon as the application is ready, so you
do not have to guess when to connect.

---

## Cost impact

| Scenario | Monthly cost |
|---|---|
| 5× T4 VMs running 24/7 | ~$1,900/month |
| 5× T4 VMs (8 h/day, 5 days/week) | ~$285/month |
| All VMs deallocated | ~$10–20/month (disk storage only) |

Deallocating VMs overnight and on weekends can reduce GPU compute
costs by **80–85%**.

---

## Prerequisites

- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) installed (`az --version`)
- Azure login (`az login`)
- Contributor access to the resource group(s) in `vm_config.sh`
- `curl` and `python3` available locally (used by `status.sh` for health checks)

---

## Troubleshooting

### VM not found

```
❌ VM 'my-vm' not found in vm_config.sh
```

Make sure the `vm-name` field in `vm_config.sh` matches the name shown in the
Azure Portal or `az vm list --output table`.

### Start times out

The default timeout is 300 seconds for the VM boot and 180 seconds for the
app health check. If your VMs or model downloads are slower, increase
`START_TIMEOUT` and `APP_HEALTH_TIMEOUT` in `vm_config.sh`.

### No public IP shown

Your VM may use a dynamic IP that changes on restart. Assign a static
(Standard SKU) public IP in the Azure Portal to get a stable address.

### "VM stopped (not deallocated)"

If the VM was shut down from inside the OS (`sudo shutdown`), Azure shows the
power state as `VM stopped` rather than `VM deallocated`. You are still billed
for the compute. Use `./stop.sh` (which calls `az vm deallocate`) to fully
stop billing.
