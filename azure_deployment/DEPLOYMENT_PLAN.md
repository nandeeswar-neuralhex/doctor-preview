# 🎯 Azure GPU Deployment - Complete Plan

## Executive Summary

This deployment migrates your Doctor Preview face swap service from RunPod to Azure GPU infrastructure. Everything is containerized, automated, and production-ready.

---

## 🏗️ Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AZURE CLOUD                              │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Resource Group: doctor-preview-rg                  │    │
│  │  Region: eastus (or your choice)                   │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────┐     │    │
│  │  │  Azure Container Registry (ACR)           │     │    │
│  │  │  - Stores Docker images                   │     │    │
│  │  │  - Private registry                       │     │    │
│  │  │  - Version control for containers         │     │    │
│  │  └──────────────────────────────────────────┘     │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────┐     │    │
│  │  │  Container Instance (GPU)                 │     │    │
│  │  │                                           │     │    │
│  │  │  Hardware:                                │     │    │
│  │  │  ├── NVIDIA T4 GPU (16GB VRAM)           │     │    │
│  │  │  ├── 4 vCPU                               │     │    │
│  │  │  ├── 28 GB RAM                            │     │    │
│  │  │  └── 50 GB SSD                            │     │    │
│  │  │                                           │     │    │
│  │  │  Software:                                │     │    │
│  │  │  ├── Ubuntu 22.04                         │     │    │
│  │  │  ├── CUDA 12.1 + cuDNN 8                 │     │    │
│  │  │  ├── Python 3.11                          │     │    │
│  │  │  ├── FastAPI + WebSocket Server          │     │    │
│  │  │  └── AI Models (Face Swap + Lip Sync)    │     │    │
│  │  │                                           │     │    │
│  │  │  Network:                                 │     │    │
│  │  │  ├── Public IP Address                    │     │    │
│  │  │  ├── DNS Name (FQDN)                      │     │    │
│  │  │  └── Port 8765 (WebSocket)               │     │    │
│  │  └──────────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         ▲
                         │
                    WebSocket (ws://)
                         │
                         ▼
┌─────────────────────────────────────┐
│      Desktop App (Electron)          │
│  ├── Camera input                   │
│  ├── Face swap preview              │
│  └── Real-time streaming            │
└─────────────────────────────────────┘
```

---

## 📦 Deployment Components

### 1. Docker Container
- **Base Image:** `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- **Size:** ~5 GB (with AI models)
- **Build Time:** 10-15 minutes
- **Startup Time:** 30-60 seconds

### 2. AI Models (Downloaded at Build Time)
| Model | Size | Purpose |
|-------|------|---------|
| inswapper_128.onnx | ~500 MB | Face swapping |
| buffalo_l | ~1.5 GB | Face detection/alignment |
| wav2lip.onnx | ~150 MB | Lip sync (optional) |
| GFPGANv1.4.pth | ~350 MB | Face enhancement (optional) |

### 3. Azure Resources Created

| Resource | Purpose | Cost/Month* |
|----------|---------|-------------|
| Resource Group | Logical container for all resources | FREE |
| Container Registry | Store Docker images | $5-40 |
| Container Instance (T4 GPU) | Run the service | $384 (24/7) |
| Public IP Address | Internet access | Included |
| Bandwidth | Data transfer | ~$0.09/GB |

*Estimated costs for 24/7 operation

---

## 🚀 Deployment Process

### Phase 1: Setup (5 minutes)
1. ✅ Install Azure CLI
2. ✅ Authenticate (`az login`)
3. ✅ Select subscription

### Phase 2: Build (10-15 minutes)
1. ✅ Create resource group
2. ✅ Create container registry
3. ✅ Build Docker image
4. ✅ Download AI models
5. ✅ Push image to registry

### Phase 3: Deploy (5-10 minutes)
1. ✅ Create GPU container instance
2. ✅ Configure networking
3. ✅ Start service
4. ✅ Health check

### Phase 4: Verify (2 minutes)
1. ✅ Test WebSocket endpoint
2. ✅ Verify GPU is active
3. ✅ Test face swap

**Total Time:** ~25-30 minutes

---

## 🔧 Configuration Options

### GPU Types Available

| SKU | GPU | vCPU | RAM | Cost/Hour | Best For |
|-----|-----|------|-----|-----------|----------|
| **Standard_NC4as_T4_v3** ⭐ | T4 | 4 | 28 GB | $0.53 | **Recommended** - Best price/performance |
| Standard_NC8as_T4_v3 | T4 | 8 | 56 GB | $1.06 | High concurrency |
| Standard_NC6s_v3 | V100 | 6 | 112 GB | $3.06 | Maximum performance |
| Standard_NC24ads_A100_v4 | A100 | 24 | 220 GB | $3.67 | Research/training |

### Azure Regions (GPU Availability)

| Region | Code | Latency (US) |
|--------|------|--------------|
| East US | `eastus` | Low (East Coast) |
| South Central US | `southcentralus` | Low (Central) |
| West US 2 | `westus2` | Low (West Coast) |
| North Europe | `northeurope` | Medium (EU) |
| West Europe | `westeurope` | Medium (EU) |

### Environment Variables

```bash
EXECUTION_PROVIDER=CUDAExecutionProvider  # Use GPU
PORT=8765                                 # WebSocket port
ENABLE_WEBRTC=true                        # WebRTC support
ENABLE_LIPSYNC=true                       # Lip sync feature
JPEG_QUALITY=85                           # Output quality (60-95)
MAX_SESSIONS=10                           # Concurrent users
TARGET_FPS=24                             # Frames per second
```

---

## 📊 Performance Expectations

### Face Swap Throughput (T4 GPU)

| Resolution | FPS | Latency |
|------------|-----|---------|
| 480p | 30-40 | ~30ms |
| 720p | 24-30 | ~40ms |
| 1080p | 15-20 | ~60ms |

### Concurrent Users (T4 GPU)

| Users | Resolution | FPS | GPU Load |
|-------|------------|-----|----------|
| 1 user | 720p | 30 FPS | 40-50% |
| 3 users | 720p | 24 FPS | 80-90% |
| 5 users | 480p | 24 FPS | 95-100% |

**For more users:** Scale horizontally (multiple GPUs)

---

## 💰 Cost Analysis

### Development (8 hrs/day × 5 days/week)

```
T4 GPU:       40 hrs/week × $0.53  = $21.20/week  = $85/month
Registry:     Basic tier            = $5/month
────────────────────────────────────────────────────────────
TOTAL: ~$90/month
```

**With $200 Azure free credit:** **First 2 months FREE!** ✨

### Production (24/7, single instance)

```
T4 GPU:       730 hrs/month × $0.53 = $387/month
Registry:     Premium tier          = $40/month
Bandwidth:    500 GB × $0.087       = $44/month
────────────────────────────────────────────────────────────
TOTAL: ~$471/month
```

**With reserved instance (1 year):**
```
T4 GPU:       30% discount          = $271/month
Registry:     Premium tier          = $40/month
Bandwidth:    500 GB × $0.087       = $44/month
────────────────────────────────────────────────────────────
TOTAL: ~$355/month
SAVINGS: $116/month = $1,392/year
```

### Cost Optimization Strategies

1. **Delete when not using** → Save ~$12/day
2. **Use free $200 credit** → First 15 days free
3. **Schedule auto-shutdown** → Save ~$8/day (nights)
4. **Reserved instances** → Save 30-50%
5. **Basic registry (dev)** → Save $35/month

---

## 🔒 Security Features

### Built-In Security

- ✅ **HTTPS/WSS support** (via Azure Front Door)
- ✅ **Private container registry** (not public)
- ✅ **Firewall rules** (restrict IP ranges)
- ✅ **Non-root container user** (UID 1000)
- ✅ **Secrets management** (Azure Key Vault integration)
- ✅ **Network isolation** (virtual networks)
- ✅ **DDoS protection** (Azure DDoS Protection)

### Compliance

Azure provides:
- SOC 1, 2, 3 certified
- ISO 27001, 27018
- HIPAA compliant
- GDPR compliant
- PCI DSS Level 1

---

## 📈 Scaling Options

### Vertical Scaling (More GPU Power)

Upgrade GPU:
```bash
# T4 → V100 (6x faster)
--gpu-sku "Standard_NC6s_v3"

# T4 → A100 (10x faster)
--gpu-sku "Standard_NC24ads_A100_v4"
```

### Horizontal Scaling (More Instances)

#### Option 1: Manual (Multiple Containers)
```bash
# Deploy 3 instances
./deploy.sh  # → instance-1
./deploy.sh  # → instance-2 (change CONTAINER_NAME)
./deploy.sh  # → instance-3 (change CONTAINER_NAME)

# Use load balancer to distribute traffic
```

#### Option 2: Auto-Scaling (Azure Kubernetes Service)
```bash
# Create AKS cluster with GPU nodes
az aks create \
    --resource-group doctor-preview-rg \
    --name doctor-preview-aks \
    --node-vm-size Standard_NC4as_T4_v3 \
    --enable-cluster-autoscaler \
    --min-count 1 \
    --max-count 10

# Deploy with Kubernetes YAML
kubectl apply -f kubernetes-deployment.yaml

# Auto-scales from 1-10 GPUs based on CPU/memory
```

---

## 🔍 Monitoring & Observability

### Built-In Monitoring

1. **Container Logs**
   ```bash
   ./monitor.sh logs    # Recent logs
   ./monitor.sh stream  # Real-time
   ```

2. **Azure Portal**
   - View metrics (CPU, memory, GPU)
   - Set up alerts
   - View billing

3. **Health Endpoint**
   ```bash
   curl http://YOUR_ENDPOINT:8765/health
   ```

### Advanced Monitoring (Optional)

1. **Azure Monitor + Application Insights**
   - Distributed tracing
   - Performance metrics
   - User analytics
   - Cost: ~$5-10/month

2. **Prometheus + Grafana**
   - Custom dashboards
   - GPU utilization graphs
   - WebSocket connection stats

---

## 🛠️ Management Scripts

All scripts located in `azure_deployment/`:

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy.sh` | Full deployment | `./deploy.sh` |
| `update.sh` | Update code and redeploy | `./update.sh` |
| `monitor.sh` | View status and logs | `./monitor.sh` |
| `cleanup.sh` | Delete all resources | `./cleanup.sh` |

---

## 🚦 Deployment Checklist

### Pre-Deployment
- [ ] Azure account created (free $200 credit claimed)
- [ ] Azure CLI installed
- [ ] Authenticated (`az login`)
- [ ] Selected subscription
- [ ] Reviewed costs (~$90/month dev, ~$470/month prod)

### Deployment
- [ ] Edited `deploy.sh` configuration (region, names)
- [ ] Ran `./deploy.sh`
- [ ] Waited 25-30 minutes
- [ ] Copied WebSocket URL from output
- [ ] Tested health endpoint

### Post-Deployment
- [ ] Updated desktop app with new WebSocket URL
- [ ] Tested face swap functionality
- [ ] Set up cost alerts in Azure Portal
- [ ] Configured auto-shutdown (if dev)
- [ ] Documented deployment for team

### Production Readiness
- [ ] Enable SSL/TLS (Azure Front Door)
- [ ] Set up monitoring/alerts
- [ ] Configure auto-scaling (if needed)
- [ ] Set up CI/CD pipeline
- [ ] Create disaster recovery plan
- [ ] Load test with expected traffic
- [ ] Review security settings

---

## 🆚 Azure vs RunPod Comparison

| Feature | Azure | RunPod |
|---------|-------|--------|
| **Reliability** | 99.9% SLA | ~95% |
| **Global Regions** | 60+ regions | 3-4 regions |
| **Auto-Scaling** | Native (AKS) | Manual |
| **Support** | 24/7 enterprise | Community |
| **Compliance** | HIPAA, SOC2, ISO | Limited |
| **Cost (T4)** | $0.53/hr | $0.34/hr |
| **Setup Time** | 25 min | 10 min |
| **Management** | Fully managed | Self-managed |
| **Free Tier** | $200 credit | None |

**Why Azure?**
- ✅ Production-grade reliability
- ✅ Global low-latency deployment
- ✅ Enterprise support and compliance
- ✅ Integrated with desktop app (easier networking)
- ✅ Auto-scaling for peak traffic

**Why RunPod?**
- ✅ Lower hourly cost (if budget-constrained)
- ✅ Faster initial setup
- ✅ Good for development/testing

---

## 📚 Documentation Files

All documentation in `azure_deployment/`:

| File | Purpose |
|------|---------|
| **[README.md](README.md)** | Complete deployment guide |
| **[QUICKSTART.md](QUICKSTART.md)** | 5-minute quick start |
| **[AUTHENTICATION.md](AUTHENTICATION.md)** | Azure login help |
| **[COSTS.md](COSTS.md)** | Detailed cost breakdown |
| **THIS FILE** | Complete deployment plan |

---

## 🎯 Success Criteria

Your deployment is successful when:

1. ✅ Container instance is running (status: Running)
2. ✅ Health endpoint returns: `{"status":"healthy","gpu":"CUDA available"}`
3. ✅ Desktop app connects to WebSocket URL
4. ✅ Face swap works in real-time (24+ FPS)
5. ✅ GPU utilization visible in logs
6. ✅ Cost alerts configured in Azure Portal

---

## 🆘 Support Resources

### Azure Resources
- **Docs:** https://docs.microsoft.com/azure/container-instances/
- **Pricing:** https://azure.microsoft.com/pricing/details/container-instances/
- **Support:** https://portal.azure.com/#blade/Microsoft_Azure_Support/

### Project Resources
- **Issues:** File an issue in the repository
- **Logs:** `./monitor.sh logs`
- **Troubleshooting:** See [README.md](README.md#troubleshooting)

---

## 🚀 Next Steps

1. **Review costs** → Read [COSTS.md](COSTS.md)
2. **Authenticate** → Follow [AUTHENTICATION.md](AUTHENTICATION.md)
3. **Deploy** → Run `./deploy.sh`
4. **Test** → Update desktop app, test face swap
5. **Monitor** → Set up alerts, watch costs
6. **Optimize** → Delete when not using, consider reserved instances

---

## 📞 Questions?

**Before deploying:**
- Review the [QUICKSTART.md](QUICKSTART.md) for fastest path
- Check [COSTS.md](COSTS.md) to understand pricing
- Read [AUTHENTICATION.md](AUTHENTICATION.md) for login help

**During deployment:**
- Watch terminal output for any errors
- Use `./monitor.sh logs` to see progress
- Check Azure Portal for resource status

**After deployment:**
- Test thoroughly before production use
- Monitor costs daily (first week)
- Set up alerts for unexpected spending

---

**Ready to deploy? Start with:**

```bash
cd azure_deployment
az login
./deploy.sh
```

🎉 **Happy deploying!**
