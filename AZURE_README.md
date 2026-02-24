# 🚀 Doctor Preview - Azure GPU Deployment

## ⚡ Quick Start

Deploying to Azure instead of RunPod? Everything you need is in the `azure_deployment/` folder!

### 3-Step Deployment:

```bash
# 1. Navigate to deployment folder
cd azure_deployment

# 2. Authenticate with Azure
az login
# Opens browser to: https://microsoft.com/devicelogin

# 3. Deploy everything
./deploy.sh
```

**That's it!** 🎉

**Time:** ~25 minutes (automatic)

---

## 📂 What's in `azure_deployment/`

| File | Description |
|------|-------------|
| **[QUICKSTART.md](azure_deployment/QUICKSTART.md)** | ⚡ Deploy in 5 minutes |
| **[README.md](azure_deployment/README.md)** | 📖 Complete deployment guide |
| **[DEPLOYMENT_PLAN.md](azure_deployment/DEPLOYMENT_PLAN.md)** | 🎯 Full architecture & plan |
| **[AUTHENTICATION.md](azure_deployment/AUTHENTICATION.md)** | 🔐 Azure login help |
| **[COSTS.md](azure_deployment/COSTS.md)** | 💰 Pricing & cost optimization |
| **[AUTH_LINKS.md](azure_deployment/AUTH_LINKS.md)** | 🔗 Browser auth links |
| `deploy.sh` | 🚀 Main deployment script |
| `update.sh` | 🔄 Update deployed container |
| `monitor.sh` | 📊 View logs & status |
| `cleanup.sh` | 🧹 Delete all resources |

---

## 💰 Cost Preview

**Development (8hrs/day, weekdays):**
- ~$90/month
- **FREE with $200 Azure credit!** (new accounts)

**Production (24/7):**
- ~$470/month (T4 GPU)
- ~$355/month with reserved instance

[Full cost breakdown →](azure_deployment/COSTS.md)

---

## 🔐 Authentication Link

Need to authenticate in browser?

**👉 https://microsoft.com/devicelogin 👈**

---

## 📋 Prerequisites

1. **Azure account** (get $200 free credit: https://azure.microsoft.com/free/)
2. **Azure CLI installed:**
   ```bash
   # macOS
   brew install azure-cli
   
   # Linux
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

---

## 🎯 What You Get

After deployment:

✅ **GPU-accelerated** face swapping (24+ FPS)  
✅ **Public WebSocket endpoint** (no tunneling needed)  
✅ **Global availability** (15+ Azure regions)  
✅ **Auto-scaling ready** (upgrade to AKS)  
✅ **99.9% uptime SLA**  
✅ **Pay-per-second billing**  

---

## 🆚 Azure vs RunPod

| Feature | Azure | RunPod |
|---------|-------|--------|
| **Reliability** | 99.9% SLA | ~95% |
| **Global Regions** | 60+ | 3-4 |
| **Auto-Scale** | Native | Manual |
| **Support** | 24/7 Enterprise | Community |
| **Cost (T4)** | $0.53/hr | $0.34/hr |
| **Free Tier** | $200 credit | None |

**Use Azure when you need:**
- Production-grade reliability
- Global low-latency deployment
- Enterprise support
- Compliance (HIPAA, SOC2)
- Auto-scaling

---

## 📖 Documentation

Start here based on your needs:

### 🏃 Just want to deploy ASAP?
→ [QUICKSTART.md](azure_deployment/QUICKSTART.md)

### 📚 Want complete details?
→ [README.md](azure_deployment/README.md)

### 🎯 Want to understand the architecture?
→ [DEPLOYMENT_PLAN.md](azure_deployment/DEPLOYMENT_PLAN.md)

### 💰 Worried about costs?
→ [COSTS.md](azure_deployment/COSTS.md)

### 🔐 Need help with Azure login?
→ [AUTHENTICATION.md](azure_deployment/AUTHENTICATION.md)

---

## 🛠️ Management Commands

```bash
# Deploy everything
./deploy.sh

# View logs
./monitor.sh logs

# Stream logs in real-time
./monitor.sh stream

# Restart container
./monitor.sh restart

# Update code and redeploy
./update.sh

# Delete everything (save money!)
./cleanup.sh
```

---

## 🚨 Important Notes

### Cost Optimization
- **Delete when not using:** `./cleanup.sh` saves ~$12/day
- **Use free credit:** New accounts get $200 (15+ days of T4 GPU)
- **Set alerts:** Configure budget alerts in Azure Portal

### Security
- Container runs as non-root user
- Private container registry (not public Docker Hub)
- Support for SSL/TLS via Azure Front Door
- Firewall rules available

### Scaling
- Single GPU: ~3-5 concurrent users (720p, 24fps)
- Multiple GPUs: Use Azure Kubernetes Service (AKS)
- Auto-scaling: Available with AKS

---

## ✅ Deployment Checklist

Before you deploy:

- [ ] Azure account created (claim free $200 credit)
- [ ] Azure CLI installed (`az --version`)
- [ ] Read cost estimates (~$90/month dev, ~$470/month prod)
- [ ] Decided on GPU type (T4 recommended)
- [ ] Selected Azure region (eastus, westus2, etc.)

After deployment:

- [ ] Tested WebSocket endpoint
- [ ] Updated desktop app with new URL
- [ ] Set up cost alerts in Azure Portal
- [ ] Verified GPU is working (`curl http://YOUR_ENDPOINT:8765/health`)

---

## 📞 Support

**Questions?** Check these docs:
1. [QUICKSTART.md](azure_deployment/QUICKSTART.md) - Fast deployment
2. [README.md](azure_deployment/README.md) - Complete guide
3. [DEPLOYMENT_PLAN.md](azure_deployment/DEPLOYMENT_PLAN.md) - Architecture
4. [Troubleshooting section](azure_deployment/README.md#troubleshooting) - Common issues

**Azure Support:**
- Portal: https://portal.azure.com/#blade/Microsoft_Azure_Support/
- Docs: https://docs.microsoft.com/azure/container-instances/

---

## 🎉 Ready to Deploy?

```bash
cd azure_deployment
az login
./deploy.sh
```

See you on Azure! 🚀
