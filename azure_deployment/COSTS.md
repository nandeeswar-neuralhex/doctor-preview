# 💰 Azure GPU Deployment - Cost Guide

## 📊 Pricing Breakdown

### GPU Container Instance Costs

| Component | SKU | Specs | Cost/Hour | Cost/Day* | Cost/Month* |
|-----------|-----|-------|-----------|-----------|-------------|
| **T4 GPU** (Recommended) | Standard_NC4as_T4_v3 | 4 vCPU, 28GB RAM, 1x T4 | $0.526 | $12.62 | $379 |
| **V100 GPU** | Standard_NC6s_v3 | 6 vCPU, 112GB RAM, 1x V100 | $3.06 | $73.44 | $2,203 |
| **A100 GPU** | Standard_NC24ads_A100_v4 | 24 vCPU, 220GB RAM, 1x A100 | $3.67 | $88.08 | $2,642 |

*Assuming 100% uptime. Actual costs vary by region and usage.

### Container Registry Costs

| Tier | Storage | Bandwidth | Cost/Month |
|------|---------|-----------|------------|
| **Premium** (required for geo-replication) | 500 GB | Unlimited | $40 |
| Basic (sufficient for testing) | 10 GB | 100 GB | $5 |

**Note:** Deploy script uses Premium. Change to Basic for dev:
```bash
# In deploy.sh, change:
--sku Premium
# to:
--sku Basic
```

### Storage Costs (Minimal)

- **Blob Storage:** ~$0.02 per GB/month
- **AI Model Storage:** ~2-3 GB = ~$0.06/month

---

## 💡 Cost Optimization Strategies

### 1. Delete When Not Using (Biggest Savings!)

```bash
# Delete container (keep registry for quick redeploy)
az container delete \
    --resource-group doctor-preview-rg \
    --name doctor-preview-gpu-instance \
    --yes

# Savings: ~$12.62/day (T4 GPU)
```

**Quick redeploy later:**
```bash
./deploy.sh  # Skip build, use cached image (~2 min)
```

### 2. Use Azure Free Credit

**New accounts get $200 credit (valid 30 days):**
- Sign up: https://azure.microsoft.com/free/
- Covers ~15 days of T4 GPU usage
- Or ~2.5 days of V100 usage

### 3. Schedule Auto-Shutdown

Use Azure Automation to stop container at night:

```bash
# Create automation account
az automation account create \
    --resource-group doctor-preview-rg \
    --name doctor-preview-automation \
    --location eastus

# Add runbook to stop at 6 PM, start at 8 AM
# Saves ~16 hours/day = ~$211/month (T4)
```

### 4. Use Spot Instances (Up to 90% Savings!)

**Not available for Container Instances, but available for VMs:**

```bash
az vm create \
    --resource-group doctor-preview-rg \
    --name doctor-preview-vm \
    --size Standard_NC4as_T4_v3 \
    --priority Spot \
    --max-price 0.20 \  # $0.20/hour vs $0.53 regular
    --eviction-policy Deallocate
```

**Trade-off:** Can be evicted when Azure needs capacity.

### 5. Reserved Instances (30-50% Savings for 24/7)

If running 24/7 for production:

| Commitment | T4 GPU Discount | V100 GPU Discount |
|------------|-----------------|-------------------|
| 1 Year | -30% | -30% |
| 3 Years | -50% | -50% |

**Purchase reserved capacity:**
1. Go to: https://portal.azure.com/#blade/Microsoft_Azure_Reservations/
2. Select "Compute" → "Virtual Machines"
3. Choose region + SKU + term

### 6. Scale Down During Low Usage

```bash
# Switch to smaller GPU during off-hours
# T4 (4 vCPU) → T4 (1 vCPU)

az container create \
    --resource-group doctor-preview-rg \
    --name doctor-preview-gpu-instance-small \
    --cpu 1 \  # Reduced from 4
    --memory 14 \  # Reduced from 28
    --gpu-count 1 \
    --gpu-sku "Standard_NC4as_T4_v3"
```

**Savings:** Minimal, GPU is the main cost.

---

## 📈 Usage Scenarios & Estimated Costs

### Development (8 hours/day, 5 days/week)

| Resource | Cost |
|----------|------|
| T4 GPU (40 hrs/week) | $84.16/month |
| Container Registry (Basic) | $5/month |
| **Total** | **~$89/month** |

**With Azure Free Credit:** First month FREE! ✨

### Testing (24/7, 1 week)

| Resource | Cost |
|----------|------|
| T4 GPU (168 hours) | $88.37 |
| Container Registry (Basic) | $1.25 |
| **Total** | **~$90/week** |

### Production (24/7, auto-scaling)

**Single Instance:**
| Resource | Cost/Month |
|----------|------------|
| T4 GPU (730 hrs) | $384 |
| Container Registry (Premium) | $40 |
| Bandwidth (~500 GB) | $43 |
| **Total** | **~$467/month** |

**Reserved Instance (1 year):**
| Resource | Cost/Month |
|----------|------------|
| T4 GPU (30% discount) | $269 |
| Container Registry | $40 |
| Bandwidth | $43 |
| **Total** | **~$352/month** |
| **Annual Savings** | **$1,380** |

### High Traffic (Auto-scaling, 3-10 instances)

| Scenario | Avg Instances | Cost/Month |
|----------|---------------|------------|
| Low traffic | 3 | ~$1,200 |
| Medium traffic | 5 | ~$2,000 |
| High traffic | 10 | ~$4,000 |

**Note:** Use Azure Kubernetes Service (AKS) for auto-scaling.

---

## 🔍 Monitoring Costs in Real-Time

### Azure Cost Management

1. Go to: https://portal.azure.com/#blade/Microsoft_Azure_CostManagement/
2. Select your subscription
3. View daily costs by resource

### Set Budget Alerts

```bash
# Create budget alert
az consumption budget create \
    --budget-name "doctor-preview-budget" \
    --amount 500 \
    --category cost \
    --time-grain monthly \
    --subscription YOUR_SUBSCRIPTION_ID

# Get alert when 80% and 100% of budget used
```

### View Current Spending

```bash
# Show costs for resource group
az consumption usage list \
    --start-date 2026-02-01 \
    --end-date 2026-02-28 \
    --query "[?contains(instanceName, 'doctor-preview')]" \
    --output table
```

---

## 🚨 Cost Alerts to Set Up

Create these alerts to avoid surprises:

```bash
# Alert 1: Daily cost > $20
# Alert 2: Monthly cost > $500
# Alert 3: Unexpected resource creation
# Alert 4: Container running > 24 hours (dev)
```

**Set up in Portal:**
1. Go to: https://portal.azure.com/#blade/Microsoft_Azure_CostManagement/
2. Click **Budgets** → **Add**
3. Set threshold alerts (80%, 100%, 120%)

---

## 💸 Hidden Costs to Watch

### 1. Bandwidth (Egress)

| Type | Cost |
|------|------|
| First 100 GB/month | FREE |
| 100 GB - 10 TB | $0.087/GB |
| 10 TB+ | $0.083/GB |

**For face swap video:** ~1 GB per 10 minutes of streaming
- **10 users, 1 hr/day:** ~300 GB/month = $17.40
- **100 users, 1 hr/day:** ~3 TB/month = $257

### 2. Storage (Models)

- **Model files:** 2-3 GB (one-time download)
- **Temporary cache:** < 1 GB
- **Cost:** ~$0.10/month (negligible)

### 3. Container Registry

- **Storage per image:** ~5 GB
- **Multiple versions:** 5 versions × 5 GB = 25 GB
- **Premium tier:** Included in $40/month
- **Basic tier:** $5/month + $0.10/GB over 10 GB

### 4. Logging & Monitoring (Optional)

- **Log Analytics:** $2.76/GB ingested
- **Application Insights:** $2.30/GB ingested
- **Typical usage:** 1-2 GB/month = $5-10/month

---

## 📝 Cost Comparison: Azure vs RunPod

| Feature | Azure (T4) | RunPod (RTX 3090) |
|---------|------------|-------------------|
| **Hourly Cost** | $0.53 | $0.34 |
| **Monthly (24/7)** | $384 | $247 |
| **Reliability** | 99.9% SLA | ~95% (variable) |
| **Network** | Global, low latency | Limited regions |
| **Support** | Enterprise support | Community |
| **Auto-scaling** | Native (AKS) | Manual |
| **Free Tier** | $200 credit | None |

**When to use Azure:**
- ✅ Need reliability (production)
- ✅ Global low latency
- ✅ Auto-scaling
- ✅ Enterprise support
- ✅ Compliance (HIPAA, SOC2, etc.)

**When to use RunPod:**
- ✅ Development/testing
- ✅ Budget-constrained
- ✅ Single region OK
- ✅ Manual scaling OK

---

## 🎯 Recommended Setup by Budget

### Budget: $100/month
- **Resource:** T4 GPU
- **Usage:** 8 hrs/day, weekdays only
- **Registry:** Basic
- **Strategy:** Delete nights/weekends

### Budget: $500/month
- **Resource:** T4 GPU
- **Usage:** 24/7 with monitoring
- **Registry:** Premium
- **Strategy:** Single instance, reserved if long-term

### Budget: $2,000/month
- **Resource:** V100 GPU or 3-5× T4 GPUs
- **Usage:** Auto-scaling (AKS)
- **Registry:** Premium with geo-replication
- **Strategy:** Reserved instances, CDN for bandwidth

### Budget: $5,000+/month
- **Resource:** A100 GPU or 10+ T4 GPUs
- **Usage:** Global deployment, multi-region
- **Registry:** Premium, geo-replicated
- **Strategy:** Reserved, Spot mix, CDN, Front Door

---

## 🧮 Cost Calculator

Use Azure's calculator: https://azure.microsoft.com/pricing/calculator/

**Pre-configured estimate:**
1. Add **Container Instances** → GPU (NC-series T4)
2. Add **Container Registry** → Premium
3. Add **Bandwidth** → Outbound data
4. Add **Storage** → Blob storage

---

## ✅ Cost Checklist

Before deploying:

- [ ] Understand hourly rate for your GPU SKU
- [ ] Set budget alerts ($100, $500, etc.)
- [ ] Enable auto-shutdown (dev environments)
- [ ] Use Basic registry tier for testing
- [ ] Consider spot instances (VMs) for non-critical workloads
- [ ] Plan for bandwidth costs (video streaming)
- [ ] Use reserved instances if 24/7 (production)
- [ ] Delete resources when not in use

---

## 🔗 Useful Links

| Resource | URL |
|----------|-----|
| **Pricing Calculator** | https://azure.microsoft.com/pricing/calculator/ |
| **Container Instances Pricing** | https://azure.microsoft.com/pricing/details/container-instances/ |
| **Cost Management** | https://portal.azure.com/#blade/Microsoft_Azure_CostManagement/ |
| **Free Account ($200 credit)** | https://azure.microsoft.com/free/ |
| **Reserved Instances** | https://portal.azure.com/#blade/Microsoft_Azure_Reservations/ |

---

**Questions about costs?** Check the billing section in Azure Portal or contact Azure support!

💡 **Pro Tip:** Start with the free $200 credit to test everything for free!
