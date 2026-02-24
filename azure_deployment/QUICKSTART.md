# ⚡ QUICK START - Deploy in 5 Minutes

## 🎯 Goal
Deploy your GPU-powered face swap service to Azure and get a WebSocket URL.

## 📋 You Need
- Azure account (get $200 free: https://azure.microsoft.com/free/)
- 5 minutes of your time
- Terminal/Command Line

---

## 🚀 3-Step Deployment

### Step 1: Install Azure CLI (if not installed)

**macOS:**
```bash
brew install azure-cli
```

**Linux:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

**Windows:**
Download: https://aka.ms/installazurecliwindows

---

### Step 2: Login to Azure

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/azure_deployment

az login
```

**Browser opens automatically** → Sign in → Done! ✅

**Browser won't open?** Visit: **https://microsoft.com/devicelogin**

---

### Step 3: Deploy

```bash
./deploy.sh
```

**Wait 20-25 minutes** ⏱️ (building Docker image + downloading AI models)

---

## 📝 What You'll Be Asked

The script will ask:

1. **Subscription ID** (if you have multiple)
   - Just press Enter to use default
   - Or paste your subscription ID

That's it! Everything else is automated.

---

## ✅ Success!

You'll see:

```
✅ DEPLOYMENT SUCCESSFUL!

WebSocket URL: ws://doctor-preview-12345.eastus.azurecontainer.io:8765/ws
API URL:       http://doctor-preview-12345.eastus.azurecontainer.io:8765
```

**Copy the WebSocket URL** - you'll need it for your desktop app! 📋

---

## 🧪 Test It

```bash
# Replace with your actual endpoint
curl http://doctor-preview-12345.eastus.azurecontainer.io:8765/health

# Expected: {"status":"healthy","gpu":"CUDA available"}
```

---

## 🎮 Use It

Update your desktop app:

```javascript
// desktop_app/src/components/Settings.jsx
const WS_URL = "ws://doctor-preview-12345.eastus.azurecontainer.io:8765/ws";
```

---

## 📊 Monitor

```bash
# View status
./monitor.sh

# View logs
./monitor.sh logs

# Stream logs
./monitor.sh stream
```

---

## 💰 Costs

**T4 GPU (default):** ~$0.53/hour = ~$12.72/day

**To save money:**
- Delete when not using: `./cleanup.sh`
- Use $200 Azure free credit (new accounts)

---

## 🛑 Delete Everything

When you're done:

```bash
./cleanup.sh
```

Confirms before deleting - saves you money! 💵

---

## 📖 Need More Info?

- **Full Guide:** [README.md](README.md)
- **Authentication Help:** [AUTHENTICATION.md](AUTHENTICATION.md)
- **Cost Details:** [COSTS.md](COSTS.md)

---

## 🐛 Something Wrong?

### Container won't start?
```bash
./monitor.sh logs
```

### Can't connect?
- Check firewall settings
- Verify WebSocket URL
- Test: `curl http://YOUR_ENDPOINT:8765/health`

### Need help?
Open an issue or check the troubleshooting section in [README.md](README.md)

---

## 🎉 You're Done!

Your AI face swap service is running on Azure GPU! 🚀

**Next:** Update your desktop app and start swapping faces! 😎
