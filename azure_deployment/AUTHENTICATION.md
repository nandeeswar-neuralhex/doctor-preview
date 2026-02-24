# 🔐 Azure CLI Authentication Guide

## Quick Start: Authenticate in Your Browser

### Method 1: Auto Browser Login (Easiest) ⭐

Just run this command - your browser will open automatically:

```bash
az login
```

**What happens:**
1. ✅ Browser opens to: `https://login.microsoftonline.com/`
2. ✅ Sign in with your Microsoft account
3. ✅ Browser shows confirmation
4. ✅ Terminal shows your subscriptions

---

### Method 2: Device Code Login (If Browser Won't Open)

Use this if you're on a remote server or browser doesn't open:

```bash
az login --use-device-code
```

**Follow these steps:**

1. Copy the code shown (e.g., `ABC123DEF`)
2. Open this link in ANY browser (even on your phone):
   
   **👉 https://microsoft.com/devicelogin 👈**

3. Paste the code
4. Sign in with your Microsoft account
5. Go back to terminal - you're authenticated!

---

### Method 3: Direct Browser Link

If you need to authenticate manually in a browser:

1. **Start authentication flow:**
   ```bash
   az login --use-device-code
   ```

2. **Open this URL in your browser:**
   
   ### 🔗 https://microsoft.com/devicelogin
   
   Or the full Azure login page:
   
   ### 🔗 https://login.microsoftonline.com/

3. **Enter the code** shown in your terminal

4. **Sign in** with your Microsoft account credentials

---

## After Authentication

### Verify You're Logged In

```bash
# Show your account info
az account show

# List all your subscriptions
az account list --output table
```

### Select a Specific Subscription (If You Have Multiple)

```bash
# List subscriptions
az account list --output table

# Set active subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Verify
az account show --output table
```

---

## Account Types

### Personal Microsoft Account
- Email: `yourname@outlook.com`, `yourname@hotmail.com`, `yourname@live.com`
- Use for: Personal Azure subscriptions, free tier

### Work/School Account (Azure AD)
- Email: `yourname@company.com`
- Use for: Enterprise subscriptions, organizational resources

### Guest Account
- Invited to another organization's Azure
- Sign in with your email, then select the organization

---

## Troubleshooting

### "No subscriptions found"

**Solution:**
1. Go to: https://portal.azure.com
2. Click **Subscriptions** in left menu
3. Click **+ Add**
4. Create a **Pay-As-You-Go** subscription
5. OR claim your **$200 free credit**: https://azure.microsoft.com/free/

### "Browser did not open"

**Solution:** Use device code flow:
```bash
az login --use-device-code
```

### "Token expired"

**Solution:** Re-authenticate:
```bash
az logout
az login
```

### "Wrong tenant/directory"

**Solution:** Specify tenant:
```bash
az login --tenant YOUR_TENANT_ID
```

---

## 🔒 Security Best Practices

### Use Service Principal for Automation

For CI/CD pipelines, create a service principal:

```bash
# Create service principal
az ad sp create-for-rbac \
    --name "doctor-preview-deploy" \
    --role contributor \
    --scopes /subscriptions/YOUR_SUBSCRIPTION_ID

# Save the output (client ID, client secret, tenant ID)

# Login with service principal
az login \
    --service-principal \
    --username YOUR_CLIENT_ID \
    --password YOUR_CLIENT_SECRET \
    --tenant YOUR_TENANT_ID
```

### Logout When Done

```bash
az logout
```

---

## Quick Links

| Purpose | URL |
|---------|-----|
| **Device Login** | https://microsoft.com/devicelogin |
| **Azure Portal** | https://portal.azure.com |
| **Free Account** | https://azure.microsoft.com/free/ |
| **Billing** | https://portal.azure.com/#blade/Microsoft_Azure_Billing/BillingMenuBlade |
| **Subscriptions** | https://portal.azure.com/#blade/Microsoft_Azure_Billing/SubscriptionsBlade |

---

## Ready to Deploy?

Once authenticated, run:

```bash
cd azure_deployment
./deploy.sh
```

🎉 **That's it!** Your GPU service will be deployed in ~20 minutes.

---

## Need Help?

**Azure CLI Docs:** https://docs.microsoft.com/cli/azure/authenticate-azure-cli

**Support:** https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade
