# 🔐 Azure CLI Authentication - Browser Links

## 🌐 Authentication URLs

### Primary Method: Device Code Login

**1. Run this command:**
```bash
az login --use-device-code
```

**2. Open this link in ANY browser (desktop, phone, tablet):**

# 👉 https://microsoft.com/devicelogin 👈

**3. Enter the code** shown in your terminal

**4. Sign in** with your Microsoft account


---

## Alternative: Full Azure Login Page

**Direct Azure login portal:**

# 👉 https://login.microsoftonline.com/ 👈

---

## Automatic Browser Login (Easiest)

Just run this - browser opens automatically:

```bash
az login
```

Your default browser will open to the Azure login page.

---

## After Authentication

**Verify you're logged in:**
```bash
az account show
```

**List subscriptions:**
```bash
az account list --output table
```

**Set active subscription (if you have multiple):**
```bash
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

---

## 🚀 Ready to Deploy?

```bash
cd azure_deployment
./deploy.sh
```

**Total deployment time:** ~25 minutes

---

## 📖 Need Help?

- **Full auth guide:** [AUTHENTICATION.md](AUTHENTICATION.md)
- **Quick start:** [QUICKSTART.md](QUICKSTART.md)
- **Complete guide:** [README.md](README.md)

---

## 💡 Quick Tips

- **No Azure account?** Get $200 free: https://azure.microsoft.com/free/
- **Browser won't open?** Use device code method above
- **Multiple accounts?** Select the right subscription before deploying
- **Questions?** Check the documentation files above

---

**Authentication Link Again:**
# 🔗 https://microsoft.com/devicelogin

Bookmark this if you need to re-authenticate later!
