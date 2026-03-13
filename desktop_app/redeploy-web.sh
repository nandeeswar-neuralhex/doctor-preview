#!/usr/bin/env bash
# Redeploy doctor-preview web app to Azure Static Web Apps
# Usage: ./redeploy-web.sh
set -e

echo "==> Building React app..."
NODE_OPTIONS='--require ./crypto-polyfill.cjs' npx vite build

echo "==> Fetching deploy token..."
DEPLOY_TOKEN=$(az staticwebapp secrets list \
  --name doctor-preview-web \
  --resource-group doctor-preview-rg \
  --query "properties.apiKey" -o tsv)

echo "==> Deploying to Azure SWA..."
npx @azure/static-web-apps-cli deploy ./dist \
  --deployment-token "$DEPLOY_TOKEN" \
  --env production

echo ""
echo "Live at: https://brave-pebble-025d9690f.4.azurestaticapps.net"
