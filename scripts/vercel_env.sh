#!/usr/bin/env bash
# Copies the runtime variables from .env into the linked Vercel project (production).
# Values go straight from the file to Vercel; nothing is printed.
set -euo pipefail
for k in AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_KEY AZURE_OPENAI_API_VERSION AZURE_OPENAI_CHAT_DEPLOYMENT \
         AZURE_OPENAI_MINI_DEPLOYMENT AZURE_OPENAI_EMBEDDING_DEPLOYMENT EMBEDDING_DIM SUPABASE_URL \
         SUPABASE_SERVICE_ROLE_KEY FO_PLAYLIST_ID; do
  v=$(grep -E "^$k=" .env | head -1 | cut -d= -f2- | sed -E "s/[[:space:]]+#.*$//; s/^[\"']|[\"']$//g")
  [ -n "$v" ] || { echo "skip $k (empty)"; continue; }
  printf '%s' "$v" | npx --yes vercel env add "$k" production --force >/dev/null && echo "set $k"
done
