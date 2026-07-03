#!/usr/bin/env bash
# Launches the AI survey-simulation web app and exposes it through a
# Cloudflare quick tunnel, so a colleague can use it from anywhere
# (home, another office) with just a browser — nothing to install on her side.
#
# The public URL changes on every launch; share the freshly printed one.
# APP_PASSWORD in .env gates the page — never run this without it set.
#
# Usage:
#   ./run_app_public.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_NAME="ssr"
PORT="${PORT:-8501}"
TUNNEL_LOG="$(mktemp /tmp/cloudflared_ssr.XXXXXX.log)"

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "Error: cloudflared not found. Install it with: brew install cloudflared" >&2
    exit 1
fi

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
fi

if [ ! -f .env ] || ! grep -q "^APP_PASSWORD=" .env; then
    echo "Error: APP_PASSWORD is not set in .env — refusing to expose the app publicly without a password." >&2
    echo "Add a line like: APP_PASSWORD=your-chosen-password" >&2
    exit 1
fi

# Pre-seed Streamlit credentials so a first-ever run never blocks on the
# interactive onboarding email prompt.
mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

cleanup() {
    kill "${STREAMLIT_PID:-}" "${TUNNEL_PID:-}" 2>/dev/null || true
    rm -f "$TUNNEL_LOG"
}
trap cleanup EXIT INT TERM

# Tunnel-only exposure: bind Streamlit to localhost so the ONLY way in is
# through the tunnel (which the password then gates).
streamlit run app.py --server.address 127.0.0.1 --server.port "$PORT" --server.headless true &
STREAMLIT_PID=$!

cloudflared tunnel --url "http://localhost:${PORT}" > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

echo "Waiting for the Cloudflare tunnel URL..."
PUBLIC_URL=""
for _ in $(seq 1 30); do
    PUBLIC_URL="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)"
    [ -n "$PUBLIC_URL" ] && break
    sleep 1
done

if [ -z "$PUBLIC_URL" ]; then
    echo "Error: tunnel did not come up within 30s. Log follows:" >&2
    cat "$TUNNEL_LOG" >&2
    exit 1
fi

echo ""
echo "=============================================================="
echo "  Public URL (share this with your colleague):"
echo ""
echo "      ${PUBLIC_URL}"
echo ""
echo "  She will also need the APP_PASSWORD from your .env file."
echo "  Send the URL and password through separate channels."
echo ""
echo "  This URL changes every time you restart this script."
echo "  Press Ctrl+C to stop sharing."
echo "=============================================================="
echo ""

# Keep the Mac awake while serving (colleagues lose access if it sleeps).
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dims -w "$STREAMLIT_PID" &
fi

wait "$STREAMLIT_PID"
