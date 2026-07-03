#!/usr/bin/env bash
# Launches the AI survey-simulation web app (app.py) and makes it reachable
# from other machines on the same network, so it can be shared with colleagues.
#
# Usage:
#   ./run_app.sh              # listens on port 8501
#   PORT=9000 ./run_app.sh    # listens on a custom port
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_NAME="ssr"
PORT="${PORT:-8501}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
fi

if [ ! -f .env ]; then
    echo "Warning: .env not found in $SCRIPT_DIR — GOOGLE_API_KEY must be set another way, or entered in the app's sidebar." >&2
fi

# On a machine's first-ever Streamlit run it blocks on an interactive
# "enter your email" onboarding prompt. Pre-seed an empty credentials file
# so it never waits on stdin (which would hang forever when launched this way).
mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<your-machine-ip>")"

echo "Starting SSR survey app on port ${PORT}..."
echo "  Local:   http://localhost:${PORT}"
echo "  Network: http://${LAN_IP}:${PORT}   <- share this URL with colleagues on the same network"
echo ""
echo "Press Ctrl+C to stop."

streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT}"
