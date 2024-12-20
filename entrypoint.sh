#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate rag_env

# Re-enable strict mode:
set -euo pipefail

# exec the final command:
# exec python run.py

echo "[DEBUG] Compute Service..."
cd /rag_services/compute_service/
# exec python main.py &
python main.py &

echo "[DEBUG] Bot Backend..."
cd /rag_services/bot_backend/
# exec python main.py &
python main.py &

echo "[DEBUG] Bot Frontend..."
cd /rag_services/bot_frontend/
python app_v2.py