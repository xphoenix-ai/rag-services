#!/usr/bin/env bash
cd compute_service/
python main.py &

cd ../bot_backend/
python main.py &

cd ../bot_frontend/
python app_v1.py