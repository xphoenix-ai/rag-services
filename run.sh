#!/usr/bin/env bash
cd compute_service/
python main.py &

cd ../bot_backend/
python main.py