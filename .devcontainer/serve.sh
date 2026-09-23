#!/bin/bash
# Launch the app on every codespace start. Logs to app.log.
pkill -f "src.app" 2>/dev/null || true
sleep 2
nohup python -m src.app --port 7860 > app.log 2>&1 &
sleep 5
tail -5 app.log
