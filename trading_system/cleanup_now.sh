#!/bin/bash
# 手動執行：備份 logs/*.log 後清空 .log 與 paper_last_heartbeat.txt，不碰 *.json / *.db / *.csv
set -e
cd "$(dirname "$0")"
LOGS_DIR="logs"
BACKUP="old_logs_backup.tar.gz"

# 備份：僅保留一份最新（僅 .log）
if [ -d "$LOGS_DIR" ]; then
  (cd "$LOGS_DIR" && tar czf "../$BACKUP" *.log 2>/dev/null) || true
fi
for f in "$LOGS_DIR"/*.log; do
  [ -f "$f" ] && truncate -s 0 "$f"
done
[ -f "$LOGS_DIR/paper_last_heartbeat.txt" ] && truncate -s 0 "$LOGS_DIR/paper_last_heartbeat.txt"
echo "[cleanup_now] Done: backup (if any .log) + truncated .log and paper_last_heartbeat.txt"
