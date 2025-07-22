#!/bin/bash

# Xinfluencer Data Sync Script: Local <-> H200
#
# Usage:
#   ./sync_data_h200.sh push   # Push local data to H200
#   ./sync_data_h200.sh pull   # Pull data from H200 to local
#   ./sync_data_h200.sh status # Show sync status (dry run)
#
# This script syncs all relevant data directories between local and H200.
# It avoids overwriting newer files unless --force is specified.
# Logs all actions to sync_data_h200.log.

set -e

SSH_KEY="/Users/max/Xinfluencer/influencer.pem"
H200_HOST="157.10.162.127"
H200_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/xinfluencer"
LOG_FILE="sync_data_h200.log"

DATA_DIRS=(
  "data/collected"
  "data/safe_collection"
  "data/manual_review"
  "data/training_ready"
  "data/seed_tweets"
  "data/scraped"
)

usage() {
  echo "Usage: $0 [push|pull|status] [--force]"
  echo "  push   : Sync local data to H200 (default: skip newer remote files)"
  echo "  pull   : Sync data from H200 to local (default: skip newer local files)"
  echo "  status : Show what would be synced (dry run)"
  echo "  --force: Overwrite newer files on destination"
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

MODE="$1"
FORCE=""
if [[ "$2" == "--force" ]]; then
  FORCE="--ignore-times"
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

sync_dir() {
  local src="$1"
  local dst="$2"
  local direction="$3"
  local dryrun="$4"
  local force_flag="$5"
  local rsync_opts="-avz --progress --update $force_flag"
  if [ "$dryrun" = "yes" ]; then
    rsync_opts="$rsync_opts --dry-run"
  fi
  log "Syncing $direction: $src <-> $dst"
  rsync $rsync_opts -e "ssh -i $SSH_KEY" "$src" "$dst"
}

case "$MODE" in
  push)
    for dir in "${DATA_DIRS[@]}"; do
      if [ -d "$dir" ]; then
        sync_dir "$dir/" "$H200_USER@$H200_HOST:$REMOTE_DIR/$dir/" "local -> h200" "no" "$FORCE"
      else
        log "[SKIP] $dir does not exist locally."
      fi
    done
    ;;
  pull)
    for dir in "${DATA_DIRS[@]}"; do
      ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "[ -d $REMOTE_DIR/$dir ]" || { log "[SKIP] $dir does not exist on H200."; continue; }
      mkdir -p "$dir"
      sync_dir "$H200_USER@$H200_HOST:$REMOTE_DIR/$dir/" "$dir/" "h200 -> local" "no" "$FORCE"
    done
    ;;
  status)
    for dir in "${DATA_DIRS[@]}"; do
      if [ -d "$dir" ]; then
        sync_dir "$dir/" "$H200_USER@$H200_HOST:$REMOTE_DIR/$dir/" "local -> h200 (dry run)" "yes" "$FORCE"
      fi
      ssh -i "$SSH_KEY" "$H200_USER@$H200_HOST" "[ -d $REMOTE_DIR/$dir ]" && \
        sync_dir "$H200_USER@$H200_HOST:$REMOTE_DIR/$dir/" "$dir/" "h200 -> local (dry run)" "yes" "$FORCE"
    done
    ;;
  *)
    usage
    ;;
esac

log "Sync completed for mode: $MODE" 