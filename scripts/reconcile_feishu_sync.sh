#!/usr/bin/env bash
set -euo pipefail

# Reconcile committed daily digests with Feishu.  This is deliberately
# independent from the GitHub Actions dispatcher: a recovery workflow may
# finish hours after the daily dispatcher has exited.

export HOME=/home/zyw
export GH_CONFIG_DIR=/home/zyw/.config/gh
export PATH=/home/zyw/.local/bin:/home/zyw/.nvm/versions/node/v20.20.0/bin:/usr/local/bin:/usr/bin:/bin

readonly REPO_DIR="/data2/zyw/other/26summer-daily_arXiv_ai_enhanced"
readonly FEISHU_FOLDER_TOKEN="WIUBfb9OClJC7qd9aHmc5Eannsd"
readonly FEISHU_PUBLISHER="$REPO_DIR/scripts/publish_feishu_doc.py"
readonly FEISHU_READING_TRACKER="IIt3dA5uHo2bsExRYcccAKRYnBg"
readonly FEISHU_READING_TRACKER_UPDATER="$REPO_DIR/scripts/update_feishu_reading_tracker.py"
readonly LOOKBACK_DAYS="${LOOKBACK_DAYS:-7}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

if ! [[ "$LOOKBACK_DAYS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[$(timestamp)] LOOKBACK_DAYS must be a positive integer, got: $LOOKBACK_DAYS" >&2
  exit 2
fi

echo "[$(timestamp)] Reconciling the last ${LOOKBACK_DAYS} days of committed digests with Feishu"
git -C "$REPO_DIR" fetch origin main --quiet

today="$(TZ=Asia/Shanghai date '+%Y-%m-%d')"
for ((offset = 0; offset < LOOKBACK_DAYS; offset++)); do
  digest_date="$(TZ=Asia/Shanghai date -d "${today} - ${offset} days" '+%Y-%m-%d')"
  digest_path="data/${digest_date}.md"

  if ! git -C "$REPO_DIR" cat-file -e "origin/main:${digest_path}" 2>/dev/null; then
    echo "[$(timestamp)] ${digest_date}: no committed digest; skipping"
    continue
  fi

  echo "[$(timestamp)] ${digest_date}: checking and publishing the Feishu document if needed"
  set +o pipefail
  git -C "$REPO_DIR" show "origin/main:${digest_path}" \
    | python3 "$FEISHU_PUBLISHER" \
        --identity user \
        --date "$digest_date" \
        --folder-token "$FEISHU_FOLDER_TOKEN" \
        --categories "cs.AI,cs.SE" \
        --markdown -
  pipeline_status=("${PIPESTATUS[@]}")
  set -o pipefail

  if [[ "${pipeline_status[1]}" -ne 0 ]] || \
     { [[ "${pipeline_status[0]}" -ne 0 ]] && [[ "${pipeline_status[0]}" -ne 141 ]]; }; then
    echo "[$(timestamp)] ${digest_date}: Feishu document reconciliation failed" >&2
    exit 1
  fi

  python3 "$FEISHU_READING_TRACKER_UPDATER" \
    --identity user \
    --doc "$FEISHU_READING_TRACKER" \
    --date "$digest_date"
done

echo "[$(timestamp)] Feishu reconciliation completed"
