#!/usr/bin/env bash
set -euo pipefail

export HOME=/home/zyw
export GH_CONFIG_DIR=/home/zyw/.config/gh
export PATH=/home/zyw/.local/bin:/home/zyw/.nvm/versions/node/v20.20.0/bin:/usr/local/bin:/usr/bin:/bin

readonly REPO="fyrsta7/26summer-daily_arXiv_ai_enhanced"
readonly WORKFLOW_ID="314783587"
readonly GH="/usr/bin/gh"
readonly JQ="/usr/bin/jq"
readonly REPO_DIR="/data2/zyw/other/26summer-daily_arXiv_ai_enhanced"
readonly FEISHU_FOLDER_TOKEN="WIUBfb9OClJC7qd9aHmc5Eannsd"
readonly FEISHU_PUBLISHER="$REPO_DIR/scripts/publish_feishu_doc.py"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(timestamp)] Checking whether today's Daily arXiv workflow already ran"

fetch_main_with_retry() {
  local attempt
  for attempt in 1 2 3 4; do
    if git -C "$REPO_DIR" fetch origin main --quiet; then
      return 0
    fi
    if [[ "$attempt" -lt 4 ]]; then
      local delay_seconds=$((attempt * 5))
      echo "[$(timestamp)] Fetch failed; retrying in ${delay_seconds}s (attempt ${attempt}/4)"
      sleep "$delay_seconds"
    fi
  done
  echo "[$(timestamp)] Unable to fetch origin/main after 4 attempts"
  return 1
}

publish_feishu_document() {
  echo "[$(timestamp)] Publishing ${local_date} digest through lark-cli"
  fetch_main_with_retry
  if git -C "$REPO_DIR" cat-file -e "origin/main:data/${local_date}.md" 2>/dev/null; then
    git -C "$REPO_DIR" show "origin/main:data/${local_date}.md" \
      | python3 "$FEISHU_PUBLISHER" \
          --identity user \
          --date "$local_date" \
          --folder-token "$FEISHU_FOLDER_TOKEN" \
          --categories "cs.AI,cs.SE" \
          --markdown -
  else
    python3 "$FEISHU_PUBLISHER" \
      --identity user \
      --date "$local_date" \
      --folder-token "$FEISHU_FOLDER_TOKEN" \
      --categories "cs.AI,cs.SE" \
      --no-new-content
  fi
}

wait_for_run_and_publish() {
  local run_id="$1"
  local status="$2"
  local conclusion="$3"
  if [[ "$status" != "completed" ]]; then
    echo "[$(timestamp)] Waiting for GitHub Actions run ${run_id}"
    "$GH" run watch "$run_id" --repo "$REPO" --exit-status
  elif [[ "$conclusion" != "success" ]]; then
    echo "[$(timestamp)] Existing run ${run_id} did not succeed: ${conclusion}"
    return 1
  fi
  publish_feishu_document
}

# The server is the only automatic trigger and runs at 09:30 Asia/Shanghai.
# Calculate Beijing midnight in UTC so late legacy runs are still recognized as
# belonging to the correct Beijing calendar day.
local_date="$(TZ=Asia/Shanghai date '+%Y-%m-%d')"
cutoff="$(date -u -d "${local_date} 00:00:00 +0800" '+%Y-%m-%dT%H:%M:%SZ')"
runs="$($GH run list \
  --repo "$REPO" \
  --limit 100 \
  --json databaseId,name,event,status,conclusion,createdAt,url)"

existing="$($JQ -r --arg cutoff "$cutoff" '
  map(
    select(.createdAt >= $cutoff)
    | select(.name == "arXiv-daily-ai-enhanced" or .name == "arXiv-daily-email")
    | select(.event == "schedule" or .event == "workflow_dispatch")
    | select(.status != "completed" or .conclusion == "success")
  )
  | sort_by(.createdAt)
  | last // empty
  | if . == "" then "" else @base64 end
' <<<"$runs")"

if [[ -n "$existing" ]]; then
  decoded="$(printf '%s' "$existing" | base64 --decode)"
  echo "[$(timestamp)] Existing healthy run found: $decoded"
  run_id="$($JQ -r '.databaseId' <<<"$decoded")"
  run_status="$($JQ -r '.status' <<<"$decoded")"
  run_conclusion="$($JQ -r '.conclusion // ""' <<<"$decoded")"
  wait_for_run_and_publish "$run_id" "$run_status" "$run_conclusion"
  exit $?
fi

echo "[$(timestamp)] No healthy run found; dispatching workflow with email enabled"
dispatch_time="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
$GH workflow run "$WORKFLOW_ID" --repo "$REPO" -f send_email=true
new_run=""
for _ in {1..12}; do
  runs="$($GH run list \
    --repo "$REPO" \
    --workflow "$WORKFLOW_ID" \
    --limit 10 \
    --json databaseId,event,status,createdAt,url)"
  new_run="$($JQ -c --arg dispatch_time "$dispatch_time" '
    map(select(.event == "workflow_dispatch" and .createdAt >= $dispatch_time))
    | sort_by(.createdAt)
    | last // empty
  ' <<<"$runs")"
  [[ -n "$new_run" ]] && break
  sleep 5
done
if [[ -z "$new_run" ]]; then
  echo "[$(timestamp)] Dispatch was accepted but its workflow run was not found"
  exit 1
fi
echo "[$(timestamp)] Dispatch accepted: $new_run"
run_id="$($JQ -r '.databaseId' <<<"$new_run")"
run_status="$($JQ -r '.status' <<<"$new_run")"
wait_for_run_and_publish "$run_id" "$run_status" ""
