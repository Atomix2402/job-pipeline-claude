"""Workflow B: recruitment emails -> update EXISTING Notion jobs.

Run from the repo root:  uv run main.py
Env (optional): LOOKBACK_HOURS=48  MAX_EMAILS=50  LLM_DELAY=4  DRY_RUN=1
Logs never contain email bodies.
"""
import json
import os
import re
import time
from pathlib import Path

from gmail.service import get_email_details, get_gmail_service, search_emails
from llm.provider import LLMError
from matching.matcher import process_email
from notion.client import create_manual_job, query_jobs, update_job

STATE_FILE = Path("state/processed.json")
KEEP_IDS = 2000

LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "48"))
MAX_EMAILS = int(os.getenv("MAX_EMAILS", "50"))
LLM_DELAY = float(os.getenv("LLM_DELAY", "4"))   # seconds; keeps free-tier rate limits happy
DRY_RUN = os.getenv("DRY_RUN") == "1"


def load_processed() -> list[str]:
    try:
        return json.loads(STATE_FILE.read_text()).get("ids", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_processed(ids: list[str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"ids": ids[-KEEP_IDS:]}))


def domain(sender: str) -> str:
    m = re.search(r"@([\w.\-]+)", sender)
    return m.group(1) if m else "unknown"


def write_summary(lines: list[str]) -> None:
    """Show results on the Actions run page (private repo only)."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")


def main() -> None:
    processed = load_processed()
    seen = set(processed)

    service = get_gmail_service()
    messages = [m for m in search_emails(service, hours=LOOKBACK_HOURS, limit=MAX_EMAILS)
                if m["id"] not in seen]
    if DRY_RUN:
        print("*** DRY RUN: nothing will be written to Notion ***")
    print(f"{len(messages)} new emails to check (lookback {LOOKBACK_HOURS}h)")
    if not messages:
        return

    jobs = query_jobs()
    print(f"{len(jobs)} jobs loaded from Notion")

    updated, added, review, skipped, failed = 0, 0, [], 0, 0

    for msg in messages:
        email = get_email_details(service, msg["id"])
        try:
            decision = process_email(email, jobs)
            time.sleep(LLM_DELAY)
        except LLMError as e:
            failed += 1
            print(f"LLM failed for {msg['id']}: {str(e)[:400]}")
            if "rate_limited" in str(e):
                print("Rate limited on all providers; stopping early, will retry next run.")
                break
            continue   # not marked processed -> retried next run

        action = decision["action"]
        if DRY_RUN or os.getenv("VERBOSE") == "1":   # local debugging only, never in CI
            print(f"  {domain(email['sender'])} | {email['subject'][:60]} -> "
                  f"{action}: {decision.get('reason', '')[:80]}")
        try:
            if action == "update":
                if not DRY_RUN:
                    update_job(decision["page_id"], decision["updates"])
                # keep local copy current so later emails see the new status
                for j in jobs:
                    if j["page_id"] == decision["page_id"]:
                        j["status"] = decision["updates"].get("status", j["status"])
                        j["last_email_event"] = decision["updates"].get(
                            "last_email_event", j["last_email_event"])
                updated += 1
            elif action == "create":
                new = decision["job"]
                page_id = "dry-run" if DRY_RUN else create_manual_job(new)
                jobs.append({"page_id": page_id or "", "company_name": new["company_name"],
                             "job_title": new["job_title"], "job_id": new["job_id"],
                             "status": "Applied", "last_email_event": new["last_email_event"],
                             "needs_review": True})   # later emails see it as already tracked
                added += 1
            elif action == "review":
                if decision.get("page_id") and not DRY_RUN:
                    update_job(decision["page_id"], decision["updates"])
                review.append(f"- {domain(email['sender'])} | {email['subject'][:60]} "
                              f"| {decision['reason']}")
            else:
                skipped += 1
        except RuntimeError as e:   # Notion error
            failed += 1
            print(f"Notion failed for {msg['id']}: {str(e)[:120]}")
            continue

        if action == "update":
            label = "[dry-run] would update" if DRY_RUN else "updated"
            job = next((j for j in jobs if j["page_id"] == decision["page_id"]), {})
            name = f"{job.get('company_name', '?')} - {job.get('job_title', '?')}"
            print(f"{label} {name}: {decision['updates']}")
        if action == "create":
            label = "[dry-run] would add" if DRY_RUN else "added"
            print(f"{label} manual application: {decision['job']['company_name']} - "
                  f"{decision['job']['job_title']}")
        processed.append(msg["id"])

    if not DRY_RUN:
        save_processed(processed)

    for line in review:
        print("review:", line.lstrip("- "))
    verb = "would update" if DRY_RUN else "updated"
    print(f"done: {verb}={updated} added={added} review={len(review)} "
          f"skipped={skipped} failed={failed}")
    write_summary([f"### Email sync: {updated} updated, {added} added, {len(review)} need review, "
                   f"{skipped} skipped, {failed} failed"] + review)


if __name__ == "__main__":
    main()