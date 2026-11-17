"""Notion client for the Job Application Pipeline database.

Env vars (GitHub Secrets): NOTION_API_KEY, NOTION_DATABASE_ID
The database must be shared with your Notion integration.
"""
import os
import time

import requests

API = "https://api.notion.com/v1"
VERSION = "2022-06-28"
MAX_TEXT = 2000  # Notion rich_text limit per block

STATUSES = ["Fetched", "Applied", "Online Examination", "Interview",
            "Offer", "Rejected", "Withdrawn"]


def _headers():
    key = os.getenv("NOTION_API_KEY")
    if not key:
        raise ValueError("NOTION_API_KEY not set")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Notion-Version": VERSION,
    }


def _db_id():
    db = os.getenv("NOTION_DATABASE_ID")
    if not db:
        raise ValueError("NOTION_DATABASE_ID not set")
    return db


def _request(method, url, retries=3, **kwargs):
    """HTTP call with retry on 429 / 5xx. Never logs secrets."""
    for attempt in range(retries):
        resp = requests.request(method, url, headers=_headers(), timeout=30, **kwargs)
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", 2)))
            continue
        if resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        break
    if resp.status_code != 200:
        raise RuntimeError(f"Notion {method} failed {resp.status_code}: {resp.text[:300]}")
    return resp.json()


# ---------- property builders ----------
def _text(v):
    return {"rich_text": [{"text": {"content": str(v)[:MAX_TEXT]}}]}


def build_properties(job: dict) -> dict:
    """Map a flat dict (snake_case keys) to Notion properties. Skips empty values."""
    p = {}
    if job.get("job_title"):
        p["Job Title"] = {"title": [{"text": {"content": job["job_title"][:MAX_TEXT]}}]}

    for key, col in [
        ("company_name", "Company Name"), ("location", "Location"),
        ("job_id", "Job ID"), ("key_matching_skills", "Key Matching Skills"),
        ("why_strong_fit", "Why Strong Fit"), ("salary", "Salary (LPA)"),
        ("recruiter_name", "Recruiter Name"), ("recruiter_outreach", "Recruiter Outreach"),
        ("last_email_event", "Last Email Event"), ("notes", "Notes"),
    ]:
        if job.get(key):
            p[col] = _text(job[key])

    for key, col in [("work_type", "Work Type"), ("source", "Source")]:
        if job.get(key):
            p[col] = {"select": {"name": job[key]}}

    if job.get("status"):
        if job["status"] not in STATUSES:
            raise ValueError(f"Invalid status: {job['status']}")
        p["Status"] = {"status": {"name": job["status"]}}  # status type, not select

    if job.get("apply_link"):
        p["Apply Link"] = {"url": job["apply_link"]}
    if job.get("recruiter_profile"):
        p["Recruiter Profile"] = {"url": job["recruiter_profile"]}
    if job.get("recruiter_email"):
        p["Recruiter Email"] = {"email": job["recruiter_email"]}

    if job.get("match_score") is not None:  # pass 0-100, stored as percent
        p["Resume Match Score"] = {"number": round(job["match_score"] / 100, 4)}

    for key, col in [("posted_date", "Posted Date"), ("applied_at", "Applied At"),
                     ("apply_by", "Apply By")]:
        if job.get(key):
            p[col] = {"date": {"start": job[key]}}  # ISO date or datetime

    if "needs_review" in job:
        p["Needs Review"] = {"checkbox": bool(job["needs_review"])}
    return p


# ---------- read ----------
def _plain(prop, kind):
    items = prop.get(kind, [])
    return "".join(i.get("plain_text", "") for i in items)


def _parse_page(page: dict) -> dict:
    pr = page["properties"]
    return {
        "page_id": page["id"],
        "job_title": _plain(pr.get("Job Title", {}), "title"),
        "company_name": _plain(pr.get("Company Name", {}), "rich_text"),
        "location": _plain(pr.get("Location", {}), "rich_text"),
        "job_id": _plain(pr.get("Job ID", {}), "rich_text"),
        "apply_link": pr.get("Apply Link", {}).get("url"),
        "status": (pr.get("Status", {}).get("status") or {}).get("name"),
        "last_email_event": _plain(pr.get("Last Email Event", {}), "rich_text"),
    }


def query_jobs(statuses=None) -> list[dict]:
    """Return all jobs (optionally only those in given statuses)."""
    payload = {}
    if statuses:
        payload["filter"] = {"or": [
            {"property": "Status", "status": {"equals": s}} for s in statuses]}
    jobs, cursor = [], None
    while True:
        if cursor:
            payload["start_cursor"] = cursor
        data = _request("POST", f"{API}/databases/{_db_id()}/query", json=payload)
        jobs += [_parse_page(pg) for pg in data["results"]]
        if not data.get("has_more"):
            return jobs
        cursor = data["next_cursor"]


# ---------- write ----------
def update_job(page_id: str, updates: dict):
    """Update an existing job (status, last_email_event, needs_review...).
    Deliberately no create function: email-driven runs must never add jobs."""
    props = build_properties(updates)
    if not props:
        return None
    return _request("PATCH", f"{API}/pages/{page_id}", json={"properties": props})