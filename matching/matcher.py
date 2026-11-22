"""Interpret a recruitment email and match it to an EXISTING Notion job.

Two small LLM tasks, run on different providers:
  read     (Groq first)       -> which tracked job does this email refer to?
  classify (OpenRouter first) -> what recruitment event is it?

Never creates jobs. Prefers false negatives: if unsure it flags for review and
does not change the status.
"""
import re
from datetime import datetime, timezone

from llm.provider import complete_json

MATCH_CONFIDENCE = 0.8
EVENT_CONFIDENCE = 0.7
BODY_CHARS = 1500      # text sent to the LLM
EVIDENCE_CHARS = 4000  # text scanned locally for the company name

# Lifecycle order. Rejected/Withdrawn are terminal.
RANK = {"Fetched": 0, "Applied": 1, "Online Examination": 2, "Interview": 3, "Offer": 4}
TERMINAL = {"Rejected", "Withdrawn"}

EVENT_TO_STATUS = {
    "acknowledgement": "Applied",
    "assessment": "Online Examination",
    "interview": "Interview",
    "offer": "Offer",
    "rejection": "Rejected",
    "info_request": None,   # logged, status unchanged
    "irrelevant": None,
}

COMPANY_NOISE = {"pvt", "ltd", "limited", "private", "inc", "llp", "india", "technologies",
                 "technology", "solutions", "services", "software", "systems", "it", "the"}

READ_SYSTEM = (
    "You read one job-application email and decide which of the candidate's tracked jobs "
    "it refers to. Reply with JSON only. Never guess: if it does not clearly refer to one "
    "listed job, use matched_index null."
)
CLASSIFY_SYSTEM = (
    "You classify what a job-application email means for the candidate. Reply with JSON only."
)


def _short_url(match) -> str:
    """Keep only the readable last path segment of a link (job slugs often name the
    company and role); drop domains, query strings and tracking tokens."""
    url = match.group(0).split("?")[0].rstrip("/.,)")
    return f"[link:{url.rsplit('/', 1)[-1][:80]}]"


def _redact(text: str) -> str:
    """Shorten URLs, strip phone numbers and long digit runs (OTPs, IDs) before an LLM."""
    text = re.sub(r"https?://\S+", _short_url, text)
    text = re.sub(r"\+?\d[\d\s\-()]{8,}\d", "[number]", text)
    return text


def _email_block(email: dict) -> str:
    slugs = [u.rsplit("/", 1)[-1][:80] for u in email.get("links", [])[:5]]
    return (f"From: {email['sender']}\nSubject: {email['subject']}\n"
            f"Body: {_redact(email['body'])[:BODY_CHARS]}\n"
            f"Link names: {', '.join(slugs)}")


def _read_prompt(email: dict, jobs: list[dict]) -> str:
    listing = "\n".join(
        f'{i}. company="{j["company_name"]}" | title="{j["job_title"]}" | job_id="{j["job_id"]}"'
        for i, j in enumerate(jobs))
    return f"""Tracked jobs:
{listing}

Email:
{_email_block(email)}

Return JSON:
{{"matched_index": <index of the tracked job, or null>,
  "confidence": <0.0-1.0 that matched_index is the right job>,
  "company": "company named in the email",
  "role": "role named in the email or empty",
  "reason": "one short sentence"}}"""


def _classify_prompt(email: dict) -> str:
    return f"""Email:
{_email_block(email)}

Return JSON:
{{"event": "acknowledgement|assessment|interview|offer|rejection|info_request|irrelevant",
  "confidence": <0.0-1.0>,
  "reason": "one short sentence"}}

Rules: job alerts, newsletters and recommendations are "irrelevant". A coding test or
online assessment is "assessment". Interview invitations and recruiter screening calls
are "interview". "acknowledgement" means the application was received or submitted."""


def _company_key(name: str) -> str:
    tokens = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if t not in COMPANY_NOISE]
    return tokens[0] if tokens else ""


def _company_evidence(job: dict, email: dict) -> bool:
    """Deterministic guard: the company must appear in sender/subject/body."""
    key = _company_key(job["company_name"])
    if not key:
        return False
    haystack = " ".join([email["sender"], email["subject"], email.get("snippet", ""),
                         email["body"][:EVIDENCE_CHARS], " ".join(email.get("links", []))]).lower()
    return key in haystack


def _email_date(email: dict) -> str:
    ms = email.get("internal_date")
    if ms:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).date().isoformat()
    return datetime.now(timezone.utc).date().isoformat()


def _can_move(current: str, new: str) -> bool:
    if current in TERMINAL:
        return False
    if new == "Rejected":
        return True
    return RANK.get(new, -1) > RANK.get(current, 99)


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---- manually applied jobs that are not in Notion yet ----
APPLICATION_RE = re.compile(
    r"application (was )?(sent|received|submitted)|thank(s| you) for applying|"
    r"successfully applied|we('ve| have)? received your application|"
    r"your application (to|for|has been)", re.I)
ALERT_RE = re.compile(r"job alert|jobs? (recommended|you may like|for you)|new jobs", re.I)

EXTRACT_SYSTEM = (
    "You extract details from a job-application confirmation email. Reply with JSON only. "
    "Never guess: leave a field empty if the email does not state it."
)


def _looks_like_application(email: dict) -> bool:
    head = f"{email['subject']} {email.get('snippet', '')} {email['body'][:800]}"
    return bool(APPLICATION_RE.search(head)) and not ALERT_RE.search(email["subject"])


def _extract_prompt(email: dict) -> str:
    return f"""Email:
{_email_block(email)}

Return JSON:
{{"is_application_confirmation": <true only if it confirms the candidate applied to a specific job>,
  "company": "company the candidate applied to",
  "role": "job title",
  "location": "job location if stated, else empty",
  "confidence": <0.0-1.0>}}"""


def _source_from(email: dict) -> str:
    sender = email["sender"].lower()
    # names must match the Notion Source select options exactly
    for needle, option in (("linkedin", "LinkedIn"), ("naukri", "Naukri"), ("indeed", "Indeed"),
                           ("internshala", "Internshala"), ("cutshort", "Cutshort")):
        if needle in sender:
            return option
    return "Other"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _new_application(email: dict, jobs: list[dict]) -> dict:
    """A confirmation email for a job that is NOT tracked yet -> create a manual entry."""
    if not _looks_like_application(email):
        return {"action": "skip", "reason": "no tracked company mentioned"}

    read = complete_json(EXTRACT_SYSTEM, _extract_prompt(email), task="read")
    company = str(read.get("company") or "").strip()
    role = str(read.get("role") or "").strip()
    conf = _num(read.get("confidence"))
    review = {"action": "review", "page_id": None, "updates": {},
              "reason": f"possible new application, unclear details ({conf:.2f})"}

    if not read.get("is_application_confirmation") or len(company) < 2 or len(role) < 2 \
            or conf < MATCH_CONFIDENCE:
        return review

    key = _company_key(company)
    haystack = (f"{email['sender']} {email['subject']} {email.get('snippet', '')} "
                f"{email['body'][:EVIDENCE_CHARS]} {' '.join(email.get('links', []))}").lower()
    if not key or key not in haystack:     # the company must really appear in the email
        return review

    for j in jobs:                         # already tracked under any status -> nothing to add
        if _company_key(j["company_name"]) == key and _norm(j["job_title"]) == _norm(role):
            return {"action": "skip", "reason": "already tracked"}

    link = next((l for l in email.get("links", []) if "/jobs/view/" in l or "job-listings" in l), "")
    date = _email_date(email)
    job = {
        "job_title": role, "company_name": company, "location": str(read.get("location") or ""),
        "source": _source_from(email), "status": "Applied", "applied_at": date,
        "last_email_event": f"acknowledgement ({date})",
        "job_id": f"manual-{email.get('thread_id') or email.get('message_id')}",
        "apply_link": link,
        "notes": f"Added automatically from a confirmation email on {date}. "
                 "Applied manually; details not verified.",
        "needs_review": True,
    }
    return {"action": "create", "job": job, "reason": f"manual application: {company} - {role}"}


def process_email(email: dict, jobs: list[dict]) -> dict:
    """Return one decision dict:
       {"action": "update"|"review"|"skip", "page_id", "updates", "reason"}
    """
    # Cheap local filter first: only jobs whose company appears in the email.
    # No match means no LLM call at all (saves quota and time).
    candidates = [j for j in jobs
                  if j["status"] not in TERMINAL and _company_evidence(j, email)]
    if not candidates:
        return _new_application(email, jobs)

    # Task 1 (read): which job?
    read = complete_json(READ_SYSTEM, _read_prompt(email, candidates), task="read")
    idx = read.get("matched_index")
    match_conf = _num(read.get("confidence"))
    job = candidates[idx] if isinstance(idx, int) and 0 <= idx < len(candidates) else None

    if not job or match_conf < MATCH_CONFIDENCE:
        return {
            "action": "review",
            "page_id": job["page_id"] if job else None,
            "updates": {"needs_review": True} if job else {},
            "reason": f"unclear job match ({match_conf:.2f}); company: {read.get('company', '')}",
        }

    # Task 2 (classify): what happened?
    cls = complete_json(CLASSIFY_SYSTEM, _classify_prompt(email), task="classify")
    event = cls.get("event", "irrelevant")
    event_conf = _num(cls.get("confidence"))

    if event not in EVENT_TO_STATUS or event == "irrelevant":
        return {"action": "skip", "reason": cls.get("reason", "irrelevant")}

    if event_conf < EVENT_CONFIDENCE:
        return {"action": "review", "page_id": job["page_id"],
                "updates": {"needs_review": True},
                "reason": f"{event} but low confidence ({event_conf:.2f})"}

    new_status = EVENT_TO_STATUS[event]
    updates = {"last_email_event": f"{event} ({_email_date(email)})"}

    if new_status and _can_move(job["status"], new_status):
        updates["status"] = new_status
        if new_status == "Applied":
            updates["applied_at"] = _email_date(email)
    elif event == "info_request":
        updates["needs_review"] = True   # surfaces requests that need your action

    if updates.get("last_email_event") == job.get("last_email_event") and "status" not in updates:
        return {"action": "skip", "reason": "already recorded"}

    return {"action": "update", "page_id": job["page_id"], "updates": updates,
            "reason": cls.get("reason", "")}