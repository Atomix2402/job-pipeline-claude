import base64
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _decode_body(data: str) -> str:
    """Decode Gmail's URL-safe base64 payload."""
    if not data:
        return ""
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def _walk(part, plain, html):
    """Collect every text/plain and text/html part, however deeply nested."""
    mime = part.get("mimeType", "")
    data = part.get("body", {}).get("data")
    if data:
        if mime == "text/plain":
            plain.append(_decode_body(data))
        elif mime == "text/html":
            html.append(_decode_body(data))
    for child in part.get("parts", []):
        _walk(child, plain, html)


def _html_to_text(html: str):
    """HTML -> readable text, keeping image alt text and link targets
    (marketing-style emails often hide the company/role there)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        img.replace_with(f" {alt} " if alt else " ")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("http"):
            links.append(href.split("?")[0][:200])   # drop tracking query strings
    text = soup.get_text(separator="\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text), links


def _extract_body(payload):
    """Return (body_text, links). Uses the HTML version when the plain-text
    part is missing or just a short stub ("view this email in your browser")."""
    plain, html = [], []
    _walk(payload, plain, html)
    plain_text = "\n".join(plain).strip()
    html_text, links = ("", [])
    if html:
        html_text, links = _html_to_text("\n".join(html))
    body = html_text if len(html_text) > len(plain_text) * 1.3 else plain_text
    unique_links = list(dict.fromkeys(links))[:15]
    return (body or plain_text or html_text), unique_links


def get_email_details(service, msg_id):
    """Fetches and normalizes one Gmail message."""
    message = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()

    payload = message.get("payload", {})
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

    body, links = _extract_body(payload)

    return {
        "message_id": message.get("id"),
        "thread_id": message.get("threadId"),
        "sender": headers.get("from", ""),
        "recipient": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "body": body,
        "links": links,
        "snippet": message.get("snippet", ""),
        "labels": message.get("labelIds", []),
        "internal_date": message.get("internalDate"),
    }


def search_emails(service, hours=24, limit=50):
    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())

    keywords = [
        # Application acknowledgements
        'subject:"your application"',
        'subject:"application received"',
        'subject:"application was sent"',   # LinkedIn
        'subject:"your application to"',    # LinkedIn
        '"thank you for applying"',
        '"we received your application"',

        # Interview and assessment
        'subject:interview',
        'subject:assessment',
        'subject:"coding challenge"',
        'subject:"technical round"',
        'from:hackerrank.com',
        'from:codility.com',
        'from:codesignal.com',

        # Selection and rejection
        'subject:shortlisted',
        'subject:"next steps"',
        '"not moving forward"',
        '"other candidates"',
        '"regret to inform"',

        # Job offers and recruiter outreach
        'subject:"offer letter"',
        'subject:"employment offer"',
        '"talent acquisition"',
        'subject:"job opportunity"',

        # India-focused job portals
        'from:naukri.com',
        'from:hirist.tech',
        'from:iimjobs.com',
        'from:cutshort.io',
        'from:instahyre.com',
        'from:internshala.com',
        'from:foundit.in',
        'from:hirist.com',

        # Recruitment and assessment platforms
        'from:hirepro.in',
        'from:mettl.com',
        'from:superset.com',

        # Global ATS platforms also used by Indian employers
        'from:greenhouse.io',
        'from:lever.co',
        'from:myworkday.com',
        'from:smartrecruiters.com',
        'from:ashbyhq.com',
    ]

    exclusions = [
        '-subject:"jobs recommended for you"',
        '-subject:"job alert"',
        '-subject:"new jobs for you"',
        '-subject:"jobs you may like"',
    ]

    query = f"after:{cutoff} " + "{" + " ".join(keywords) + "} " + " ".join(exclusions)

    messages = []
    page_token = None

    while len(messages) < limit:
        response = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=min(100, limit - len(messages)),
            pageToken=page_token,
            includeSpamTrash=False,
        ).execute()

        messages.extend(response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return messages


def get_gmail_service():
    """Auth from env vars (GitHub Secrets) if set, else local token.json."""
    client_id = os.getenv("GMAIL_CLIENT_ID")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET")
    refresh_token = os.getenv("GMAIL_REFRESH_TOKEN")

    # GitHub Actions path: no files, no browser
    if client_id and client_secret and refresh_token:
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        creds.refresh(Request())
        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    # Local path
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


if __name__ == "__main__":
    # Local testing only: never print email bodies into CI logs
    if os.getenv("GITHUB_ACTIONS"):
        sys.exit("Refusing to print emails in CI.")

    service = get_gmail_service()
    messages = search_emails(service, hours=24, limit=20)
    print(f"\nFound {len(messages)} matching emails.\n")

    for message in messages:
        email = get_email_details(service, message["id"])
        print("=" * 80)
        print("FROM:", email["sender"])
        print("SUBJECT:", email["subject"])
        print("DATE:", email["date"])
        print()
        print(email["body"][:500])
        print()