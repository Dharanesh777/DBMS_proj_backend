import os
import json
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from dotenv import load_dotenv

# Load .env from project root
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env")
load_dotenv(dotenv_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_FILE = os.path.join(BASE_DIR, "token.json")
CREDENTIALS_FILE = os.path.join(BASE_DIR, "credentials.json")

import hashlib
import base64
import secrets

# The verifier lives only for the lifetime of a single auth flow (auth_url -> callback),
# regenerated per call to get_auth_url(). Not safe under concurrent overlapping flows,
# but this app only ever runs one flow at a time.
_pending_verifier = None

def get_flow():
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    print(f"\n[DEBUG] Flow using Redirect URI: {redirect_uri}\n")
    return Flow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )

def get_auth_url():
    global _pending_verifier
    _pending_verifier = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(_pending_verifier.encode()).digest()
    ).decode().replace("=", "")

    flow = get_flow()
    # Manually pass the challenge to Google
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        code_challenge=code_challenge,
        code_challenge_method="S256"
    )
    return auth_url

def exchange_code_for_token(code: str):
    if _pending_verifier is None:
        raise RuntimeError("No pending auth flow — call get_auth_url() first.")
    flow = get_flow()
    flow.fetch_token(code=code, code_verifier=_pending_verifier)
    creds = flow.credentials
    # Save token to file so you don't need to login again
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    # Auto-refresh if expired
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds