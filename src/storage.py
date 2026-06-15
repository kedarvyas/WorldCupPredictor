"""Optional remote persistence for the paper ledger.

Streamlit Community Cloud runs on an ephemeral filesystem: anything written
at runtime (logged bets, recorded closing odds) is lost on the next reboot
or redeploy. To keep the ledger, we mirror the CSV to a private GitHub gist
when credentials are configured in st.secrets (github_token + gist_id).

With no secrets configured — e.g. running locally on the Mac — every
function here is a no-op and the local CSV stays the single source of truth,
exactly as before. The dashboard pulls once per session and pushes after
each write.

A gist (not a repo commit) is used deliberately: committing to the repo on
every bet would trigger a Streamlit Cloud redeploy. A gist write does not.
"""

GIST_API = "https://api.github.com/gists"
LEDGER_FILENAME = "paper_ledger.csv"
_TIMEOUT = 10


def _config():
    """(token, gist_id) from st.secrets, or None if not configured.
    Guarded so importing/calling never crashes when there are no secrets."""
    try:
        import streamlit as st

        token = st.secrets.get("github_token", "")
        gist_id = st.secrets.get("gist_id", "")
    except Exception:
        return None
    if token and gist_id:
        return token, gist_id
    return None


def enabled() -> bool:
    return _config() is not None


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"}


def pull(local_path) -> bool:
    """Overwrite the local ledger with the gist copy. No-op (False) when
    persistence is off, the gist has no ledger yet, or the call fails."""
    cfg = _config()
    if cfg is None:
        return False
    token, gist_id = cfg
    try:
        import requests

        r = requests.get(f"{GIST_API}/{gist_id}", headers=_headers(token),
                         timeout=_TIMEOUT)
        r.raise_for_status()
        files = r.json().get("files", {})
        if LEDGER_FILENAME not in files:
            return False
        content = files[LEDGER_FILENAME]["content"]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content)
        return True
    except Exception:
        return False


def push(local_path) -> bool:
    """Mirror the local ledger up to the gist. Returns success so the caller
    can warn the user if a bet was saved locally but not persisted."""
    cfg = _config()
    if cfg is None or not local_path.exists():
        return False
    token, gist_id = cfg
    try:
        import requests

        r = requests.patch(
            f"{GIST_API}/{gist_id}", headers=_headers(token),
            json={"files": {LEDGER_FILENAME: {
                "content": local_path.read_text()}}},
            timeout=_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception:
        return False
