import hashlib
import json
import os
import secrets
from pathlib import Path


USER_STORE = Path(__file__).with_name(".research_users.json")


def _load_users() -> dict:
    if not USER_STORE.exists():
        return {}
    try:
        return json.loads(USER_STORE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_users(users: dict) -> None:
    USER_STORE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"


def _password_matches(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split("$", 1)
        candidate = _hash_password(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
        return secrets.compare_digest(candidate, digest_hex)
    except (ValueError, TypeError):
        return False


def signup(name: str, email: str, password: str) -> tuple[bool, str]:
    email = email.strip().lower()
    name = name.strip()
    if not name or not email or not password:
        return False, "Please complete all fields."
    if "@" not in email:
        return False, "Enter a valid email address."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    users = _load_users()
    if email in users:
        return False, "An account with this email already exists."
    users[email] = {"name": name, "password": _hash_password(password)}
    _save_users(users)
    return True, "Account created. You can now log in."


def login(email: str, password: str) -> tuple[bool, str, str | None]:
    email = email.strip().lower()
    user = _load_users().get(email)
    if not user or not _password_matches(password, user.get("password", "")):
        return False, "Incorrect email or password.", None
    return True, "Welcome back.", user.get("name")


def logout() -> None:
    os.environ.pop("RESEARCH_DESK_USER", None)
