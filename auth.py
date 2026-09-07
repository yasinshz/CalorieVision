"""Local password hashing, reset-code hashing and account validation.

The module uses only Python's standard library. Passwords and password-reset
codes are stored as salted PBKDF2 hashes and are never persisted as plain text.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import secrets

PBKDF2_ITERATIONS = 310_000
PASSWORD_SCHEME = "pbkdf2_sha256"
RESET_CODE_ITERATIONS = 120_000
RESET_CODE_SCHEME = "reset_pbkdf2_sha256"
RESET_CODE_LENGTH = 6


class AccountValidationError(ValueError):
    """Raised when account data does not meet the local account rules."""


def normalize_username(username: str) -> str:
    """Return the canonical username used for lookup and uniqueness."""
    return username.strip().lower()


def validate_username(username: str) -> str:
    normalized = normalize_username(username)
    if not 3 <= len(normalized) <= 40:
        raise AccountValidationError("نام کاربری باید بین ۳ تا ۴۰ نویسه باشد.")
    if any(char.isspace() for char in normalized):
        raise AccountValidationError("نام کاربری نباید فاصله داشته باشد.")
    if not re.fullmatch(r"[\w.\-]+", normalized, flags=re.UNICODE):
        raise AccountValidationError(
            "نام کاربری فقط می‌تواند شامل حروف، عدد، نقطه، خط تیره و زیرخط باشد."
        )
    return normalized


def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_email(email: str, *, required: bool = True) -> str:
    normalized = normalize_email(email)
    if not normalized:
        if required:
            raise AccountValidationError("واردکردن ایمیل الزامی است.")
        return ""
    if len(normalized) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", normalized):
        raise AccountValidationError("نشانی ایمیل معتبر نیست.")
    return normalized


def validate_password(password: str) -> None:
    if len(password) < 8:
        raise AccountValidationError("رمز عبور باید حداقل ۸ نویسه باشد.")
    if len(password) > 256:
        raise AccountValidationError("رمز عبور بیش از حد طولانی است.")


def _hash_secret(secret: str, *, scheme: str, iterations: int) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        salt,
        iterations,
    )
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
    hash_b64 = base64.urlsafe_b64encode(derived).decode("ascii")
    return f"{scheme}${iterations}${salt_b64}${hash_b64}"


def _verify_secret(
    secret: str,
    encoded_hash: str,
    *,
    expected_scheme: str,
    min_iterations: int,
    max_iterations: int = 2_000_000,
) -> bool:
    try:
        scheme, iterations_text, salt_b64, expected_b64 = encoded_hash.split("$", 3)
        if scheme != expected_scheme:
            return False
        iterations = int(iterations_text)
        if iterations < min_iterations or iterations > max_iterations:
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(expected_b64.encode("ascii"))
    except (ValueError, TypeError, binascii.Error):
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Hash a password using salted PBKDF2-HMAC-SHA256."""
    validate_password(password)
    return _hash_secret(password, scheme=PASSWORD_SCHEME, iterations=iterations)


def verify_password(password: str, encoded_hash: str) -> bool:
    """Return True only when *password* matches the stored encoded hash."""
    return _verify_secret(
        password,
        encoded_hash,
        expected_scheme=PASSWORD_SCHEME,
        min_iterations=100_000,
    )


def generate_reset_code(length: int = RESET_CODE_LENGTH) -> str:
    """Generate a numeric one-time code suitable for email delivery."""
    if not 6 <= int(length) <= 10:
        raise ValueError("طول کد بازیابی باید بین ۶ تا ۱۰ رقم باشد.")
    upper = 10 ** int(length)
    return f"{secrets.randbelow(upper):0{int(length)}d}"


def hash_reset_code(code: str, *, iterations: int = RESET_CODE_ITERATIONS) -> str:
    normalized = code.strip()
    if not re.fullmatch(r"\d{6,10}", normalized):
        raise AccountValidationError("کد بازیابی باید ۶ تا ۱۰ رقم باشد.")
    return _hash_secret(normalized, scheme=RESET_CODE_SCHEME, iterations=iterations)


def verify_reset_code(code: str, encoded_hash: str) -> bool:
    normalized = code.strip()
    if not re.fullmatch(r"\d{6,10}", normalized):
        return False
    return _verify_secret(
        normalized,
        encoded_hash,
        expected_scheme=RESET_CODE_SCHEME,
        min_iterations=100_000,
    )
