"""Email delivery for account verification and password reset.

SMTP credentials are read only from environment variables. Secrets must never
be hard-coded in the source tree.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def reset_code_debug_enabled() -> bool:
    """Whether a reset code may be shown locally when SMTP is absent."""
    return _env_bool("RESET_CODE_DEBUG", default=False)


def verification_code_debug_enabled() -> bool:
    """Whether a registration verification code may be shown locally."""
    return _env_bool("VERIFICATION_CODE_DEBUG", default=False)


def smtp_is_configured() -> bool:
    host = os.getenv("SMTP_HOST", "").strip()
    from_email = os.getenv("SMTP_FROM_EMAIL", "").strip()
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    return bool(host and from_email and username and password)


def _send_code_email(
    recipient: str,
    code: str,
    *,
    subject: str,
    intro: str,
    expires_minutes: int,
) -> bool:
    """Send one numeric code through the configured SMTP server."""
    host = os.getenv("SMTP_HOST", "").strip()
    from_email = os.getenv("SMTP_FROM_EMAIL", "").strip()
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    if not host or not from_email or not username or not password:
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    use_ssl = _env_bool("SMTP_USE_SSL", default=False)
    use_tls = _env_bool("SMTP_USE_TLS", default=not use_ssl)

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = recipient
    message.set_content(
        f"{intro}\n\n"
        f"کد شما: {code}\n\n"
        f"این کد تا {expires_minutes} دقیقه معتبر است. "
        "اگر شما این درخواست را ثبت نکرده‌اید، این ایمیل را نادیده بگیرید."
    )

    context = ssl.create_default_context()
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as server:
            server.login(username, password)
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.ehlo()
            if use_tls:
                server.starttls(context=context)
                server.ehlo()
            server.login(username, password)
            server.send_message(message)
    return True


def send_email_verification_code(recipient: str, code: str, *, expires_minutes: int = 15) -> bool:
    """Send the one-time code used to verify a newly registered email."""
    return _send_code_email(
        recipient,
        code,
        subject="کد تأیید ایمیل دفترچه هوشمند تغذیه",
        intro="برای تکمیل ساخت حساب، کد زیر را در برنامه وارد کنید.",
        expires_minutes=expires_minutes,
    )


def send_password_reset_code(recipient: str, code: str, *, expires_minutes: int = 15) -> bool:
    """Send a password-reset code. Return False when SMTP is not configured."""
    return _send_code_email(
        recipient,
        code,
        subject="کد بازیابی رمز عبور دفترچه هوشمند تغذیه",
        intro="برای تعیین رمز عبور جدید، کد زیر را در برنامه وارد کنید.",
        expires_minutes=expires_minutes,
    )
