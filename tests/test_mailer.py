import mailer


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=20):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logged_in = None
        self.message = None
        self.started_tls = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def ehlo(self):
        pass

    def starttls(self, context=None):
        self.started_tls = True

    def login(self, username, password):
        self.logged_in = (username, password)

    def send_message(self, message):
        self.message = message


def test_real_smtp_configuration_and_verification_email(monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("SMTP_FROM_EMAIL", "sender@example.com")
    monkeypatch.setenv("SMTP_USE_TLS", "true")
    monkeypatch.setenv("SMTP_USE_SSL", "false")
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)

    assert mailer.smtp_is_configured()
    assert mailer.send_email_verification_code("recipient@example.com", "654321")
    sent = FakeSMTP.instances[-1]
    assert sent.started_tls
    assert sent.logged_in == ("sender@example.com", "app-password")
    assert sent.message["To"] == "recipient@example.com"
    assert "654321" in sent.message.get_content()


def test_smtp_is_not_configured_without_password(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_USERNAME", "sender@example.com")
    monkeypatch.setenv("SMTP_FROM_EMAIL", "sender@example.com")
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    assert not mailer.smtp_is_configured()
