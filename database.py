"""SQLite persistence for users, meals, nutrition cache and reports."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

DB_PATH = Path(__file__).resolve().parent / "data" / "food_calorie.db"

MEAL_TYPES = (
    "breakfast",
    "morning_snack",
    "lunch",
    "afternoon_snack",
    "dinner",
    "evening_snack",
    "treat",
    "other",
    "snack",  # legacy value retained for existing data
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _meal_table_definition(table_name: str = "meals") -> str:
    allowed = ", ".join(f"'{item}'" for item in MEAL_TYPES)
    return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            meal_type TEXT NOT NULL CHECK (meal_type IN ({allowed})),
            meal_date TEXT NOT NULL,
            meal_time TEXT CHECK (
                meal_time IS NULL OR (
                    length(meal_time) = 5 AND
                    substr(meal_time, 3, 1) = ':' AND
                    substr(meal_time, 1, 2) BETWEEN '00' AND '23' AND
                    substr(meal_time, 4, 2) BETWEEN '00' AND '59'
                )
            ),
            image_hash TEXT,
            predicted_label TEXT,
            total_calories REAL NOT NULL CHECK (total_calories >= 0),
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """


def _upgrade_meals_table(connection: sqlite3.Connection) -> None:
    """Upgrade old meal constraints/columns without deleting existing records."""
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'meals'"
    ).fetchone()
    if table_row is None:
        return

    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(meals)").fetchall()
    }
    table_sql = str(table_row["sql"] or "")
    needs_upgrade = "meal_time" not in columns or any(
        meal_type not in table_sql for meal_type in MEAL_TYPES
    )
    if not needs_upgrade:
        return

    has_meal_time = "meal_time" in columns
    connection.commit()
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TABLE IF EXISTS meals_upgrade")
        connection.execute(_meal_table_definition("meals_upgrade"))
        meal_time_expression = "meal_time" if has_meal_time else "NULL"
        connection.execute(
            f"""
            INSERT INTO meals_upgrade (
                id, user_id, meal_type, meal_date, meal_time, image_hash,
                predicted_label, total_calories, notes, created_at
            )
            SELECT id, user_id, meal_type, meal_date, {meal_time_expression},
                   image_hash, predicted_label, total_calories, notes, created_at
            FROM meals
            """
        )
        connection.execute("DROP TABLE meals")
        connection.execute("ALTER TABLE meals_upgrade RENAME TO meals")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_meals_user_date "
            "ON meals(user_id, meal_date)"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")

    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError("مهاجرت دیتابیس با خطای ارتباط جداول روبه‌رو شد.")


def init_db() -> None:
    """Create or safely upgrade the schema while preserving previous data."""
    with closing(_connect()) as connection:
        connection.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS nutrition_cache (
                query TEXT PRIMARY KEY,
                fdc_id INTEGER,
                description TEXT NOT NULL,
                data_type TEXT,
                calories_per_100g REAL NOT NULL,
                fetched_at TEXT NOT NULL
            );

            -- Legacy table retained so existing MVP data is not destroyed.
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                selected_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                weight_grams REAL NOT NULL,
                calories_per_100g REAL NOT NULL,
                estimated_calories REAL NOT NULL,
                nutrition_description TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                email TEXT COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                email_verified INTEGER NOT NULL DEFAULT 0 CHECK (email_verified IN (0, 1)),
                daily_calorie_goal REAL NOT NULL DEFAULT 2000
                    CHECK (daily_calorie_goal BETWEEN 500 AND 10000),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 5),
                used_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 5),
                used_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            {_meal_table_definition()}

            CREATE TABLE IF NOT EXISTS meal_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meal_id INTEGER NOT NULL,
                food_label TEXT NOT NULL,
                food_name_fa TEXT NOT NULL,
                weight_grams REAL NOT NULL CHECK (weight_grams > 0),
                measure_unit TEXT NOT NULL DEFAULT 'g' CHECK (measure_unit IN ('g', 'ml')),
                calories_per_100g REAL NOT NULL CHECK (calories_per_100g >= 0),
                estimated_calories REAL NOT NULL CHECK (estimated_calories >= 0),
                confidence REAL NOT NULL DEFAULT 0 CHECK (confidence BETWEEN 0 AND 1),
                nutrition_description TEXT,
                source TEXT,
                FOREIGN KEY (meal_id) REFERENCES meals(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_meals_user_date
                ON meals(user_id, meal_date);
            CREATE INDEX IF NOT EXISTS idx_password_reset_user_created
                ON password_reset_tokens(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_password_reset_active
                ON password_reset_tokens(user_id, used_at, expires_at);
            CREATE INDEX IF NOT EXISTS idx_email_verification_user_created
                ON email_verification_tokens(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_email_verification_active
                ON email_verification_tokens(user_id, used_at, expires_at);
            CREATE INDEX IF NOT EXISTS idx_meal_items_meal
                ON meal_items(meal_id);
            CREATE INDEX IF NOT EXISTS idx_meal_items_label
                ON meal_items(food_label);
            """
        )
        user_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(users)").fetchall()
        }
        if "email_verified" not in user_columns:
            # Accounts created by earlier versions are trusted as already verified.
            connection.execute(
                "ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 1"
            )

        meal_item_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(meal_items)").fetchall()
        }
        if "measure_unit" not in meal_item_columns:
            # Existing rows were food-weight records, so preserve them as grams.
            connection.execute(
                "ALTER TABLE meal_items ADD COLUMN measure_unit TEXT NOT NULL DEFAULT 'g'"
            )

        connection.commit()
        _upgrade_meals_table(connection)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def create_user(
    *,
    username: str,
    email: str | None,
    password_hash: str,
    daily_calorie_goal: float = 2000,
    email_verified: bool = True,
) -> int:
    goal = float(daily_calorie_goal)
    if not 500 <= goal <= 10000:
        raise ValueError("هدف کالری باید بین ۵۰۰ تا ۱۰۰۰۰ باشد.")
    normalized_email = (email or "").strip().lower()
    if not normalized_email:
        raise ValueError("واردکردن ایمیل الزامی است.")

    try:
        with closing(_connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (
                    username, email, password_hash, email_verified,
                    daily_calorie_goal, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    normalized_email,
                    password_hash,
                    1 if email_verified else 0,
                    goal,
                    _utc_now(),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        message = str(exc).lower()
        if "username" in message:
            raise ValueError("این نام کاربری قبلاً ثبت شده است.") from exc
        if "email" in message:
            raise ValueError("این ایمیل قبلاً ثبت شده است.") from exc
        raise ValueError("ساخت حساب کاربری ناموفق بود.") from exc


def _public_user(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    result.pop("password_hash", None)
    return result


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT id, username, email, email_verified, daily_calorie_goal, created_at
            FROM users WHERE id = ?
            """,
            (int(user_id),),
        ).fetchone()
    return _public_user(row)


def get_user_for_login(identifier: str) -> dict[str, Any] | None:
    normalized = identifier.strip().lower()
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT id, username, email, password_hash, email_verified,
                   daily_calorie_goal, created_at
            FROM users
            WHERE lower(username) = ? OR lower(COALESCE(email, '')) = ?
            LIMIT 1
            """,
            (normalized, normalized),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    normalized = email.strip().lower()
    if not normalized:
        return None
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT id, username, email, email_verified, daily_calorie_goal, created_at
            FROM users
            WHERE lower(email) = ?
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
    return _public_user(row)


def update_daily_goal(user_id: int, daily_calorie_goal: float) -> None:
    goal = float(daily_calorie_goal)
    if not 500 <= goal <= 10000:
        raise ValueError("هدف کالری باید بین ۵۰۰ تا ۱۰۰۰۰ باشد.")
    with closing(_connect()) as connection:
        cursor = connection.execute(
            "UPDATE users SET daily_calorie_goal = ? WHERE id = ?",
            (goal, int(user_id)),
        )
        if cursor.rowcount != 1:
            raise ValueError("کاربر پیدا نشد.")
        connection.commit()


def get_user_credentials(user_id: int) -> dict[str, Any] | None:
    """Return credentials for password verification inside the local app only."""
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT id, username, email, password_hash FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
    return dict(row) if row else None


def update_account_profile(user_id: int, *, username: str, email: str | None) -> None:
    """Update username/email while enforcing required email and uniqueness."""
    normalized_email = (email or "").strip().lower()
    if not normalized_email:
        raise ValueError("واردکردن ایمیل الزامی است.")
    try:
        with closing(_connect()) as connection:
            cursor = connection.execute(
                "UPDATE users SET username = ?, email = ? WHERE id = ?",
                (username, normalized_email, int(user_id)),
            )
            if cursor.rowcount != 1:
                raise ValueError("کاربر پیدا نشد.")
            connection.commit()
    except sqlite3.IntegrityError as exc:
        message = str(exc).lower()
        if "username" in message:
            raise ValueError("این نام کاربری قبلاً ثبت شده است.") from exc
        if "email" in message:
            raise ValueError("این ایمیل قبلاً ثبت شده است.") from exc
        raise ValueError("به‌روزرسانی اطلاعات حساب ناموفق بود.") from exc


def update_password_hash(user_id: int, password_hash: str) -> None:
    if not password_hash:
        raise ValueError("هش رمز عبور معتبر نیست.")
    with closing(_connect()) as connection:
        cursor = connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, int(user_id)),
        )
        if cursor.rowcount != 1:
            raise ValueError("کاربر پیدا نشد.")
        connection.commit()


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------


def create_email_verification_token(
    user_id: int,
    token_hash: str,
    *,
    expires_minutes: int = 15,
    cooldown_seconds: int = 60,
) -> int:
    """Create one active verification token and invalidate older unused ones."""
    if not token_hash:
        raise ValueError("هش کد تأیید معتبر نیست.")
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=int(expires_minutes))

    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        latest = connection.execute(
            """
            SELECT created_at FROM email_verification_tokens
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (int(user_id),),
        ).fetchone()
        if latest and cooldown_seconds > 0:
            try:
                latest_at = datetime.fromisoformat(str(latest["created_at"]))
            except ValueError:
                latest_at = now - timedelta(days=1)
            if (now - latest_at).total_seconds() < cooldown_seconds:
                connection.rollback()
                raise ValueError("برای ارسال دوباره کد کمی صبر کنید.")

        connection.execute(
            """
            UPDATE email_verification_tokens
            SET used_at = ?
            WHERE user_id = ? AND used_at IS NULL
            """,
            (now.isoformat(), int(user_id)),
        )
        cursor = connection.execute(
            """
            INSERT INTO email_verification_tokens (
                user_id, token_hash, expires_at, attempts, used_at, created_at
            ) VALUES (?, ?, ?, 0, NULL, ?)
            """,
            (int(user_id), token_hash, expires_at.isoformat(), now.isoformat()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_active_email_verification(email: str) -> dict[str, Any] | None:
    normalized = email.strip().lower()
    if not normalized:
        return None
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT ev.id, ev.user_id, ev.token_hash, ev.expires_at,
                   ev.attempts, ev.created_at, u.email, u.email_verified
            FROM email_verification_tokens AS ev
            JOIN users AS u ON u.id = ev.user_id
            WHERE lower(u.email) = ?
              AND u.email_verified = 0
              AND ev.used_at IS NULL
              AND ev.expires_at > ?
              AND ev.attempts < 5
            ORDER BY ev.created_at DESC
            LIMIT 1
            """,
            (normalized, _utc_now()),
        ).fetchone()
    return dict(row) if row else None


def record_email_verification_failure(verification_id: int) -> None:
    with closing(_connect()) as connection:
        connection.execute(
            """
            UPDATE email_verification_tokens
            SET attempts = CASE WHEN attempts < 5 THEN attempts + 1 ELSE 5 END
            WHERE id = ? AND used_at IS NULL
            """,
            (int(verification_id),),
        )
        connection.commit()


def complete_email_verification(verification_id: int, user_id: int) -> bool:
    """Atomically consume a valid code and mark the user's email as verified."""
    now = _utc_now()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        token = connection.execute(
            """
            SELECT id FROM email_verification_tokens
            WHERE id = ? AND user_id = ? AND used_at IS NULL
              AND expires_at > ? AND attempts < 5
            """,
            (int(verification_id), int(user_id), now),
        ).fetchone()
        if token is None:
            connection.rollback()
            return False

        cursor = connection.execute(
            "UPDATE users SET email_verified = 1 WHERE id = ?",
            (int(user_id),),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        connection.execute(
            """
            UPDATE email_verification_tokens
            SET used_at = ?
            WHERE user_id = ? AND used_at IS NULL
            """,
            (now, int(user_id)),
        )
        connection.commit()
        return True


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------


def create_password_reset_token(
    user_id: int,
    token_hash: str,
    *,
    expires_minutes: int = 15,
    cooldown_seconds: int = 60,
) -> int:
    """Create one active reset token and invalidate older unused tokens."""
    if not token_hash:
        raise ValueError("هش کد بازیابی معتبر نیست.")
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=int(expires_minutes))

    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        latest = connection.execute(
            """
            SELECT created_at FROM password_reset_tokens
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (int(user_id),),
        ).fetchone()
        if latest and cooldown_seconds > 0:
            try:
                latest_at = datetime.fromisoformat(str(latest["created_at"]))
            except ValueError:
                latest_at = now - timedelta(days=1)
            if (now - latest_at).total_seconds() < cooldown_seconds:
                connection.rollback()
                raise ValueError("برای درخواست کد جدید کمی صبر کنید.")

        connection.execute(
            """
            UPDATE password_reset_tokens
            SET used_at = ?
            WHERE user_id = ? AND used_at IS NULL
            """,
            (now.isoformat(), int(user_id)),
        )
        cursor = connection.execute(
            """
            INSERT INTO password_reset_tokens (
                user_id, token_hash, expires_at, attempts, used_at, created_at
            ) VALUES (?, ?, ?, 0, NULL, ?)
            """,
            (int(user_id), token_hash, expires_at.isoformat(), now.isoformat()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_active_password_reset(email: str) -> dict[str, Any] | None:
    normalized = email.strip().lower()
    if not normalized:
        return None
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT pr.id, pr.user_id, pr.token_hash, pr.expires_at,
                   pr.attempts, pr.created_at, u.email
            FROM password_reset_tokens AS pr
            JOIN users AS u ON u.id = pr.user_id
            WHERE lower(u.email) = ?
              AND pr.used_at IS NULL
              AND pr.expires_at > ?
              AND pr.attempts < 5
            ORDER BY pr.created_at DESC
            LIMIT 1
            """,
            (normalized, _utc_now()),
        ).fetchone()
    return dict(row) if row else None


def record_password_reset_failure(reset_id: int) -> None:
    with closing(_connect()) as connection:
        connection.execute(
            """
            UPDATE password_reset_tokens
            SET attempts = CASE WHEN attempts < 5 THEN attempts + 1 ELSE 5 END
            WHERE id = ? AND used_at IS NULL
            """,
            (int(reset_id),),
        )
        connection.commit()


def complete_password_reset(reset_id: int, user_id: int, password_hash: str) -> bool:
    """Atomically consume an active reset token and replace the password hash."""
    if not password_hash:
        raise ValueError("هش رمز عبور معتبر نیست.")
    now = _utc_now()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        token = connection.execute(
            """
            SELECT id FROM password_reset_tokens
            WHERE id = ? AND user_id = ? AND used_at IS NULL
              AND expires_at > ? AND attempts < 5
            """,
            (int(reset_id), int(user_id), now),
        ).fetchone()
        if token is None:
            connection.rollback()
            return False

        cursor = connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, int(user_id)),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        connection.execute(
            """
            UPDATE password_reset_tokens
            SET used_at = ?
            WHERE user_id = ? AND used_at IS NULL
            """,
            (now, int(user_id)),
        )
        connection.commit()
        return True


# ---------------------------------------------------------------------------
# Nutrition cache
# ---------------------------------------------------------------------------


def get_cached_nutrition(query: str) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM nutrition_cache WHERE query = ?",
            (query,),
        ).fetchone()
    return dict(row) if row else None


def save_cached_nutrition(
    *,
    query: str,
    fdc_id: int | None,
    description: str,
    data_type: str | None,
    calories_per_100g: float,
) -> None:
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO nutrition_cache (
                query, fdc_id, description, data_type, calories_per_100g, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(query) DO UPDATE SET
                fdc_id = excluded.fdc_id,
                description = excluded.description,
                data_type = excluded.data_type,
                calories_per_100g = excluded.calories_per_100g,
                fetched_at = excluded.fetched_at
            """,
            (
                query,
                fdc_id,
                description,
                data_type,
                float(calories_per_100g),
                _utc_now(),
            ),
        )
        connection.commit()


# ---------------------------------------------------------------------------
# Meals
# ---------------------------------------------------------------------------


def _date_text(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ValueError("تاریخ وعده معتبر نیست.") from exc


def _time_text(value: time | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, time):
        return value.strftime("%H:%M")
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%H:%M").strftime("%H:%M")
    except ValueError as exc:
        raise ValueError("زمان وعده معتبر نیست.") from exc


def create_meal(
    *,
    user_id: int,
    meal_type: str,
    meal_date: date | str,
    items: Iterable[Mapping[str, Any]],
    meal_time: time | str | None = None,
    image_hash: str | None = None,
    predicted_label: str | None = None,
    notes: str | None = None,
) -> int:
    if meal_type not in MEAL_TYPES:
        raise ValueError("نوع وعده معتبر نیست.")

    normalized_items: list[dict[str, Any]] = []
    for raw in items:
        weight = float(raw["weight_grams"])
        unit = str(raw.get("measure_unit", "g")).strip().lower()
        per_100 = float(raw["calories_per_100g"])
        if unit not in {"g", "ml"}:
            raise ValueError("واحد مقدار مصرف باید گرم یا میلی‌لیتر باشد.")
        if weight <= 0 or per_100 < 0:
            raise ValueError("مقدار مصرف و کالری اقلام باید معتبر باشند.")
        estimated = weight * per_100 / 100.0
        confidence = max(0.0, min(float(raw.get("confidence", 0.0)), 1.0))
        normalized_items.append(
            {
                "food_label": str(raw["food_label"]),
                "food_name_fa": str(raw["food_name_fa"]),
                "weight_grams": weight,
                "measure_unit": unit,
                "calories_per_100g": per_100,
                "estimated_calories": estimated,
                "confidence": confidence,
                "nutrition_description": raw.get("nutrition_description"),
                "source": raw.get("source"),
            }
        )

    if not normalized_items:
        raise ValueError("وعده باید حداقل یک قلم غذا داشته باشد.")

    total_calories = sum(item["estimated_calories"] for item in normalized_items)
    date_value = _date_text(meal_date)
    time_value = _time_text(meal_time)

    with closing(_connect()) as connection:
        try:
            cursor = connection.execute(
                """
                INSERT INTO meals (
                    user_id, meal_type, meal_date, meal_time, image_hash,
                    predicted_label, total_calories, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(user_id),
                    meal_type,
                    date_value,
                    time_value,
                    image_hash,
                    predicted_label,
                    total_calories,
                    (notes or "").strip() or None,
                    _utc_now(),
                ),
            )
            meal_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO meal_items (
                    meal_id, food_label, food_name_fa, weight_grams, measure_unit,
                    calories_per_100g, estimated_calories, confidence,
                    nutrition_description, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        meal_id,
                        item["food_label"],
                        item["food_name_fa"],
                        item["weight_grams"],
                        item["measure_unit"],
                        item["calories_per_100g"],
                        item["estimated_calories"],
                        item["confidence"],
                        item["nutrition_description"],
                        item["source"],
                    )
                    for item in normalized_items
                ],
            )
            connection.commit()
            return meal_id
        except Exception:
            connection.rollback()
            raise


def get_meals_between(
    user_id: int,
    start_date: date | str,
    end_date: date | str,
) -> list[dict[str, Any]]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    if end < start:
        raise ValueError("تاریخ پایان نباید قبل از تاریخ شروع باشد.")

    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT m.id, m.meal_type, m.meal_date, m.meal_time,
                   m.total_calories, m.notes, m.predicted_label, m.created_at,
                   COUNT(mi.id) AS item_count
            FROM meals AS m
            LEFT JOIN meal_items AS mi ON mi.meal_id = m.id
            WHERE m.user_id = ? AND m.meal_date BETWEEN ? AND ?
            GROUP BY m.id
            ORDER BY m.meal_date DESC,
                     COALESCE(m.meal_time, '23:59') DESC,
                     m.id DESC
            """,
            (int(user_id), start, end),
        ).fetchall()
    return [dict(row) for row in rows]


def get_meals_for_date(user_id: int, meal_date: date | str) -> list[dict[str, Any]]:
    value = _date_text(meal_date)
    meals = get_meals_between(user_id, value, value)
    return sorted(
        meals,
        key=lambda item: (item.get("meal_time") or "99:99", int(item["id"])),
    )


def get_meal_items(user_id: int, meal_id: int) -> list[dict[str, Any]]:
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT mi.id, mi.food_label, mi.food_name_fa, mi.weight_grams, mi.measure_unit,
                   mi.calories_per_100g, mi.estimated_calories, mi.confidence,
                   mi.nutrition_description, mi.source
            FROM meal_items AS mi
            JOIN meals AS m ON m.id = mi.meal_id
            WHERE mi.meal_id = ? AND m.user_id = ?
            ORDER BY mi.id
            """,
            (int(meal_id), int(user_id)),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_meal(user_id: int, meal_id: int) -> bool:
    with closing(_connect()) as connection:
        cursor = connection.execute(
            "DELETE FROM meals WHERE id = ? AND user_id = ?",
            (int(meal_id), int(user_id)),
        )
        connection.commit()
        return cursor.rowcount == 1


def daily_summary(user_id: int, meal_date: date | str) -> dict[str, Any]:
    value = _date_text(meal_date)
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS meal_count,
                   COALESCE(SUM(total_calories), 0) AS total_calories
            FROM meals
            WHERE user_id = ? AND meal_date = ?
            """,
            (int(user_id), value),
        ).fetchone()
        item_row = connection.execute(
            """
            SELECT COUNT(mi.id) AS item_count
            FROM meal_items AS mi
            JOIN meals AS m ON m.id = mi.meal_id
            WHERE m.user_id = ? AND m.meal_date = ?
            """,
            (int(user_id), value),
        ).fetchone()
    return {
        "meal_count": int(row["meal_count"]),
        "item_count": int(item_row["item_count"]),
        "total_calories": float(row["total_calories"]),
    }


def daily_totals(
    user_id: int,
    start_date: date | str,
    end_date: date | str,
) -> list[dict[str, Any]]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT meal_date,
                   COUNT(*) AS meal_count,
                   SUM(total_calories) AS total_calories
            FROM meals
            WHERE user_id = ? AND meal_date BETWEEN ? AND ?
            GROUP BY meal_date
            ORDER BY meal_date
            """,
            (int(user_id), start, end),
        ).fetchall()
    return [dict(row) for row in rows]


def meal_type_totals(
    user_id: int,
    start_date: date | str,
    end_date: date | str,
) -> list[dict[str, Any]]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT meal_type,
                   COUNT(*) AS meal_count,
                   SUM(total_calories) AS total_calories
            FROM meals
            WHERE user_id = ? AND meal_date BETWEEN ? AND ?
            GROUP BY meal_type
            ORDER BY total_calories DESC
            """,
            (int(user_id), start, end),
        ).fetchall()
    return [dict(row) for row in rows]


def top_foods(
    user_id: int,
    start_date: date | str,
    end_date: date | str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    safe_limit = max(1, min(int(limit), 20))
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT mi.food_label, mi.food_name_fa,
                   COUNT(*) AS times_consumed,
                   SUM(mi.estimated_calories) AS total_calories
            FROM meal_items AS mi
            JOIN meals AS m ON m.id = mi.meal_id
            WHERE m.user_id = ? AND m.meal_date BETWEEN ? AND ?
            GROUP BY mi.food_label, mi.food_name_fa
            ORDER BY times_consumed DESC, total_calories DESC
            LIMIT ?
            """,
            (int(user_id), start, end, safe_limit),
        ).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Backward-compatible helpers from the original MVP
# ---------------------------------------------------------------------------


def save_analysis(
    *,
    image_hash: str,
    predicted_label: str,
    selected_label: str,
    confidence: float,
    weight_grams: float,
    calories_per_100g: float,
    estimated_calories: float,
    nutrition_description: str | None,
) -> int:
    with closing(_connect()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                image_hash, predicted_label, selected_label, confidence,
                weight_grams, calories_per_100g, estimated_calories,
                nutrition_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_hash,
                predicted_label,
                selected_label,
                confidence,
                weight_grams,
                calories_per_100g,
                estimated_calories,
                nutrition_description,
                _utc_now(),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def recent_analyses(limit: int = 10) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 100))
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT id, selected_label, confidence, weight_grams,
                   estimated_calories, created_at
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [dict(row) for row in rows]
