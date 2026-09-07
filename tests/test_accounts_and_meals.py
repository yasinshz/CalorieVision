from datetime import date, timedelta

import database
from auth import (
    generate_reset_code,
    hash_password,
    hash_reset_code,
    verify_password,
    verify_reset_code,
)


def use_temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_food_calorie.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    return db_path


def test_password_hash_round_trip():
    encoded = hash_password("strong-pass-123", iterations=100_000)
    assert "strong-pass-123" not in encoded
    assert verify_password("strong-pass-123", encoded)
    assert not verify_password("wrong-password", encoded)


def test_user_meal_and_reports_are_isolated(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)

    user_one = database.create_user(
        username="user_one",
        email="one@example.com",
        password_hash="test-hash",
        daily_calorie_goal=2100,
    )
    user_two = database.create_user(
        username="user_two",
        email="two@example.com",
        password_hash="test-hash",
        daily_calorie_goal=1800,
    )

    meal_day = date(2026, 8, 6)
    meal_id = database.create_meal(
        user_id=user_one,
        meal_type="lunch",
        meal_date=meal_day,
        items=[
            {
                "food_label": "cooked_white_rice",
                "food_name_fa": "برنج سفید پخته",
                "weight_grams": 200,
                "calories_per_100g": 130,
                "confidence": 0,
                "source": "cache",
            },
            {
                "food_label": "grilled_chicken_breast",
                "food_name_fa": "سینه مرغ گریل‌شده",
                "weight_grams": 150,
                "calories_per_100g": 165,
                "confidence": 0.75,
                "source": "usda",
            },
        ],
    )

    summary = database.daily_summary(user_one, meal_day)
    assert summary["meal_count"] == 1
    assert summary["item_count"] == 2
    assert summary["total_calories"] == 507.5

    assert database.daily_summary(user_two, meal_day)["total_calories"] == 0
    assert len(database.get_meal_items(user_one, meal_id)) == 2
    assert database.get_meal_items(user_two, meal_id) == []

    totals = database.daily_totals(user_one, meal_day, meal_day + timedelta(days=6))
    assert totals[0]["meal_date"] == meal_day.isoformat()
    assert totals[0]["total_calories"] == 507.5

    foods = database.top_foods(user_one, meal_day, meal_day + timedelta(days=6))
    assert len(foods) == 2
    assert foods[0]["times_consumed"] == 1


def test_delete_meal_cascades_items(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)
    user_id = database.create_user(
        username="delete_user",
        email="delete@example.com",
        password_hash="test-hash",
    )
    meal_id = database.create_meal(
        user_id=user_id,
        meal_type="snack",
        meal_date="2026-08-06",
        items=[
            {
                "food_label": "apple_pie",
                "food_name_fa": "پای سیب",
                "weight_grams": 100,
                "calories_per_100g": 250,
            }
        ],
    )

    assert database.delete_meal(user_id, meal_id)
    assert database.get_meal_items(user_id, meal_id) == []
    assert database.get_meals_for_date(user_id, "2026-08-06") == []


def test_extended_meal_types_and_time_are_saved(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)
    user_id = database.create_user(
        username="meal_types_user",
        email="mealtypes@example.com",
        password_hash="test-hash",
    )

    expected_types = [
        "breakfast",
        "morning_snack",
        "lunch",
        "afternoon_snack",
        "dinner",
        "evening_snack",
        "treat",
        "other",
    ]
    for index, meal_type in enumerate(expected_types):
        database.create_meal(
            user_id=user_id,
            meal_type=meal_type,
            meal_date="2026-08-06",
            meal_time=f"{8 + index:02d}:15",
            items=[
                {
                    "food_label": f"item_{index}",
                    "food_name_fa": f"خوراکی {index}",
                    "weight_grams": 100,
                    "calories_per_100g": 100 + index,
                }
            ],
        )

    meals = database.get_meals_for_date(user_id, "2026-08-06")
    assert [meal["meal_type"] for meal in meals] == expected_types
    assert meals[0]["meal_time"] == "08:15"
    assert meals[-1]["meal_time"] == "15:15"


def test_profile_and_password_can_be_updated(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)
    user_one = database.create_user(
        username="old_name",
        email="old@example.com",
        password_hash="old-hash",
    )
    database.create_user(
        username="reserved_name",
        email="reserved@example.com",
        password_hash="other-hash",
    )

    database.update_account_profile(
        user_one,
        username="new_name",
        email="new@example.com",
    )
    updated = database.get_user_by_id(user_one)
    assert updated["username"] == "new_name"
    assert updated["email"] == "new@example.com"

    database.update_password_hash(user_one, "new-hash")
    credentials = database.get_user_credentials(user_one)
    assert credentials["password_hash"] == "new-hash"

    try:
        database.update_account_profile(
            user_one,
            username="reserved_name",
            email="new@example.com",
        )
    except ValueError as exc:
        assert "نام کاربری" in str(exc)
    else:
        raise AssertionError("duplicate username should be rejected")



def test_email_is_required_for_new_accounts_and_profile(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)

    try:
        database.create_user(
            username="missing_email",
            email=None,
            password_hash="test-hash",
        )
    except ValueError as exc:
        assert "ایمیل" in str(exc)
    else:
        raise AssertionError("email must be required for registration")

    user_id = database.create_user(
        username="email_owner",
        email="owner@example.com",
        password_hash="test-hash",
    )
    try:
        database.update_account_profile(
            user_id,
            username="email_owner",
            email="",
        )
    except ValueError as exc:
        assert "ایمیل" in str(exc)
    else:
        raise AssertionError("email must remain required in profile")


def test_reset_code_hash_round_trip():
    code = generate_reset_code()
    assert len(code) == 6
    assert code.isdigit()
    encoded = hash_reset_code(code, iterations=100_000)
    assert code not in encoded
    assert verify_reset_code(code, encoded)
    assert not verify_reset_code("000000" if code != "000000" else "111111", encoded)


def test_password_reset_token_is_one_time_and_changes_password(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)
    old_hash = hash_password("old-password", iterations=100_000)
    user_id = database.create_user(
        username="reset_user",
        email="reset@example.com",
        password_hash=old_hash,
    )

    code = "483921"
    reset_id = database.create_password_reset_token(
        user_id,
        hash_reset_code(code, iterations=100_000),
        cooldown_seconds=0,
    )
    reset_record = database.get_active_password_reset("RESET@example.com")
    assert reset_record is not None
    assert reset_record["id"] == reset_id
    assert verify_reset_code(code, reset_record["token_hash"])

    new_hash = hash_password("new-password", iterations=100_000)
    assert database.complete_password_reset(reset_id, user_id, new_hash)
    credentials = database.get_user_credentials(user_id)
    assert verify_password("new-password", credentials["password_hash"])
    assert not verify_password("old-password", credentials["password_hash"])
    assert database.get_active_password_reset("reset@example.com") is None
    assert not database.complete_password_reset(reset_id, user_id, old_hash)


def test_reset_code_locks_after_five_failures(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)
    user_id = database.create_user(
        username="locked_reset",
        email="locked@example.com",
        password_hash="test-hash",
    )
    reset_id = database.create_password_reset_token(
        user_id,
        hash_reset_code("123456", iterations=100_000),
        cooldown_seconds=0,
    )
    for _ in range(5):
        database.record_password_reset_failure(reset_id)
    assert database.get_active_password_reset("locked@example.com") is None

def test_old_meals_schema_is_migrated_without_data_loss(tmp_path, monkeypatch):
    import sqlite3

    db_path = tmp_path / "legacy_food_calorie.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            email TEXT COLLATE NOCASE UNIQUE,
            password_hash TEXT NOT NULL,
            daily_calorie_goal REAL NOT NULL DEFAULT 2000,
            created_at TEXT NOT NULL
        );
        CREATE TABLE meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            meal_type TEXT NOT NULL
                CHECK (meal_type IN ('breakfast', 'lunch', 'dinner', 'snack')),
            meal_date TEXT NOT NULL,
            image_hash TEXT,
            predicted_label TEXT,
            total_calories REAL NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE meal_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meal_id INTEGER NOT NULL,
            food_label TEXT NOT NULL,
            food_name_fa TEXT NOT NULL,
            weight_grams REAL NOT NULL,
            calories_per_100g REAL NOT NULL,
            estimated_calories REAL NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            nutrition_description TEXT,
            source TEXT,
            FOREIGN KEY (meal_id) REFERENCES meals(id) ON DELETE CASCADE
        );
        INSERT INTO users (
            username, email, password_hash, daily_calorie_goal, created_at
        ) VALUES ('legacy_user', NULL, 'hash', 2000, '2026-08-01T00:00:00Z');
        INSERT INTO meals (
            user_id, meal_type, meal_date, total_calories, notes, created_at
        ) VALUES (1, 'snack', '2026-08-05', 200, 'legacy', '2026-08-05T10:00:00Z');
        INSERT INTO meal_items (
            meal_id, food_label, food_name_fa, weight_grams,
            calories_per_100g, estimated_calories, confidence
        ) VALUES (1, 'cake', 'کیک', 100, 200, 200, 0);
        """
    )
    connection.commit()
    connection.close()

    database.init_db()
    legacy_meals = database.get_meals_for_date(1, "2026-08-05")
    assert len(legacy_meals) == 1
    assert legacy_meals[0]["meal_type"] == "snack"
    assert legacy_meals[0]["meal_time"] is None
    legacy_items = database.get_meal_items(1, 1)
    assert len(legacy_items) == 1
    assert legacy_items[0]["measure_unit"] == "g"

    database.create_meal(
        user_id=1,
        meal_type="afternoon_snack",
        meal_date="2026-08-06",
        meal_time="16:30",
        items=[
            {
                "food_label": "pastry",
                "food_name_fa": "شیرینی",
                "weight_grams": 50,
                "calories_per_100g": 400,
            }
        ],
    )
    assert database.get_meals_for_date(1, "2026-08-06")[0]["meal_time"] == "16:30"


def test_new_registration_can_require_email_verification(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)
    user_id = database.create_user(
        username="verify_user",
        email="verify@example.com",
        password_hash="test-hash",
        email_verified=False,
    )
    account = database.get_user_for_login("verify@example.com")
    assert account is not None
    assert account["email_verified"] == 0

    code = "731942"
    verification_id = database.create_email_verification_token(
        user_id,
        hash_reset_code(code, iterations=100_000),
        cooldown_seconds=0,
    )
    record = database.get_active_email_verification("VERIFY@example.com")
    assert record is not None
    assert record["id"] == verification_id
    assert verify_reset_code(code, record["token_hash"])

    assert database.complete_email_verification(verification_id, user_id)
    account = database.get_user_for_login("verify@example.com")
    assert account["email_verified"] == 1
    assert database.get_active_email_verification("verify@example.com") is None


def test_email_verification_failure_counter_locks_after_five(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)
    user_id = database.create_user(
        username="verify_lock_user",
        email="verify-lock@example.com",
        password_hash="test-hash",
        email_verified=False,
    )
    verification_id = database.create_email_verification_token(
        user_id,
        hash_reset_code("123456", iterations=100_000),
        cooldown_seconds=0,
    )
    for _ in range(5):
        database.record_email_verification_failure(verification_id)
    assert database.get_active_email_verification("verify-lock@example.com") is None


def test_drink_volume_unit_is_saved_and_calculated(tmp_path, monkeypatch):
    use_temp_db(tmp_path, monkeypatch)
    user_id = database.create_user(
        username="drink_user",
        email="drink@example.com",
        password_hash="test-hash",
    )
    meal_id = database.create_meal(
        user_id=user_id,
        meal_type="snack",
        meal_date="2026-08-16",
        items=[
            {
                "food_label": "orange_juice",
                "food_name_fa": "آب پرتقال",
                "weight_grams": 250,
                "measure_unit": "ml",
                "calories_per_100g": 45,
            }
        ],
    )

    items = database.get_meal_items(user_id, meal_id)
    assert items[0]["measure_unit"] == "ml"
    assert items[0]["weight_grams"] == 250
    assert items[0]["estimated_calories"] == 112.5
