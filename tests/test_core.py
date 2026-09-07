from services.labels_fa import to_persian
from services.usda import _extract_kcal, _normalise_label, _query_for_label


def test_persian_label():
    assert to_persian("pizza") == "پیتزا"


def test_normalise_label():
    assert _normalise_label("French Fries") == "french_fries"


def test_query_alias():
    assert _query_for_label("french_fries") == "french fries prepared"


def test_extract_kcal():
    food = {
        "foodNutrients": [
            {"nutrientName": "Energy", "unitName": "kJ", "value": 500},
            {"nutrientName": "Energy", "unitName": "KCAL", "value": 250},
        ]
    }
    assert _extract_kcal(food) == 250.0


def test_manual_multi_food_labels():
    assert to_persian("cooked_white_rice") == "برنج سفید پخته"
    assert to_persian("grilled_tomato") == "گوجه کبابی"


def test_manual_multi_food_queries():
    assert _query_for_label("cooked_white_rice") == "rice white long grain cooked"
    assert _query_for_label("grilled_tomato") == "tomatoes cooked"


def test_food_search_finds_persian_and_english_labels():
    from services.labels_fa import search_food_labels

    assert "cooked_white_rice" in search_food_labels("برنج", limit=50)
    assert "grilled_salmon" in search_food_labels("salmon", limit=50)
    assert "cola" in search_food_labels("نوشابه", limit=50)


def test_zero_kcal_is_valid_for_water_like_items():
    from services.usda import _extract_kcal

    food = {
        "foodNutrients": [
            {"nutrientName": "Energy", "unitName": "KCAL", "value": 0}
        ]
    }
    assert _extract_kcal(food) == 0.0


def test_measurement_units_for_foods_and_drinks():
    from services.measurements import measure_unit_for_label

    assert measure_unit_for_label("cooked_white_rice") == "g"
    assert measure_unit_for_label("cola") == "ml"
    assert measure_unit_for_label("orange_juice") == "ml"
    assert measure_unit_for_label("doogh") == "ml"
    assert measure_unit_for_label("khoresh_karafs") == "g"


def test_drink_usda_basis_is_converted_to_per_100ml():
    from services.measurements import calories_for_display_unit

    assert round(calories_for_display_unit("whole_milk", 61, source="usda"), 2) == 62.83
    assert calories_for_display_unit("doogh", 30, source="local_estimate") == 30
    assert calories_for_display_unit("pizza", 266, source="usda") == 266
