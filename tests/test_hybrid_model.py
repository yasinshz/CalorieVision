from services.classifier import Prediction
from services.food_catalog import get_catalog_item, get_fine_grain_clip_prompts, iranian_food_count
from services.hybrid_classifier import (
    IRANIAN_LABELS,
    _apply_fine_grain_group_result,
    _fine_grain_group_for_labels,
    _merge_rankings,
)
from services.labels_fa import search_food_labels, to_persian
from services import usda


def test_expanded_catalog_has_many_iranian_foods():
    assert iranian_food_count() >= 80
    assert get_catalog_item("ghormeh_sabzi") is not None
    assert get_catalog_item("zereshk_polo_ba_morgh") is not None
    assert get_catalog_item("kabab_koobideh") is not None


def test_persian_search_finds_iranian_dishes_and_aliases():
    assert "ghormeh_sabzi" in search_food_labels("قورمه", limit=20)
    assert "zereshk_polo_ba_morgh" in search_food_labels("زرشک پلو", limit=20)
    assert "kabab_koobideh" in search_food_labels("کوبیده", limit=20)
    assert to_persian("ghormeh_sabzi") == "قورمه‌سبزی"


def test_expanded_global_foods_remain_available_for_manual_search():
    # Global extras remain useful for manual correction/search even though they
    # no longer compete in automatic CLIP recognition.
    matches = search_food_labels("پاستا", limit=50)
    assert "pasta_tomato_sauce" in matches
    assert "pasta_alfredo" in matches
    assert "pasta_pesto" in matches


def test_confident_food101_hamburger_is_not_overridden_by_unrelated_global_clip_label():
    food101 = [
        Prediction(label="hamburger", confidence=0.91),
        Prediction(label="club_sandwich", confidence=0.05),
    ]
    zero = [
        ("turkey_sandwich", 0.63),  # expanded global label: must be ignored
        ("hamburger", 0.52),        # Food-101 anchor used by specialist gate
        ("kabab_koobideh", 0.16),
        ("joojeh_kabab", 0.08),
    ]
    merged = _merge_rankings(food101, zero, top_k=4)
    assert merged[0].label == "hamburger"
    assert all(item.label != "turkey_sandwich" for item in merged)


def test_low_confidence_food101_can_be_overridden_by_clear_iranian_evidence():
    food101 = [
        Prediction(label="beef_tartare", confidence=0.31),
        Prediction(label="miso_soup", confidence=0.18),
    ]
    zero = [
        ("ghormeh_sabzi", 0.46),
        ("khoresh_karafs", 0.12),
        ("beef_tartare", 0.20),
        ("miso_soup", 0.06),
    ]
    merged = _merge_rankings(food101, zero, top_k=4)
    assert merged[0].label == "ghormeh_sabzi"
    assert merged[0].source == "iranian-zero-shot"


def test_strong_food101_can_still_be_overridden_when_iranian_clip_evidence_is_exceptional():
    # Food-101 sometimes confidently maps an Iranian dish to a visually similar
    # native class. A very large specialist-vs-anchor gap must still recover it.
    food101 = [
        Prediction(label="chicken_curry", confidence=0.81),
        Prediction(label="fried_rice", confidence=0.07),
    ]
    zero = [
        ("fesenjan", 0.55),
        ("gheimeh", 0.10),
        ("chicken_curry", 0.22),
        ("fried_rice", 0.04),
    ]
    merged = _merge_rankings(food101, zero, top_k=4)
    assert merged[0].label == "fesenjan"


def test_iranian_specialist_label_set_does_not_include_expanded_global_turkey_sandwich():
    assert "ghormeh_sabzi" in IRANIAN_LABELS
    assert "kabab_koobideh" in IRANIAN_LABELS
    assert "turkey_sandwich" not in IRANIAN_LABELS
    assert "pasta_tomato_sauce" not in IRANIAN_LABELS


def test_iranian_nutrition_uses_local_fallback_when_usda_has_no_match(monkeypatch):
    monkeypatch.setattr(usda, "get_cached_nutrition", lambda _: None)
    monkeypatch.setattr(usda, "_request_search", lambda query, api_key: [])
    result = usda.lookup_calories("ghormeh_sabzi", api_key="dummy")
    assert result.source == "local_estimate"
    assert result.calories_per_100g > 0
    assert "برآورد داخلی" in result.description


def test_pasta_family_blocks_iranian_stew_override_even_when_food101_is_only_moderate():
    # Regression for the user's red penne photo: a saucy orange/red pasta must
    # never become gheimeh merely because CLIP finds both visually similar.
    food101 = [
        Prediction(label="macaroni_and_cheese", confidence=0.34),
        Prediction(label="spaghetti_bolognese", confidence=0.18),
        Prediction(label="ravioli", confidence=0.07),
    ]
    zero = [
        ("gheimeh", 0.42),
        ("gheimeh_bademjan", 0.14),
        ("macaroni_and_cheese", 0.12),
    ]
    merged = _merge_rankings(food101, zero, top_k=5)
    assert merged[0].label == "macaroni_and_cheese"
    assert all(item.label not in {"gheimeh", "gheimeh_bademjan"} for item in merged)


def test_pasta_family_can_refine_mac_and_cheese_to_tomato_pasta_without_iranian_competition():
    food101 = [
        Prediction(label="macaroni_and_cheese", confidence=0.57),
        Prediction(label="spaghetti_bolognese", confidence=0.16),
        Prediction(label="ravioli", confidence=0.06),
    ]
    zero = [
        ("pasta_tomato_sauce", 0.52),
        ("penne_arrabbiata", 0.20),
        ("macaroni_and_cheese", 0.19),
        ("gheimeh", 0.80),  # must be ignored on the protected pasta route
    ]
    merged = _merge_rankings(food101, zero, top_k=5)
    assert merged[0].label == "pasta_tomato_sauce"
    assert all(item.label != "gheimeh" for item in merged)


def test_family_consensus_can_detect_pasta_even_when_top_prediction_is_below_point_two():
    # Split Food-101 probability across several pasta classes should still
    # establish the pasta family and protect it from Iranian stew false positives.
    from services.hybrid_classifier import _detect_distinctive_family

    food101 = [
        Prediction(label="macaroni_and_cheese", confidence=0.18),
        Prediction(label="spaghetti_bolognese", confidence=0.13),
        Prediction(label="ravioli", confidence=0.08),
        Prediction(label="pizza", confidence=0.06),
    ]
    assert _detect_distinctive_family(food101) == "pasta"


def test_green_stew_prompts_emphasize_celery_vs_prunes():
    celery = " ".join(get_fine_grain_clip_prompts("khoresh_karafs")).lower()
    spinach_prune = " ".join(get_fine_grain_clip_prompts("khoresh_aloo_esfenaj")).lower()
    assert "celery" in celery
    assert "stalk" in celery or "ribbed" in celery
    assert "prune" in spinach_prune
    assert "spinach" in spinach_prune


def test_green_stew_family_is_detected_for_celery_and_spinach_prune():
    assert _fine_grain_group_for_labels(["khoresh_karafs"]) == "green_stews"
    assert _fine_grain_group_for_labels(["khoresh_aloo_esfenaj"]) == "green_stews"


def test_fine_grain_refiner_can_correct_spinach_prune_to_celery_stew():
    from services.hybrid_classifier import HybridPrediction

    merged = [
        HybridPrediction(
            label="khoresh_aloo_esfenaj",
            score=1.0,
            zero_shot_score=0.31,
            food101_confidence=0.0,
            source="iranian-zero-shot",
        ),
        HybridPrediction(
            label="khoresh_karafs",
            score=0.66,
            zero_shot_score=0.28,
            food101_confidence=0.0,
            source="iranian-zero-shot",
        ),
    ]
    refinement = [
        ("khoresh_karafs", 0.67),
        ("khoresh_aloo_esfenaj", 0.18),
        ("ghormeh_sabzi", 0.08),
        ("morgh_torsh", 0.04),
        ("ghelyeh_mahi", 0.03),
    ]
    corrected = _apply_fine_grain_group_result(
        merged, refinement, group="green_stews"
    )
    assert corrected[0].label == "khoresh_karafs"
    assert corrected[0].source == "iranian-fine-grain"


def test_fine_grain_refiner_does_not_flip_when_evidence_is_ambiguous():
    from services.hybrid_classifier import HybridPrediction

    merged = [
        HybridPrediction(
            label="khoresh_aloo_esfenaj",
            score=1.0,
            zero_shot_score=0.31,
            food101_confidence=0.0,
            source="iranian-zero-shot",
        ),
        HybridPrediction(
            label="khoresh_karafs",
            score=0.66,
            zero_shot_score=0.28,
            food101_confidence=0.0,
            source="iranian-zero-shot",
        ),
    ]
    refinement = [
        ("khoresh_karafs", 0.41),
        ("khoresh_aloo_esfenaj", 0.38),
    ]
    kept = _apply_fine_grain_group_result(merged, refinement, group="green_stews")
    assert kept[0].label == "khoresh_aloo_esfenaj"
