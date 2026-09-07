"""Measurement helpers for foods and drinks.

Food portions are entered in grams. Beverage portions are entered in millilitres.
Internally the legacy ``weight_grams`` numeric column is retained for backward
compatibility, while ``measure_unit`` records whether that numeric amount is g or ml.
"""

from __future__ import annotations

from services.food_catalog import get_catalog_item


# Common drink labels that live in the general Persian label map rather than the
# expanded food catalog.
DRINK_LABELS: frozenset[str] = frozenset(
    {
        "water",
        "black_tea",
        "sweetened_tea",
        "black_coffee",
        "cola",
        "diet_cola",
        "orange_juice",
        "apple_juice",
        "whole_milk",
        "low_fat_milk",
        "chocolate_milk",
        "lemonade",
        "energy_drink",
    }
)


def normalise_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def is_drink_label(label: str) -> bool:
    """Return True when a catalog/search label represents a beverage."""
    normalized = normalise_label(label)
    item = get_catalog_item(normalized)
    if item is not None and item.category == "drink":
        return True
    return normalized in DRINK_LABELS


def measure_unit_for_label(label: str) -> str:
    """Return ``ml`` for drinks and ``g`` for foods."""
    return "ml" if is_drink_label(label) else "g"


def amount_label_fa(unit: str) -> str:
    return "حجم مصرف‌شده (میلی‌لیتر)" if unit == "ml" else "وزن مصرف‌شده (گرم)"


def short_unit_fa(unit: str) -> str:
    return "میلی‌لیتر" if unit == "ml" else "گرم"


def per_100_label_fa(unit: str) -> str:
    return "کالری در ۱۰۰ میلی‌لیتر" if unit == "ml" else "کالری در ۱۰۰ گرم"


def default_amount(unit: str) -> float:
    return 250.0 if unit == "ml" else 100.0


def max_amount(unit: str) -> float:
    return 10000.0 if unit == "ml" else 5000.0

# Approximate beverage densities (g/ml) used only to translate USDA's standard
# kcal/100 g basis into the UI's kcal/100 ml basis. Values near 1.0 are enough
# for calorie journaling and remain user-editable in the form.
DRINK_DENSITY_G_PER_ML: dict[str, float] = {
    "water": 1.00,
    "black_tea": 1.00,
    "sweetened_tea": 1.02,
    "black_coffee": 1.00,
    "cola": 1.04,
    "diet_cola": 1.00,
    "orange_juice": 1.04,
    "apple_juice": 1.04,
    "whole_milk": 1.03,
    "low_fat_milk": 1.03,
    "chocolate_milk": 1.04,
    "lemonade": 1.03,
    "energy_drink": 1.04,
    "doogh": 1.03,
    "doogh_gazdar": 1.02,
    "chai_nabat": 1.02,
    "khakshir_drink": 1.02,
    "tokhm_sharbati_drink": 1.02,
    "sekanjabin_drink": 1.04,
}


def calories_for_display_unit(
    label: str,
    calories_per_100g: float,
    *,
    source: str | None = None,
) -> float:
    """Return kcal/100 g for foods or approximate kcal/100 ml for drinks.

    Local drink estimates in the catalog are already authored per 100 ml. USDA
    and cache values are standardized per 100 g, so beverages are converted
    using an approximate density and remain editable by the user.
    """
    normalized = normalise_label(label)
    value = float(calories_per_100g)
    if not is_drink_label(normalized):
        return value
    if source == "local_estimate":
        return value
    density = DRINK_DENSITY_G_PER_ML.get(normalized, 1.0)
    return value * density
