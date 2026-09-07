"""USDA FoodData Central nutrition lookup service."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from database import get_cached_nutrition, save_cached_nutrition
from services.food_catalog import get_local_kcal, get_nutrition_query

SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
DETAIL_URL = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"

# More explicit search phrases improve matching for a number of Food-101 labels.
QUERY_ALIASES = {
    "baby_back_ribs": "pork ribs cooked",
    "beef_carpaccio": "beef carpaccio",
    "beef_tartare": "beef tartare",
    "breakfast_burrito": "breakfast burrito",
    "cheese_plate": "cheese platter",
    "chicken_quesadilla": "chicken quesadilla",
    "clam_chowder": "clam chowder soup",
    "creme_brulee": "creme brulee",
    "deviled_eggs": "deviled eggs",
    "filet_mignon": "beef tenderloin steak cooked",
    "fish_and_chips": "fish and chips prepared",
    "french_fries": "french fries prepared",
    "fried_rice": "fried rice prepared",
    "grilled_cheese_sandwich": "grilled cheese sandwich",
    "grilled_salmon": "salmon grilled cooked",
    "hamburger": "hamburger prepared",
    "hot_dog": "hot dog sandwich prepared",
    "ice_cream": "ice cream regular",
    "lobster_roll_sandwich": "lobster roll sandwich",
    "macaroni_and_cheese": "macaroni and cheese prepared",
    "omelette": "omelet prepared",
    "onion_rings": "onion rings prepared",
    "pancakes": "pancakes prepared",
    "pizza": "pizza cheese regular crust",
    "spaghetti_bolognese": "spaghetti with meat sauce",
    "spaghetti_carbonara": "spaghetti carbonara",
    "spring_rolls": "spring roll prepared",
    "strawberry_shortcake": "strawberry shortcake",
    "tuna_tartare": "tuna tartare",
    "waffles": "waffles prepared",
    "cooked_white_rice": "rice white long grain cooked",
    "grilled_tomato": "tomatoes cooked",
    "mixed_green_salad": "salad mixed greens raw",
    "grilled_chicken_breast": "chicken breast grilled cooked",
    "water": "water plain",
    "black_tea": "tea brewed prepared with tap water unsweetened",
    "sweetened_tea": "tea brewed sweetened with sugar",
    "black_coffee": "coffee brewed prepared with tap water",
    "cola": "soft drink cola regular",
    "diet_cola": "soft drink cola diet",
    "orange_juice": "orange juice",
    "apple_juice": "apple juice",
    "whole_milk": "milk whole 3.25 percent",
    "low_fat_milk": "milk lowfat 1 percent",
    "chocolate_milk": "milk chocolate",
    "lemonade": "lemonade",
    "energy_drink": "energy drink",
}


@dataclass(frozen=True)
class NutritionResult:
    query: str
    calories_per_100g: float
    description: str
    fdc_id: int | None
    data_type: str | None
    source: str


class NutritionLookupError(RuntimeError):
    pass


def _normalise_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _query_for_label(label: str) -> str:
    normalized = _normalise_label(label)
    catalog_query = get_nutrition_query(normalized)
    if catalog_query:
        return catalog_query
    return QUERY_ALIASES.get(normalized, normalized.replace("_", " "))


def _extract_kcal(food: dict[str, Any]) -> float | None:
    for nutrient in food.get("foodNutrients", []) or []:
        name = str(
            nutrient.get("nutrientName")
            or nutrient.get("name")
            or nutrient.get("nutrient", {}).get("name")
            or ""
        ).strip().lower()
        unit = str(
            nutrient.get("unitName")
            or nutrient.get("unit")
            or nutrient.get("nutrient", {}).get("unitName")
            or ""
        ).strip().upper()
        value = nutrient.get("value", nutrient.get("amount"))

        if name == "energy" and unit in {"KCAL", "KCALORIES"} and value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric >= 0:
                return numeric
    return None


def _token_score(query: str, description: str) -> float:
    query_tokens = set(re.findall(r"[a-z]+", query.lower()))
    description_tokens = set(re.findall(r"[a-z]+", description.lower()))
    if not query_tokens:
        return 0.0
    return len(query_tokens & description_tokens) / len(query_tokens)


def _select_best_food(query: str, foods: list[dict[str, Any]]) -> tuple[dict[str, Any], float] | None:
    ranked: list[tuple[float, dict[str, Any], float]] = []
    type_bonus = {
        "Foundation": 0.30,
        "SR Legacy": 0.25,
        "Survey (FNDDS)": 0.20,
        "Branded": 0.0,
    }

    for food in foods:
        kcal = _extract_kcal(food)
        if kcal is None:
            continue
        description = str(food.get("description", ""))
        data_type = str(food.get("dataType", ""))
        score = _token_score(query, description) + type_bonus.get(data_type, 0.05)
        ranked.append((score, food, kcal))

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0], reverse=True)
    _, food, kcal = ranked[0]
    return food, kcal


def _request_search(query: str, api_key: str) -> list[dict[str, Any]]:
    payload = {
        "query": query,
        "pageSize": 25,
        "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)"],
    }
    response = requests.post(
        SEARCH_URL,
        params={"api_key": api_key},
        json=payload,
        timeout=30,
    )

    # Some prepared dishes are available only outside the preferred data types.
    if response.status_code == 200 and response.json().get("foods"):
        return list(response.json()["foods"])

    fallback = requests.post(
        SEARCH_URL,
        params={"api_key": api_key},
        json={"query": query, "pageSize": 25},
        timeout=30,
    )
    fallback.raise_for_status()
    return list(fallback.json().get("foods", []))


def lookup_calories(label: str, api_key: str | None = None) -> NutritionResult:
    """Return kcal per 100 g, preferring cache/USDA and using local recipe fallback.

    Iranian prepared dishes often do not have a sufficiently specific USDA entry.
    The experimental catalog therefore carries an approximate per-100-g fallback.
    The UI explicitly marks it as an estimate and keeps the value editable.
    """
    normalized = _normalise_label(label)
    explicit_catalog_query = get_nutrition_query(normalized)
    query = _query_for_label(normalized)
    cache_key = query.lower().strip()
    cached = get_cached_nutrition(cache_key)
    if cached:
        return NutritionResult(
            query=cache_key,
            calories_per_100g=float(cached["calories_per_100g"]),
            description=str(cached["description"]),
            fdc_id=cached["fdc_id"],
            data_type=cached["data_type"],
            source="cache",
        )

    local_kcal = get_local_kcal(normalized)

    # Complex Iranian recipes do not map reliably to a single USDA item.
    # When the catalog intentionally has no USDA query, use the explicit local
    # recipe estimate instead of risking a misleading fuzzy USDA match.
    if local_kcal is not None and explicit_catalog_query is None:
        return NutritionResult(
            query=cache_key,
            calories_per_100g=float(local_kcal),
            description="برآورد داخلی برای غذای ایرانی؛ مقدار واقعی با دستور پخت تغییر می‌کند",
            fdc_id=None,
            data_type="Local estimate",
            source="local_estimate",
        )

    effective_key = api_key or os.getenv("USDA_API_KEY", "DEMO_KEY")

    try:
        foods = _request_search(query, effective_key)
    except requests.RequestException as exc:
        if local_kcal is not None:
            return NutritionResult(
                query=cache_key,
                calories_per_100g=float(local_kcal),
                description="برآورد داخلی برای غذای ایرانی؛ مقدار واقعی با دستور پخت تغییر می‌کند",
                fdc_id=None,
                data_type="Local estimate",
                source="local_estimate",
            )
        raise NutritionLookupError(
            "ارتباط با پایگاه USDA برقرار نشد. اینترنت و کلید API را بررسی کنید."
        ) from exc

    selection = _select_best_food(query, foods)
    if selection is None:
        if local_kcal is not None:
            return NutritionResult(
                query=cache_key,
                calories_per_100g=float(local_kcal),
                description="برآورد داخلی برای غذای ایرانی؛ مقدار واقعی با دستور پخت تغییر می‌کند",
                fdc_id=None,
                data_type="Local estimate",
                source="local_estimate",
            )
        raise NutritionLookupError(
            "برای این غذا مقدار کالری مناسبی در نتایج USDA پیدا نشد."
        )

    food, calories = selection
    fdc_id_raw = food.get("fdcId")
    fdc_id = int(fdc_id_raw) if fdc_id_raw is not None else None
    description = str(food.get("description") or query)
    data_type = str(food.get("dataType") or "") or None

    save_cached_nutrition(
        query=cache_key,
        fdc_id=fdc_id,
        description=description,
        data_type=data_type,
        calories_per_100g=calories,
    )

    return NutritionResult(
        query=cache_key,
        calories_per_100g=calories,
        description=description,
        fdc_id=fdc_id,
        data_type=data_type,
        source="usda",
    )
