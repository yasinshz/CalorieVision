"""Persian display names for Food-101 plus the experimental expanded catalog."""

from services.food_catalog import EXTRA_LABELS_FA, search_terms_for_label

LABELS_FA = {
    "apple_pie": "پای سیب",
    "baby_back_ribs": "دنده کبابی",
    "baklava": "باقلوا",
    "beef_carpaccio": "کارپاچوی گوشت",
    "beef_tartare": "تارتار گوشت",
    "beet_salad": "سالاد چغندر",
    "beignets": "بینیه",
    "bibimbap": "بیبیم‌باپ",
    "bread_pudding": "پودینگ نان",
    "breakfast_burrito": "بوریتوی صبحانه",
    "bruschetta": "بروشتا",
    "caesar_salad": "سالاد سزار",
    "cannoli": "کانولی",
    "caprese_salad": "سالاد کاپرزه",
    "carrot_cake": "کیک هویج",
    "ceviche": "سویچه",
    "cheese_plate": "بشقاب پنیر",
    "cheesecake": "چیزکیک",
    "chicken_curry": "کاری مرغ",
    "chicken_quesadilla": "کسادیای مرغ",
    "chicken_wings": "بال مرغ",
    "chocolate_cake": "کیک شکلاتی",
    "chocolate_mousse": "موس شکلات",
    "churros": "چوروس",
    "clam_chowder": "سوپ صدف",
    "club_sandwich": "کلاب ساندویچ",
    "crab_cakes": "کیک خرچنگ",
    "creme_brulee": "کرم بروله",
    "croque_madame": "کروک مادام",
    "cup_cakes": "کاپ‌کیک",
    "deviled_eggs": "تخم‌مرغ شکم‌پر",
    "donuts": "دونات",
    "dumplings": "دامپلینگ",
    "edamame": "ادامامه",
    "eggs_benedict": "تخم‌مرغ بندیکت",
    "escargots": "حلزون خوراکی",
    "falafel": "فلافل",
    "filet_mignon": "فیله مینیون",
    "fish_and_chips": "ماهی و سیب‌زمینی سرخ‌شده",
    "foie_gras": "فوا گرا",
    "french_fries": "سیب‌زمینی سرخ‌شده",
    "french_onion_soup": "سوپ پیاز فرانسوی",
    "french_toast": "فرنچ تست",
    "fried_calamari": "کالاماری سرخ‌شده",
    "fried_rice": "برنج سرخ‌شده",
    "frozen_yogurt": "ماست یخ‌زده",
    "garlic_bread": "نان سیر",
    "gnocchi": "نیوکی",
    "greek_salad": "سالاد یونانی",
    "grilled_cheese_sandwich": "ساندویچ پنیر گریل‌شده",
    "grilled_salmon": "سالمون گریل‌شده",
    "guacamole": "گواکاموله",
    "gyoza": "گیوزا",
    "hamburger": "همبرگر",
    "hot_and_sour_soup": "سوپ ترش و تند",
    "hot_dog": "هات‌داگ",
    "huevos_rancheros": "هوئوس رانچروس",
    "hummus": "حمص",
    "ice_cream": "بستنی",
    "lasagna": "لازانیا",
    "lobster_bisque": "بیسک خرچنگ دریایی",
    "lobster_roll_sandwich": "ساندویچ رول خرچنگ",
    "macaroni_and_cheese": "ماکارونی و پنیر",
    "macarons": "ماکارون",
    "miso_soup": "سوپ میسو",
    "mussels": "صدف سیاه",
    "nachos": "ناچو",
    "omelette": "املت",
    "onion_rings": "حلقه پیاز",
    "oysters": "صدف خوراکی",
    "pad_thai": "پد تای",
    "paella": "پائیا",
    "pancakes": "پنکیک",
    "panna_cotta": "پاناکوتا",
    "peking_duck": "اردک پکنی",
    "pho": "فو",
    "pizza": "پیتزا",
    "pork_chop": "چاپ گوشت خوک",
    "poutine": "پوتین",
    "prime_rib": "دنده مخصوص",
    "pulled_pork": "گوشت خوک ریش‌ریش",
    "ramen": "رامن",
    "ravioli": "راویولی",
    "red_velvet_cake": "کیک ردولوت",
    "risotto": "ریزوتو",
    "samosa": "سمبوسه",
    "sashimi": "ساشیمی",
    "scallops": "اسکالوپ",
    "seaweed_salad": "سالاد جلبک",
    "shrimp_and_grits": "میگو و گریتس",
    "spaghetti_bolognese": "اسپاگتی بولونز",
    "spaghetti_carbonara": "اسپاگتی کربونارا",
    "spring_rolls": "اسپرینگ رول",
    "steak": "استیک",
    "strawberry_shortcake": "شورت‌کیک توت‌فرنگی",
    "sushi": "سوشی",
    "tacos": "تاکو",
    "takoyaki": "تاکویاکی",
    "tiramisu": "تیرامیسو",
    "tuna_tartare": "تارتار ماهی تن",
    "waffles": "وافل",

    # اقلام رایج برای افزودن دستی به وعده‌های چندبخشی
    "cooked_white_rice": "برنج سفید پخته",
    "grilled_tomato": "گوجه کبابی",
    "mixed_green_salad": "سالاد سبز",
    "grilled_chicken_breast": "سینه مرغ گریل‌شده",

    # نوشیدنی‌های رایج برای جست‌وجو و ثبت سریع
    "water": "آب",
    "black_tea": "چای بدون قند",
    "sweetened_tea": "چای شیرین",
    "black_coffee": "قهوه سیاه",
    "cola": "نوشابه معمولی",
    "diet_cola": "نوشابه رژیمی",
    "orange_juice": "آب پرتقال",
    "apple_juice": "آب سیب",
    "whole_milk": "شیر پرچرب",
    "low_fat_milk": "شیر کم‌چرب",
    "chocolate_milk": "شیرکاکائو",
    "lemonade": "لیموناد",
    "energy_drink": "نوشیدنی انرژی‌زا",
}


# Experimental catalog: Iranian dishes and additional global foods.
LABELS_FA.update(EXTRA_LABELS_FA)


def _normalise_search_text(value: str) -> str:
    return (
        value.strip().lower()
        .replace("ي", "ی")
        .replace("ك", "ک")
        .replace("‌", " ")
        .replace("_", " ")
    )


def search_food_labels(query: str, *, limit: int = 40) -> list[str]:
    """Search labels by Persian display name or English/model label.

    The explicit helper is used by the UI so food options remain discoverable
    on mobile instead of depending only on the multiselect popup search.
    """
    cleaned = _normalise_search_text(query)
    labels = sorted(LABELS_FA, key=lambda label: LABELS_FA[label])
    if not cleaned:
        return labels[:limit] if limit > 0 else labels

    matches: list[tuple[int, str]] = []
    for label in labels:
        fa = _normalise_search_text(LABELS_FA[label])
        en = _normalise_search_text(label)
        extra_terms = tuple(_normalise_search_text(term) for term in search_terms_for_label(label))
        searchable = (fa, en, *extra_terms)
        if any(cleaned in term for term in searchable):
            starts = any(term.startswith(cleaned) for term in searchable)
            matches.append((0 if starts else 1, label))

    matches.sort(key=lambda item: (item[0], LABELS_FA[item[1]]))
    result = [label for _, label in matches]
    return result[:limit] if limit > 0 else result


def to_persian(label: str) -> str:
    """Return a Persian display label, falling back to a cleaned English label."""
    normalized = label.strip().lower().replace(" ", "_")
    return LABELS_FA.get(normalized, normalized.replace("_", " ").title())
