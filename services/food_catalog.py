"""Expanded food catalog for hybrid image recognition and nutrition fallback.

The catalog deliberately uses English visual prompts because the CLIP checkpoint used
by the experimental hybrid recogniser was trained primarily on image/text pairs in
English. Persian names and aliases are kept for UI/search.

Calories in ``local_kcal`` are approximate kcal per 100 g for foods and per 100 ml for beverages.
They are a fallback only and remain editable by the user in the meal form.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoodCatalogItem:
    label: str
    name_fa: str
    clip_prompt: str
    category: str = "other"
    aliases: tuple[str, ...] = ()
    nutrition_query: str | None = None
    local_kcal: float | None = None


def _f(
    label: str,
    fa: str,
    prompt: str,
    *,
    category: str = "iranian",
    aliases: tuple[str, ...] = (),
    nutrition_query: str | None = None,
    kcal: float | None = None,
) -> FoodCatalogItem:
    return FoodCatalogItem(
        label=label,
        name_fa=fa,
        clip_prompt=prompt,
        category=category,
        aliases=aliases,
        nutrition_query=nutrition_query,
        local_kcal=kcal,
    )


# Iranian dishes: prompts describe visible appearance, not just transliterated names.
IRANIAN_FOODS: tuple[FoodCatalogItem, ...] = (
    # Khoresh / stews
    _f("ghormeh_sabzi", "قورمه‌سبزی", "a bowl of Persian ghormeh sabzi, a dark green herb stew with kidney beans and chunks of beef, served with white rice", aliases=("قرمه سبزی", "خورشت قورمه سبزی"), kcal=150),
    _f("gheimeh", "خورش قیمه", "Persian gheimeh stew, yellow split peas and meat in red tomato sauce with thin french fries on top, served with rice", aliases=("قیمه", "خورشت قیمه"), kcal=155),
    _f("gheimeh_bademjan", "قیمه بادمجان", "Persian gheimeh bademjan, tomato meat and split pea stew with fried eggplant, served with rice", aliases=("خورش قیمه بادمجان",), kcal=165),
    _f("fesenjan", "خورش فسنجان", "Persian fesenjan, thick dark brown walnut and pomegranate stew with chicken, served beside white rice", aliases=("فسنجون", "فسنجان"), kcal=210),
    _f("khoresh_karafs", "خورش کرفس", "Persian khoresh karafs celery stew with many clearly visible chopped celery stalk segments, short ribbed rectangular celery pieces browned in a dark green herb sauce with chunks of meat, served with white rice; no whole dried prunes", aliases=("خورشت کرفس",), kcal=115),
    _f("khoresh_bademjan", "خورش بادمجان", "Persian eggplant stew with fried eggplants, tomato sauce and meat, served with rice", aliases=("خورشت بادمجان",), kcal=145),
    _f("khoresh_bamieh", "خورش بامیه", "Persian okra stew with whole okra, tomato sauce and chunks of meat, served with rice", aliases=("خورشت بامیه",), kcal=115),
    _f("khoresh_aloo_esfenaj", "خورش آلو اسفناج", "Persian spinach and prune stew with soft wilted dark leafy spinach, several whole dark brown oval dried prunes and chunks of meat, served with rice; no chopped ribbed celery stalk pieces", aliases=("آلو اسفناج",), kcal=125),
    _f("morgh_torsh", "مرغ ترش", "northern Iranian morgh torsh, chicken in a thick green herb and walnut sauce, served with rice", aliases=("مرغ ترش شمالی",), kcal=185),
    _f("ghelyeh_mahi", "قلیه ماهی", "southern Iranian ghelyeh mahi, fish pieces in a dark green tangy herb and tamarind stew, served with rice", aliases=("قلیه ماهی",), kcal=130),
    _f("anarbij", "اناربیج", "Gilani Persian anarbij, dark pomegranate walnut herb stew with small meatballs", aliases=("انار بیج",), kcal=210),
    _f("torsh_tareh", "ترش‌تره", "Gilani torsh tareh, thick bright green herb stew with egg and garlic", aliases=("ترش تره",), kcal=95),
    _f("baghala_ghatogh", "باقلاقاتق", "Gilani baghala ghatogh, pale green fava bean dill stew topped with a poached egg", aliases=("باقلا قاتق", "باقالی قاتق"), kcal=135),

    # Rice dishes
    _f("zereshk_polo_ba_morgh", "زرشک‌پلو با مرغ", "Persian saffron rice topped with red barberries, served with a braised or roasted chicken piece", aliases=("زرشک پلو", "زرشک پلو با مرغ"), kcal=190),
    _f("baghali_polo_ba_mahicheh", "باقالی‌پلو با ماهیچه", "Persian dill rice with green fava beans served with a large braised lamb shank", aliases=("باقالی پلو", "باقلا پلو با ماهیچه"), kcal=205),
    _f("baghali_polo_ba_morgh", "باقالی‌پلو با مرغ", "Persian dill and fava bean rice served with a cooked chicken piece", aliases=("باقالی پلو با مرغ",), kcal=180),
    _f("adas_polo", "عدس‌پلو", "Persian rice mixed with brown lentils, often topped with raisins and fried onions", aliases=("عدس پلو",), kcal=175),
    _f("lubia_polo", "لوبیاپلو", "Persian rice mixed with chopped green beans, tomato and small pieces of meat", aliases=("لوبیا پلو",), kcal=185),
    _f("tahchin_morgh", "ته‌چین مرغ", "Persian tahchin, a golden saffron yogurt rice cake with chicken inside and a crisp browned crust", aliases=("ته چین", "ته چین مرغ"), kcal=210),
    _f("tahchin_goosht", "ته‌چین گوشت", "Persian saffron yogurt rice cake with meat filling and a crisp golden crust", aliases=("ته چین گوشت",), kcal=220),
    _f("kalam_polo_shirazi", "کلم‌پلو شیرازی", "Shirazi kalam polo, Persian rice mixed with shredded cabbage, herbs and small meatballs", aliases=("کلم پلو", "کلم پلو شیرازی"), kcal=185),
    _f("reshteh_polo", "رشته‌پلو", "Persian rice mixed with toasted brown noodles, often served with raisins, dates or meat", aliases=("رشته پلو",), kcal=180),
    _f("albaloo_polo", "آلبالوپلو", "Persian rice with bright red sour cherries and saffron, often served with chicken or small meatballs", aliases=("آلبالو پلو",), kcal=180),
    _f("morasa_polo", "مرصع‌پلو", "Persian jeweled rice decorated with orange peel, pistachios, almonds, raisins and barberries", aliases=("مرصع پلو", "جواهر پلو"), kcal=205),
    _f("shirin_polo", "شیرین‌پلو", "Persian sweet saffron rice with candied orange peel, carrots, almonds and pistachios", aliases=("شیرین پلو",), kcal=200),
    _f("estamboli_polo", "استانبولی‌پلو", "Persian tomato rice, orange-red rice cooked with tomatoes and diced potatoes", aliases=("استانبولی", "دمی گوجه"), kcal=150),
    _f("sabzi_polo_ba_mahi", "سبزی‌پلو با ماهی", "Persian herb rice speckled green, served with a whole or filleted fried fish", aliases=("سبزی پلو", "سبزی پلو با ماهی"), kcal=175),
    _f("kateh", "کته", "a mound of plain Persian steamed white rice cooked absorption style", aliases=("برنج کته",), nutrition_query="rice white cooked", kcal=130),
    _f("chelo", "چلو", "plain Persian fluffy steamed white rice, separate long grains", aliases=("برنج چلو", "برنج سفید"), nutrition_query="rice white long grain cooked", kcal=130),
    _f("saffron_rice", "برنج زعفرانی", "Persian white basmati rice topped or mixed with bright yellow saffron rice", aliases=("پلو زعفرانی",), nutrition_query="white rice cooked", kcal=135),
    _f("tahdig", "ته‌دیگ", "crispy golden Persian rice crust, a crunchy browned layer of rice from the bottom of the pot", aliases=("ته دیگ",), kcal=300),

    # Kebabs and grilled meats
    _f("kabab_koobideh", "کباب کوبیده", "Persian koobideh kebab, long flat skewers of grilled minced meat with saffron rice and grilled tomato", aliases=("کوبیده", "چلو کباب کوبیده"), kcal=250),
    _f("kabab_barg", "کباب برگ", "Persian barg kebab, wide thin strips of grilled beef or lamb on skewers with rice and grilled tomato", aliases=("برگ", "چلو کباب برگ"), kcal=220),
    _f("kabab_chenjeh", "کباب چنجه", "Persian chenjeh kebab, chunky cubes of grilled lamb or beef on metal skewers", aliases=("چنجه",), kcal=235),
    _f("kabab_soltani", "کباب سلطانی", "Persian soltani kebab plate with one skewer of koobideh minced meat and one skewer of barg steak, rice and tomato", aliases=("سلطانی",), kcal=235),
    _f("joojeh_kabab", "جوجه‌کباب", "Persian joojeh kebab, yellow saffron marinated grilled chicken pieces on skewers with rice and grilled tomato", aliases=("جوجه کباب", "چلو جوجه"), kcal=190),
    _f("joojeh_kabab_bone_in", "جوجه‌کباب با استخوان", "Persian bone-in grilled saffron chicken pieces on skewers, charred golden skin", aliases=("جوجه با استخوان",), kcal=205),
    _f("shishlik", "شیشلیک", "Persian shishlik, grilled lamb rib chops on skewers, browned and juicy", aliases=("شیشلیک شاندیز",), kcal=275),
    _f("kabab_tabei", "کباب تابه‌ای", "Persian pan kebab, flattened minced meat cooked in a pan with sliced tomatoes", aliases=("کباب تابه ای",), kcal=225),
    _f("torsh_kabab", "کباب ترش", "northern Iranian sour kebab, grilled beef cubes coated with green herbs, walnut and pomegranate marinade", aliases=("کباب ترش شمالی",), kcal=240),
    _f("akbar_joojeh", "اکبرجوجه", "northern Iranian Akbar Joojeh, half chicken fried until golden, served with rice and dark pomegranate sauce", aliases=("اکبر جوجه",), kcal=245),
    _f("mahi_shekam_por", "ماهی شکم‌پر", "Persian whole stuffed fish, baked or fried, filled with green herbs, walnuts and pomegranate", aliases=("ماهی شکم پر",), kcal=185),

    # Soups, hearty dishes and starters
    _f("abgoosht_dizi", "آبگوشت / دیزی", "Persian abgoosht or dizi, lamb and chickpea potato tomato stew served in a small stone crock", aliases=("آبگوشت", "دیزی"), kcal=130),
    _f("haleem", "حلیم", "Persian haleem, thick beige wheat porridge with shredded meat, often topped with cinnamon and sugar", aliases=("هلیم",), kcal=135),
    _f("ash_reshteh", "آش رشته", "Persian ash reshteh, thick green herb and bean soup with noodles, topped with white kashk and fried onions", aliases=("آش رشته",), kcal=105),
    _f("ash_doogh", "آش دوغ", "Persian ash doogh, pale yogurt herb soup with chickpeas and green herbs", aliases=("آش دوغ",), kcal=80),
    _f("ash_jo", "آش جو", "Persian barley soup, thick soup with barley, beans and herbs", aliases=("آش جو",), kcal=90),
    _f("soup_jo", "سوپ جو", "Persian barley soup with carrots, chicken and creamy or tomato broth", aliases=("سوپ جو",), kcal=75),
    _f("kashk_bademjan", "کشک بادمجان", "Persian kashk bademjan, mashed fried eggplant topped with white kashk, fried mint and onions", aliases=("کشک و بادمجان",), kcal=180),
    _f("mirza_ghasemi", "میرزاقاسمی", "Persian mirza ghasemi, smoky mashed eggplant with tomato, garlic and scrambled egg", aliases=("میرزا قاسمی",), kcal=115),
    _f("kuku_sabzi", "کوکو سبزی", "Persian kuku sabzi, thick dark green herb omelet cut into wedges", aliases=("کوکوسبزی",), kcal=190),
    _f("kuku_sibzamini", "کوکو سیب‌زمینی", "Persian potato kuku, golden pan-fried potato and egg patties", aliases=("کوکو سیب زمینی",), kcal=205),
    _f("kotlet", "کتلت", "Persian kotlet, oval pan-fried minced meat and potato patties, browned crisp outside", aliases=("کتلت گوشت",), kcal=220),
    _f("shami_kabab", "شامی کباب", "Persian shami, round pan-fried meat and split pea patties", aliases=("شامی",), kcal=220),
    _f("dolmeh_barg_mo", "دلمه برگ مو", "Persian stuffed grape leaves, small rolled vine leaves filled with rice, herbs and meat", aliases=("دلمه برگ", "دلمه برگ مو"), kcal=150),
    _f("dolmeh_felfel", "دلمه فلفل", "Persian stuffed bell peppers filled with rice, herbs and minced meat", aliases=("دلمه فلفل دلمه ای",), kcal=145),
    _f("dolmeh_bademjan", "دلمه بادمجان", "Persian stuffed eggplant filled with rice, herbs and minced meat", aliases=("دلمه بادمجان",), kcal=150),
    _f("salad_olivieh", "سالاد الویه", "Persian Olivier salad, creamy potato chicken egg pea and pickle salad with mayonnaise", aliases=("الویه", "سالاد الویه"), kcal=220),
    _f("salad_shirazi", "سالاد شیرازی", "Persian Shirazi salad, finely diced cucumber tomato and onion with herbs and lime", aliases=("سالاد شیرازی",), kcal=35),
    _f("mast_khiar", "ماست و خیار", "Persian yogurt and cucumber dip with dried mint and herbs", aliases=("ماست خیار",), kcal=65),
    _f("mast_musir", "ماست موسیر", "Persian thick yogurt mixed with shallots", aliases=("ماست و موسیر",), kcal=85),
    _f("borani_esfenaj", "بورانی اسفناج", "Persian spinach yogurt dip, green spinach mixed with thick white yogurt", aliases=("بورانی", "بورانی اسفناج"), kcal=80),

    # Breakfast and breads
    _f("omelet_gojeh", "املت گوجه", "Persian tomato omelet, scrambled eggs cooked in a bright red tomato sauce in a frying pan", aliases=("املت ایرانی", "املت گوجه فرنگی"), kcal=125),
    _f("nimroo", "نیمرو", "Persian fried eggs sunny side up, usually served in a small pan", aliases=("تخم مرغ نیمرو",), nutrition_query="fried egg", kcal=195),
    _f("kaleh_pacheh", "کله‌پاچه", "Persian kaleh pacheh breakfast, cooked sheep head meat, tongue and trotters in broth", aliases=("کله پاچه",), kcal=190),
    _f("noon_panir_sabzi", "نان پنیر سبزی", "Persian breakfast plate with flatbread, white feta-like cheese, walnuts and fresh herbs", aliases=("نان و پنیر", "نان پنیر گردو"), kcal=235),
    _f("sangak_bread", "نان سنگک", "Persian sangak flatbread, long thin whole-wheat bread with a dimpled pebble-baked surface", aliases=("سنگک",), nutrition_query="whole wheat flatbread", kcal=250),
    _f("barbari_bread", "نان بربری", "Persian barbari bread, thick oval flatbread with parallel ridges and sesame seeds", aliases=("بربری",), nutrition_query="white flatbread", kcal=275),
    _f("lavash_bread", "نان لواش", "Persian lavash, very thin soft rectangular flatbread", aliases=("لواش",), nutrition_query="lavash flatbread", kcal=280),
    _f("taftoon_bread", "نان تافتون", "Persian taftoon, thin round flatbread with small holes", aliases=("تافتون",), nutrition_query="flatbread", kcal=270),

    # Desserts and sweets
    _f("sholeh_zard", "شله‌زرد", "Persian sholeh zard, bright yellow saffron rice pudding decorated with cinnamon and pistachios", aliases=("شله زرد",), kcal=145),
    _f("faloodeh_shirazi", "فالوده شیرازی", "Persian faloodeh, white frozen thin starch noodles in icy rosewater syrup with lime", aliases=("فالوده",), kcal=120),
    _f("bastani_sonnati", "بستنی سنتی", "Persian saffron ice cream, pale yellow with pistachio pieces and possible frozen cream flakes", aliases=("بستنی زعفرانی", "بستنی سنتی زعفرانی"), kcal=210),
    _f("zoolbia_bamieh", "زولبیا و بامیه", "Persian zoolbia bamieh sweets, glossy deep-fried syrup soaked orange spirals and small ridged pastries", aliases=("زولبیا", "بامیه شیرینی"), kcal=380),
    _f("halva_irani", "حلوا", "Persian halva, smooth brown flour and saffron sweet paste decorated on a plate", aliases=("حلوای ایرانی",), kcal=360),
    _f("samanu", "سمنو", "Persian samanu, thick glossy brown wheat sprout pudding", aliases=("سمنو",), kcal=180),
    _f("ranginak", "رنگینک", "southern Iranian ranginak, date and walnut dessert covered with toasted flour and cinnamon", aliases=("رنگینک خرما",), kcal=330),
    _f("gaz", "گز", "Persian gaz nougat, white chewy nougat pieces with pistachios", aliases=("گز اصفهان",), kcal=390),
    _f("sohan", "سوهان", "Persian sohan, thin brittle saffron toffee discs covered with pistachios", aliases=("سوهان قم",), kcal=500),
    _f("qottab", "قطاب", "Persian qottab, small round fried or baked pastries coated in powdered sugar", aliases=("قطاب یزد",), kcal=430),
    _f("kolompeh", "کلمپه", "Persian kolompeh, round stamped date-filled pastry", aliases=("کلمپه کرمان",), kcal=360),
    _f("noon_khamei", "نان خامه‌ای", "Persian cream puff, round choux pastry filled with white whipped cream", aliases=("نون خامه ای",), kcal=310),

    # Drinks
    _f("doogh", "دوغ", "a glass or bottle of Persian doogh, white yogurt drink sometimes topped with dried mint", category="drink", aliases=("دوغ نعنا",), nutrition_query="yogurt drink", kcal=30),
    _f("doogh_gazdar", "دوغ گازدار", "a glass or bottle of fizzy Persian yogurt drink, white and foamy", category="drink", aliases=("دوغ گازدار",), nutrition_query="yogurt drink", kcal=30),
    _f("chai_nabat", "چای با نبات", "a clear glass of Persian black tea with a saffron rock candy stick", category="drink", aliases=("چای نبات",), kcal=25),
    _f("khakshir_drink", "شربت خاکشیر", "Persian khakshir drink, clear sweet drink filled with tiny brown seeds and ice", category="drink", aliases=("خاکشیر",), kcal=55),
    _f("tokhm_sharbati_drink", "شربت تخم‌شربتی", "Persian basil seed drink, clear sweet beverage with many swollen black basil seeds", category="drink", aliases=("تخم شربتی",), kcal=55),
    _f("sekanjabin_drink", "شربت سکنجبین", "Persian sekanjabin mint syrup drink, pale green sweet drink with ice", category="drink", aliases=("سکنجبین",), kcal=70),
)


EXPANDED_GLOBAL_FOODS: tuple[FoodCatalogItem, ...] = (
    _f("pasta_tomato_sauce", "پاستا با سس گوجه", "a plate of pasta or penne coated in red tomato marinara sauce with herbs, no cheese sauce", category="global", aliases=("پاستا گوجه", "پاستا قرمز", "ماکارونی با سس گوجه"), nutrition_query="pasta with tomato sauce", kcal=None),
    _f("penne_arrabbiata", "پنه آربیاتا", "penne arrabbiata, tube pasta in a bright red spicy tomato sauce with parsley", category="global", aliases=("آربیاتا",), nutrition_query="pasta with tomato sauce"),
    _f("pasta_alfredo", "پاستا آلفردو", "creamy Alfredo pasta with white cream and parmesan sauce, often with chicken and mushrooms", category="global", aliases=("آلفردو", "پاستا سفید"), nutrition_query="pasta with cream sauce"),
    _f("pasta_pesto", "پاستا پستو", "pasta coated in bright green basil pesto sauce with parmesan", category="global", aliases=("پستو",), nutrition_query="pasta with pesto sauce"),
    _f("spaghetti_tomato_sauce", "اسپاگتی با سس گوجه", "a plate of spaghetti noodles coated in plain red tomato sauce without meat", category="global", aliases=("اسپاگتی گوجه",), nutrition_query="spaghetti with tomato sauce"),
    _f("grilled_chicken_with_rice", "مرغ گریل‌شده با برنج", "a plate with grilled chicken breast, a mound of white or saffron rice and vegetables", category="global", aliases=("مرغ و برنج", "چیکن رایس"), nutrition_query="grilled chicken and rice"),
    _f("roast_chicken", "مرغ بریان", "golden brown roasted chicken or roast chicken pieces with crisp skin", category="global", aliases=("مرغ روست",), nutrition_query="chicken roasted"),
    _f("fried_chicken", "مرغ سوخاری", "crispy golden fried chicken pieces with breaded crunchy coating", category="global", aliases=("چیکن سوخاری",), nutrition_query="fried chicken"),
    _f("grilled_fish", "ماهی گریل‌شده", "a grilled white fish fillet with browned grill marks and lemon", category="global", aliases=("ماهی کبابی",), nutrition_query="fish grilled cooked"),
    _f("fried_fish", "ماهی سرخ‌شده", "a golden pan fried or deep fried fish fillet", category="global", aliases=("ماهی سوخاری",), nutrition_query="fish fried cooked"),
    _f("baked_salmon", "سالمون تنوری", "a baked salmon fillet, pink orange fish with browned surface and lemon", category="global", aliases=("سالمون پخته",), nutrition_query="salmon baked cooked"),
    _f("chicken_caesar_salad", "سالاد سزار با مرغ", "Caesar salad topped with sliced grilled chicken breast, romaine lettuce, croutons and creamy dressing", category="global", aliases=("سزار مرغ",), nutrition_query="caesar salad with chicken"),
    _f("chicken_salad", "سالاد مرغ", "mixed green salad topped with sliced cooked chicken, tomato and cucumber", category="global", aliases=("سالاد با مرغ",), nutrition_query="chicken salad"),
    _f("oatmeal_banana", "اوتمیل با موز", "a bowl of oatmeal porridge topped with sliced banana", category="global", aliases=("جو دوسر با موز",), nutrition_query="oatmeal cooked banana"),
    _f("oatmeal_banana_nuts", "اوتمیل با موز و مغزها", "a bowl of oatmeal topped with banana slices, almonds or walnuts and berries", category="global", aliases=("اوتمیل موز بادام",), nutrition_query="oatmeal cooked with fruit and nuts"),
    _f("turkey_sandwich", "ساندویچ بوقلمون", "whole wheat sandwich with sliced turkey, lettuce and tomato", category="global", aliases=("ساندویچ ترکی",), nutrition_query="turkey sandwich"),
    _f("lentil_soup", "سوپ عدس", "a bowl of thick brown or orange lentil soup with herbs", category="global", aliases=("عدسی سوپی",), nutrition_query="lentil soup"),
    _f("lentil_stew", "خوراک عدس", "a bowl of thick cooked brown lentils, lentil stew", category="global", aliases=("عدسی", "خوراک عدسی"), nutrition_query="lentils cooked"),
    _f("greek_yogurt_honey_walnuts", "ماست یونانی با عسل و گردو", "a bowl of thick Greek yogurt topped with honey and walnut pieces", category="global", aliases=("ماست عسل گردو",), nutrition_query="Greek yogurt honey walnuts"),
    _f("apple", "سیب", "a whole fresh red or green apple", category="global", aliases=("سیب قرمز", "سیب سبز"), nutrition_query="apple raw"),
    _f("banana", "موز", "a whole ripe yellow banana", category="global", nutrition_query="banana raw"),
    _f("orange", "پرتقال", "a whole fresh orange fruit", category="global", nutrition_query="orange raw"),
    _f("boiled_egg", "تخم‌مرغ آب‌پز", "peeled boiled eggs or halved hard boiled eggs", category="global", aliases=("تخم مرغ آب پز",), nutrition_query="egg hard boiled"),
    _f("scrambled_eggs", "تخم‌مرغ هم‌زده", "soft yellow scrambled eggs on a plate", category="global", aliases=("اسکرامبل اگ",), nutrition_query="scrambled eggs"),
    _f("mashed_potatoes", "پوره سیب‌زمینی", "a mound of creamy mashed potatoes", category="global", aliases=("پوره",), nutrition_query="mashed potatoes prepared"),
    _f("roasted_vegetables", "سبزیجات تنوری", "mixed roasted vegetables such as broccoli carrots peppers and zucchini on a plate", category="global", aliases=("سبزیجات گریل",), nutrition_query="mixed vegetables roasted"),
    _f("white_rice_chicken", "برنج سفید با مرغ", "plain white rice served beside cooked chicken pieces", category="global", aliases=("چلو مرغ",), nutrition_query="rice chicken prepared"),
)


# Prompt ensembles for visually similar Iranian dishes.  These are used only in
# the second-stage fine-grained CLIP comparison, after the main Iranian route
# has already narrowed the image to a confusing family.  Multiple descriptions
# reduce the chance that CLIP relies on a single coarse cue such as dark colour.
FINE_GRAIN_CLIP_PROMPTS: dict[str, tuple[str, ...]] = {
    "khoresh_karafs": (
        "a close-up food photo of Persian khoresh karafs, celery stew with many obvious chopped celery stalk segments, short ribbed rectangular pieces of celery, browned green herbs and chunks of meat",
        "Iranian celery stew where the distinctive visible ingredient is thick fibrous celery stems cut into short pieces, in a dark herb broth with meat; there are no whole oval prunes",
        "dark Persian meat stew with numerous chunky celery ribs and stalk sections, served beside white saffron rice",
    ),
    "khoresh_aloo_esfenaj": (
        "a close-up food photo of Persian khoresh aloo esfenaj, a spinach and prune stew with soft wilted leafy spinach, whole dark oval dried prunes and chunks of meat",
        "Iranian spinach stew showing dark leafy spinach and several soft brown or black prunes; there are no ribbed celery stalk chunks",
        "dark green Persian stew with a soft leafy texture, dried plums or prunes and meat, served with white rice",
    ),
    "ghormeh_sabzi": (
        "Persian ghormeh sabzi, finely chopped dark green herbs with red kidney beans, chunks of meat and dried lime; no large celery stalk pieces and no whole prunes",
        "close-up of Iranian ghormeh sabzi showing chopped herbs, kidney beans and beef in a dark green stew beside white rice",
    ),
    "morgh_torsh": (
        "northern Iranian morgh torsh, chicken pieces in a thick green herb and walnut sauce, without kidney beans, prunes or obvious celery stalk chunks",
        "Persian sour herb chicken stew with green herbs, walnuts and chicken pieces",
    ),
    "ghelyeh_mahi": (
        "southern Iranian ghelyeh mahi, fish pieces in a dark green herb and tamarind stew, visibly fish rather than red meat",
        "dark tangy Persian fish stew with chopped herbs and chunks of fish served with rice",
    ),
    "gheimeh": (
        "Persian gheimeh, red tomato stew with yellow split peas, meat and thin fried potato sticks, not eggplant or okra",
        "Iranian tomato and split-pea meat stew topped with thin french fries",
    ),
    "gheimeh_bademjan": (
        "Persian gheimeh bademjan with long fried eggplant pieces, yellow split peas, meat and red tomato sauce",
        "Iranian tomato split-pea stew containing obvious fried eggplant halves or strips",
    ),
    "khoresh_bademjan": (
        "Persian eggplant stew with large fried eggplant pieces, meat and tomato sauce, without yellow split peas",
        "Iranian khoresh bademjan showing soft browned eggplants in red tomato meat sauce",
    ),
    "khoresh_bamieh": (
        "Persian okra stew with many clearly visible whole green okra pods, tomato sauce and chunks of meat",
        "Iranian bamieh stew where the distinctive ingredient is whole ridged okra pods",
    ),
}


def get_fine_grain_clip_prompts(label: str) -> tuple[str, ...]:
    """Return an ensemble of discriminative prompts for difficult Iranian dishes."""
    key = label.strip().lower().replace(" ", "_")
    prompts = FINE_GRAIN_CLIP_PROMPTS.get(key)
    if prompts:
        return prompts
    return (get_clip_prompt(key),)

ALL_EXTRA_ITEMS: tuple[FoodCatalogItem, ...] = IRANIAN_FOODS + EXPANDED_GLOBAL_FOODS
CATALOG_BY_LABEL: dict[str, FoodCatalogItem] = {item.label: item for item in ALL_EXTRA_ITEMS}
EXTRA_LABELS_FA: dict[str, str] = {item.label: item.name_fa for item in ALL_EXTRA_ITEMS}


def get_catalog_item(label: str) -> FoodCatalogItem | None:
    return CATALOG_BY_LABEL.get(label.strip().lower().replace(" ", "_"))


def get_clip_prompt(label: str, display_name: str | None = None) -> str:
    item = get_catalog_item(label)
    if item:
        return item.clip_prompt
    readable = label.strip().lower().replace("_", " ")
    return f"a clear close-up food photograph of {readable}, served on a plate"


def get_nutrition_query(label: str) -> str | None:
    item = get_catalog_item(label)
    return item.nutrition_query if item else None


def get_local_kcal(label: str) -> float | None:
    item = get_catalog_item(label)
    return item.local_kcal if item else None


def search_terms_for_label(label: str) -> tuple[str, ...]:
    item = get_catalog_item(label)
    if not item:
        return ()
    return (item.name_fa, item.label.replace("_", " "), *item.aliases)


def iranian_food_count() -> int:
    return len(IRANIAN_FOODS)
