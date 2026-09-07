🧠 Hybrid Food Recognition Engine

NutriVision uses a hybrid food-recognition pipeline designed to preserve strong predictions from Food-101 while extending recognition to Persian foods through CLIP Zero-Shot Classification.

The key idea is simple: the two models do not compete equally in every situation. Instead, the system first determines which recognition route is appropriate and then applies CLIP only where it is useful.

🎯 Design Goals

The hybrid engine is designed to:

Preserve reliable Food-101 predictions for common foods

Extend recognition beyond Food-101's fixed class set

Improve support for Persian dishes

Reduce incorrect overrides between unrelated food families

Perform additional fine-grained comparison for visually similar Persian foods

Keep the final prediction editable by the user

🔄 Decision Flow

flowchart TD
    A[Input Food Image] --> B[Food-101 Classification]

    B --> C{Is a known food family clear?}

    C -->|Yes| D[Protect Food-101 Family]
    D --> E[Optional CLIP Refinement Within Same Family]
    E --> H[Final Candidate]

    C -->|No / Ambiguous| F[CLIP Zero-Shot Persian Food Recognition]
    F --> G{Is the result in an ambiguous Persian group?}

    G -->|Yes| I[Fine-Grained Comparison]
    G -->|No| H

    I --> H
    H --> J[User Confirmation or Correction]

Step-by-Step

Food-101 analyzes the input image first.

If a clear general-food family is detected, such as:

Pasta

Burger

Sandwich

Pizza

Sushi

the result is treated as a protected family.

In a protected family, CLIP may be used only for limited refinement and is not allowed to freely replace the result with an unrelated Persian dish.

If the Food-101 result is ambiguous or does not adequately represent the food, CLIP is activated as the Persian-food specialist.

If the Persian result belongs to a visually ambiguous group, an additional fine-grained comparison is performed.

The final result is shown to the user and remains editable.

🇮🇷 Fine-Grained Persian Food Recognition

Some Persian dishes share very similar colors, textures, and serving styles. To reduce confusion, the system performs a second comparison only among closely related candidates.

This stage focuses on visible discriminative cues rather than only the overall appearance of the dish.

🥬 Green Stew Group

Khoresh Karafs — Celery Stew

Visual cues include:

Clearly visible celery stalk pieces

Fibrous texture

Short rectangular or ridged celery segments

Green herb-based stew appearance

Aloo Esfenaj — Spinach & Plum Stew

Visual cues include:

Soft leafy spinach texture

Dark oval plums

Absence of obvious celery stalk pieces

Ghormeh Sabzi

Visual cues include:

Finely chopped herbs

Red kidney beans

Meat pieces

Dried lime

Morgh Torsh

Visual cues include:

Chicken pieces

Green herb-based sauce

Walnut-like texture

Ghalieh Mahi

Visual cues include:

Fish pieces

Green herb stew

Tamarind-based appearance

🍅 Tomato-Based Stew Group

Gheimeh

Visual cues include:

Yellow split peas

Meat

Tomato-based sauce

Thin fried potato strips

Gheimeh Bademjan

Visual cues include:

Yellow split peas

Fried eggplant

Tomato-based sauce

Khoresh Bademjan

Visual cues include:

Eggplant

Meat

Tomato-based sauce

No split peas as a defining feature

Khoresh Bamieh

Visual cues include:

Whole okra pieces

Distinct ridged okra texture

Tomato-based stew appearance

🛡️ Protected Food Families

Fine-grained Persian recognition is intentionally restricted to the selected Persian-food route.

It is not allowed to convert a strongly recognized protected general-food family—such as pasta, burger, pizza, sandwich, or sushi—into an unrelated Persian stew.

This prevents CLIP from overriding a strong Food-101 result simply because of superficial visual similarities.

🧪 Scientific Interpretation

The hybrid score should be interpreted as a ranking and decision score, not as a statistically calibrated probability.

In other words:

A higher hybrid score means that a candidate is preferred by the decision logic, but it should not automatically be interpreted as an exact probability of correctness.

Recognition quality depends on factors such as:

Image quality

Camera angle

Lighting conditions

Food presentation

Visual similarity between dishes

Prompt quality

Visibility of distinguishing ingredients

⚠️ Current Limitations

Visually similar dishes can still be confused.

Recognition depends heavily on visible ingredients.

Foods with hidden ingredients are inherently harder to distinguish.

Prompt quality directly affects zero-shot performance.

The hybrid score is not probability-calibrated.

Final prediction quality may vary across images and serving styles.

For this reason, NutriVision keeps user confirmation and manual correction as part of the final workflow.

💡 Why This Architecture?

A simple combination of Food-101 and CLIP can introduce new errors because the two models have different strengths.

NutriVision instead uses a route-aware hybrid strategy:

Food-101 remains the main reference for common food categories.

CLIP extends the recognition space to foods outside Food-101.

Fine-grained prompts help distinguish visually similar Persian dishes.

Protected families reduce unnecessary cross-family overrides.

User confirmation provides a final safety layer before storing the result.
