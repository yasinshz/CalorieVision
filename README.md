🥗 CalorieVision

AI-powered food recognition and calorie tracking with special support for Persian foods.

NutriVision is a multi-user nutrition journal that combines Food-101 and CLIP in a route-aware hybrid pipeline to recognize food images, estimate calorie intake, and track daily and weekly nutrition history.

✨ Features

🤖 Hybrid food recognition using Food-101 + CLIP

🇮🇷 Special support for Persian foods

🧠 Zero-shot classification for foods outside Food-101 classes

🎯 Route-aware decision logic to protect confident Food-101 predictions

🔍 Fine-grained recognition for visually similar Persian dishes

🍽️ Support for multi-item meals

⚖️ Food quantity in grams

🥤 Drink quantity in milliliters

🔥 Calorie estimation based on USDA FoodData Central

🗂️ Internal calorie fallback for selected Persian dishes

👤 Multi-user accounts with isolated user data

✉️ Email verification with a 6-digit one-time code

🔐 Password reset via email

📊 Daily dashboard and weekly reports

📅 Weekly reporting based on Saturday → Friday

📱 Responsive UI for desktop, tablet, and mobile

🌙 Light / Dark mode

✅ Automated tests for core application logic

🧠 How It Works

Food Image
   │
   ▼
Image Preprocessing
   │
   ▼
Food-101 Classifier
   │
   ├── Confident known food family
   │      │
   │      ▼
   │   Protected / family-aware refinement
   │
   └── Ambiguous or unsupported food
          │
          ▼
       CLIP Zero-Shot
          │
          ▼
   Persian Food Recognition
          │
          ▼
 Fine-Grained Comparison
          │
          ▼
 User Confirmation / Correction
          │
          ▼
 USDA / Internal Nutrition Data
          │
          ▼
 Quantity Entry
          │
          ▼
 Calorie Calculation
          │
          ▼
 SQLite Storage + Dashboard

🔬 Hybrid Recognition

NutriVision does not simply average the outputs of two models.

Food-101

The pretrained nateraw/food model is used as the primary classifier for common food categories such as:

Pizza

Burger

Pasta

Sushi

Sandwiches

Desserts

Other Food-101 classes

CLIP

openai/clip-vit-base-patch32 is used as a zero-shot image classifier to extend recognition beyond the fixed Food-101 class set.

This is especially useful for Persian dishes such as:

Ghormeh Sabzi

Gheimeh

Fesenjan

Khoresh Karafs

Zereshk Polo

Baghali Polo

Koobideh

Joojeh Kebab

Ash Reshteh

Mirza Ghasemi

Kashk-e Bademjan

and more

Route-Aware Decision

For visually distinctive Food-101 families such as pizza, burger, pasta, sandwich, and sushi, the system protects a confident Food-101 result from unrelated CLIP overrides.

When the main classifier is ambiguous or the food is likely outside the Food-101 class set, the Persian-food CLIP route becomes more influential.

Fine-Grained Persian Food Recognition

Some Persian dishes can look very similar. NutriVision performs an additional restricted comparison for selected groups using more descriptive prompts.

Examples include:

Khoresh Karafs vs. Aloo Esfenaj

Gheimeh vs. Gheimeh Bademjan

Khoresh Bademjan vs. Khoresh Bamieh

The prompts focus on visible cues such as celery stalks, eggplant, okra, split peas, beans, plums, fish, or chicken.

🔥 Calorie Estimation

For general foods, NutriVision can retrieve reference nutrition data from USDA FoodData Central.

The calorie estimate is calculated as:

Calories = (Consumed Amount × Reference Calories) / 100

Example:

Reference: 150 kcal / 100 g
Consumed:  200 g

Estimated calories = 300 kcal

For selected Persian mixed dishes without a reliable direct USDA match, an editable internal kcal/100g estimate is used.

Calorie values are estimates and can vary depending on ingredients, cooking method, oil, serving size, and recipe.

🥤 Food & Drink Measurement

NutriVision uses different measurement units depending on the item type:

Item Type

Quantity Unit

Reference Basis

Solid food

grams (g)

kcal / 100 g

Drinks

milliliters (ml)

kcal / 100 ml

Common drinks such as water, tea, coffee, soda, juice, milk, doogh, and Persian syrups are handled as volume-based items.

🛠️ Tech Stack

Technology

Purpose

Python

Main application language

Streamlit

Web UI

PyTorch

Deep learning runtime

Hugging Face Transformers

Model loading and inference

Food-101

Primary food classifier

CLIP

Zero-shot image-text matching

Pillow

Image preprocessing

SQLite

User, meal, and nutrition data

USDA FoodData Central

Nutrition data

SMTP

Email verification and password reset

Pytest

Automated testing

📁 Project Structure

NutriVision/
├── app.py
├── auth.py
├── database.py
├── mailer.py
├── calendar_utils.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── HYBRID_MODEL.md
│
├── services/
│   ├── classifier.py
│   ├── hybrid_classifier.py
│   ├── food_catalog.py
│   ├── labels_fa.py
│   ├── measurements.py
│   └── usda.py
│
├── tests/
│   └── ...
│
├── data/
│   └── ...
│
└── .streamlit/
    └── config.toml

🚀 Installation

Requirements

Python 3.10 or 3.11

Internet connection for the first model download

Internet connection for USDA and email features

Around 1–2 GB of free disk space for dependencies and model files

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/NutriVision.git
cd NutriVision

2. Create a virtual environment

Windows

python -m venv .venv
.venv\Scripts\activate

Linux / macOS

python -m venv .venv
source .venv/bin/activate

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

🔑 Environment Variables

Create a .env file based on .env.example.

USDA_API_KEY=YOUR_DATA_GOV_API_KEY

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=YOUR_EMAIL@example.com
SMTP_PASSWORD=YOUR_APP_PASSWORD
SMTP_FROM_EMAIL=YOUR_EMAIL@example.com
SMTP_USE_TLS=true
SMTP_USE_SSL=false

RESET_CODE_DEBUG=false
VERIFICATION_CODE_DEBUG=false

Important

Never commit your real .env file.

Never store your personal email password in the project.

Use an app-specific password or provider-recommended credential for SMTP.

Keep API keys and credentials outside the source code.

If no USDA key is configured, the application may fall back to DEMO_KEY, which has stricter request limits.

▶️ Run the App

python -m streamlit run app.py

The app normally opens at:

http://localhost:8501

👤 Typical User Flow

Create an account.

Verify the account using the 6-digit email code.

Log in.

Open New Meal.

Upload a food image or search manually.

Review the AI prediction.

Confirm or correct the food.

Retrieve nutrition information.

Enter food weight or drink volume.

Add additional meal items if needed.

Save the meal.

Review the daily dashboard, history, or weekly report.

🗄️ Data Model

User
 └── Meals
      └── Meal Items

Main SQLite tables:

Table

Purpose

users

Accounts and daily calorie targets

meals

Meal-level information

meal_items

Foods, quantities, units, and calories

nutrition_cache

Cached USDA results

analyses

Legacy MVP compatibility

🔐 Security

The current version includes:

Password hashing with PBKDF2-HMAC-SHA256

Random salt per password

Hashed email-verification and password-reset codes

15-minute OTP expiration

One-time-use verification/reset codes

Lockout after repeated invalid OTP attempts

User-scoped database queries

SQLite foreign keys and cascade deletion

Sensitive credentials stored outside the source code

The current authentication design is suitable for a local or MVP deployment. For a large public deployment, a production database such as PostgreSQL and a dedicated authentication service would be more appropriate.

🧪 Tests

Run the test suite with:

pip install pytest
pytest -q

Tests cover areas such as:

Food-label normalization

USDA calorie extraction

Password hashing and verification

User and meal creation

User data isolation

Daily and weekly calculations

Cascade deletion

Email verification

Password reset flows

Database migration

Saturday-to-Friday weekly reporting

Hybrid recognition logic

Persian food catalog behavior

Food/drink measurement handling

⚠️ Current Limitations

The system classifies the main food in the image; it is not a full multi-object food detector.

Food weight is not estimated from the image and must be entered by the user.

USDA values are reference values and may differ from the actual recipe.

Persian-food internal calorie values are approximate.

Visually similar foods can still be misclassified.

Hybrid scores are used for ranking and decision logic and should not be interpreted as perfectly calibrated probabilities.

This project is a nutrition tracking tool, not a medical or clinical diet recommendation system.

🗺️ Future Improvements

Possible future directions include:

Training or fine-tuning on a dedicated Persian-food dataset

Larger evaluation benchmark for Persian dishes

Portion-size estimation from images

Multi-food detection in a single plate

Better confidence calibration

PostgreSQL migration for production-scale deployment

Dedicated authentication service

Cloud deployment

Mobile application

Personalized nutrition analytics

📌 About the Project

NutriVision is a computer vision and nutrition-tracking project focused on combining:

Computer Vision

Deep Learning

Zero-Shot Learning

Nutrition Data

Web Application Development

User Data Management

The main technical idea is the controlled hybrid use of Food-101 and CLIP to preserve strong common-food predictions while extending recognition to Persian foods.
