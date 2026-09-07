"""Food image classification service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

MODEL_NAME = "nateraw/food"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class FoodClassifier:
    """Lazy wrapper around a pretrained Food-101 ViT model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._pipeline: Any | None = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "کتابخانه transformers نصب نیست. دستور pip install -r requirements.txt را اجرا کنید."
            ) from exc

        self._pipeline = pipeline(
            task="image-classification",
            model=self.model_name,
        )

    def predict(self, image: Image.Image, top_k: int = 5) -> list[Prediction]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self._load()
        rgb_image = image.convert("RGB")
        raw_results = self._pipeline(rgb_image, top_k=top_k)

        predictions: list[Prediction] = []
        for item in raw_results:
            label = str(item["label"]).strip().lower().replace(" ", "_")
            score = float(item["score"])
            predictions.append(Prediction(label=label, confidence=score))

        return predictions
