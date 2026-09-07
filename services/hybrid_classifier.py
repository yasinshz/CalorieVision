"""Route-aware hybrid food image recogniser.

Version 12 route-aware strategy
--------------------------
Food-101 remains the primary general-food expert.  CLIP is *not* allowed to
freely compete with every Iranian dish for every image anymore.

The recogniser first looks at the Food-101 top predictions and decides whether
they form a visually distinctive global family such as pasta, burger,
sandwich, pizza, sushi or noodle soup.  When such a family is detected:

* Iranian candidates are completely blocked from automatic override.
* CLIP may only refine *inside the same family* (for example tomato pasta vs.
  macaroni-and-cheese).  This fixes false positives such as a red pasta image
  being called ``gheimeh`` simply because both are orange/red saucy foods.

If no distinctive global family is supported by Food-101, CLIP switches to the
Iranian-specialist route.  In that route it compares all Iranian catalog foods
against only the current Food-101 leaders.  An Iranian result can override only
when the specialist evidence is clearly stronger.

The returned ``score`` is a ranking/match score, not a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from PIL import Image

from services.classifier import FoodClassifier, Prediction
from services.food_catalog import IRANIAN_FOODS, get_clip_prompt, get_fine_grain_clip_prompts
from services.labels_fa import to_persian

ZERO_SHOT_MODEL_NAME = "openai/clip-vit-base-patch32"
IRANIAN_LABELS = frozenset(item.label for item in IRANIAN_FOODS)


# Food-101 families that have visually distinctive shapes/structures.  If the
# Food-101 results strongly support one of these families, an unrelated Iranian
# stew/rice/kebab candidate must not hijack the result.
DISTINCTIVE_FOOD101_FAMILIES: dict[str, frozenset[str]] = {
    "pasta": frozenset(
        {
            "macaroni_and_cheese",
            "lasagna",
            "ravioli",
            "spaghetti_bolognese",
            "spaghetti_carbonara",
            "gnocchi",
        }
    ),
    "burger": frozenset({"hamburger"}),
    "sandwich": frozenset(
        {
            "club_sandwich",
            "grilled_cheese_sandwich",
            "lobster_roll_sandwich",
            "croque_madame",
        }
    ),
    "pizza": frozenset({"pizza"}),
    "sushi": frozenset({"sushi", "sashimi"}),
    "noodle_soup": frozenset({"ramen", "pho", "pad_thai"}),
}


# Optional zero-shot refiners that are allowed ONLY after the corresponding
# family has already been established by Food-101.  These labels never compete
# against a hamburger, pizza, Iranian stew, etc. outside their own family.
FAMILY_REFINEMENT_LABELS: dict[str, frozenset[str]] = {
    "pasta": frozenset(
        {
            "pasta_tomato_sauce",
            "penne_arrabbiata",
            "pasta_alfredo",
            "pasta_pesto",
            "spaghetti_tomato_sauce",
        }
    ),
    "sandwich": frozenset({"turkey_sandwich"}),
    "burger": frozenset(),
    "pizza": frozenset(),
    "sushi": frozenset(),
    "noodle_soup": frozenset(),
}


# Iranian dishes that are easy to confuse because colour and sauce texture are
# similar.  They receive a second CLIP pass with highly discriminative prompt
# ensembles that focus on visible ingredients (celery stalks, prunes, beans,
# eggplant, okra) rather than the overall colour of the stew.
IRANIAN_FINE_GRAIN_GROUPS: dict[str, frozenset[str]] = {
    "green_stews": frozenset(
        {
            "khoresh_karafs",
            "khoresh_aloo_esfenaj",
            "ghormeh_sabzi",
            "morgh_torsh",
            "ghelyeh_mahi",
        }
    ),
    "tomato_stews": frozenset(
        {
            "gheimeh",
            "gheimeh_bademjan",
            "khoresh_bademjan",
            "khoresh_bamieh",
        }
    ),
}


def _fine_grain_group_for_labels(labels: Iterable[str]) -> str | None:
    labels_set = set(labels)
    for group, members in IRANIAN_FINE_GRAIN_GROUPS.items():
        if labels_set & members:
            return group
    return None


def _apply_fine_grain_group_result(
    merged: list[HybridPrediction],
    refinement: list[tuple[str, float]],
    *,
    group: str,
    min_winner_probability: float = 0.44,
    min_margin: float = 0.08,
) -> list[HybridPrediction]:
    """Refine only *within* an Iranian confusion group.

    This function never turns a protected general Food-101 result into an
    Iranian dish. It is called only after the main Iranian route has already
    selected a group member as the top result. The fine-grain pass may then
    swap that Iranian label for a visually better sibling (e.g. celery stew
    instead of spinach-prune stew).
    """
    if not merged or not refinement:
        return merged

    members = IRANIAN_FINE_GRAIN_GROUPS.get(group, frozenset())
    if not members or merged[0].label not in members:
        return merged

    ranked = sorted(
        ((label, max(0.0, float(score))) for label, score in refinement if label in members),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return merged

    winner_label, winner_prob = ranked[0]
    runner_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    if winner_prob < min_winner_probability or winner_prob - runner_prob < min_margin:
        return merged

    old_top_score = merged[0].score
    old_top_label = merged[0].label
    refinement_map = dict(ranked)
    by_label = {item.label: item for item in merged}

    # Ensure every group label can be surfaced even if it was just below the
    # first-stage top-k.  Fine-grain results are still marked as Iranian CLIP.
    for label, prob in ranked:
        if label not in by_label:
            by_label[label] = HybridPrediction(
                label=label,
                score=min(0.74, 0.50 * prob),
                zero_shot_score=prob,
                food101_confidence=0.0,
                source="iranian-fine-grain",
            )

    updated: list[HybridPrediction] = []
    for label, item in by_label.items():
        if label == winner_label:
            updated.append(
                HybridPrediction(
                    label=label,
                    score=max(old_top_score, 0.99),
                    zero_shot_score=max(item.zero_shot_score, winner_prob),
                    food101_confidence=item.food101_confidence,
                    source="iranian-fine-grain",
                )
            )
        elif label in members:
            prob = refinement_map.get(label, 0.0)
            # Penalize siblings according to the second-stage evidence.
            sibling_score = min(item.score, 0.70 * (prob / max(winner_prob, 1e-9)))
            if label == old_top_label and label != winner_label:
                sibling_score = min(sibling_score, 0.72)
            updated.append(
                HybridPrediction(
                    label=label,
                    score=float(max(0.0, sibling_score)),
                    zero_shot_score=max(item.zero_shot_score, prob),
                    food101_confidence=item.food101_confidence,
                    source="iranian-fine-grain" if prob > 0 else item.source,
                )
            )
        else:
            updated.append(item)

    updated.sort(key=lambda item: item.score, reverse=True)
    return updated


@dataclass(frozen=True)
class HybridPrediction:
    label: str
    score: float
    zero_shot_score: float
    food101_confidence: float
    source: str


def _unique_labels(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        label = str(item).strip().lower().replace(" ", "_")
        if label and label not in seen:
            seen.add(label)
            result.append(label)
    return result


def _detect_distinctive_family(food101: list[Prediction]) -> str | None:
    """Return a protected general-food family when Food-101 supports it.

    We use both the top confidence and the summed confidence of several results.
    This is important for pasta: Food-101 may split probability across
    macaroni-and-cheese, spaghetti and ravioli, while all three still clearly
    indicate *pasta* rather than an Iranian stew.
    """
    if not food101:
        return None

    top_label = food101[0].label
    top_conf = max(0.0, float(food101[0].confidence))

    family_scores: dict[str, float] = {}
    for family, labels in DISTINCTIVE_FOOD101_FAMILIES.items():
        family_scores[family] = sum(
            max(0.0, float(pred.confidence))
            for pred in food101[:8]
            if pred.label in labels
        )

    # A reasonably confident top result in a distinctive family is enough.
    for family, labels in DISTINCTIVE_FOOD101_FAMILIES.items():
        if top_label in labels and top_conf >= 0.20:
            return family

    # Or multiple same-family results may jointly establish the family.
    ranked = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
    if ranked and ranked[0][1] >= 0.30:
        best_family, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score >= second_score + 0.12:
            return best_family

    return None


def _food101_weight(confidence: float) -> tuple[float, float]:
    """Return (Food-101 weight, Iranian-CLIP weight) for Iranian routing."""
    if confidence >= 0.75:
        return 0.90, 0.10
    if confidence >= 0.45:
        return 0.78, 0.22
    return 0.64, 0.36


def _iranian_override_allowed(
    *,
    food101_confidence: float,
    best_iranian_score: float,
    best_food101_anchor_score: float,
    second_iranian_score: float,
    protected_family: str | None = None,
) -> bool:
    """Gate an Iranian override using CLIP evidence and family compatibility."""
    # The central v3 guard: once Food-101 has established a distinctive global
    # family (pasta/burger/etc.), Iranian CLIP cannot override it at all.
    if protected_family:
        return False

    if best_iranian_score <= 0:
        return False

    anchor = max(best_food101_anchor_score, 1e-9)
    ratio = best_iranian_score / anchor
    margin_over_second = best_iranian_score - max(0.0, second_iranian_score)

    # Strong Food-101 evidence is protected unless Iranian evidence is truly
    # exceptional.  Thresholds are intentionally stricter than v2.
    if food101_confidence >= 0.75:
        return ratio >= 1.80 and margin_over_second >= 0.012

    if food101_confidence >= 0.45:
        return ratio >= 1.35 and margin_over_second >= 0.008

    # Low-confidence Food-101 is exactly where the Iranian specialist is useful,
    # but still require a modest advantage over the best anchor.
    return ratio >= 1.08 and margin_over_second >= 0.003


def _merge_iranian_rankings(
    food101: list[Prediction],
    zero_shot: list[tuple[str, float]],
    *,
    top_k: int = 8,
    iranian_labels: frozenset[str] = IRANIAN_LABELS,
) -> list[HybridPrediction]:
    """Fuse Food-101 with the Iranian-only specialist route."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not food101 and not zero_shot:
        return []

    f_map = {p.label: max(0.0, float(p.confidence)) for p in food101}
    f_rank = {p.label: idx for idx, p in enumerate(food101, start=1)}
    z_map = {
        label: max(0.0, float(score))
        for label, score in zero_shot
        if label in iranian_labels or label in f_map
    }

    top_food_label = food101[0].label if food101 else ""
    top_food_conf = f_map.get(top_food_label, 0.0)
    protected_family = _detect_distinctive_family(food101)

    f_best = max(f_map.values(), default=0.0) or 1.0
    z_best = max(z_map.values(), default=0.0) or 1.0

    iranian_pairs = sorted(
        ((label, score) for label, score in z_map.items() if label in iranian_labels),
        key=lambda item: item[1],
        reverse=True,
    )
    best_iranian_label = iranian_pairs[0][0] if iranian_pairs else ""
    best_iranian_score = iranian_pairs[0][1] if iranian_pairs else 0.0
    second_iranian_score = iranian_pairs[1][1] if len(iranian_pairs) > 1 else 0.0
    best_anchor_score = max(
        (score for label, score in z_map.items() if label in f_map),
        default=0.0,
    )

    iranian_override = bool(best_iranian_label) and _iranian_override_allowed(
        food101_confidence=top_food_conf,
        best_iranian_score=best_iranian_score,
        best_food101_anchor_score=best_anchor_score,
        second_iranian_score=second_iranian_score,
        protected_family=protected_family,
    )

    food_weight, iranian_weight = _food101_weight(top_food_conf)
    union = set(f_map) | {label for label in z_map if label in iranian_labels}
    merged: list[HybridPrediction] = []

    for label in union:
        f_conf = f_map.get(label, 0.0)
        z_score = z_map.get(label, 0.0)
        f_rel = f_conf / f_best if f_best else 0.0
        z_rel = z_score / z_best if z_best else 0.0

        if label in iranian_labels:
            score = iranian_weight * z_rel
            if iranian_override:
                if label == best_iranian_label:
                    score = 1.0
                else:
                    score = min(0.78, 0.62 * z_rel)
            else:
                # Iranian alternatives remain visible for manual correction but
                # are intentionally kept well below the general expert winner.
                score = min(0.48, 0.40 * z_rel)
            source = "iranian-zero-shot"
        else:
            clip_support = z_rel if label in z_map else 0.0
            score = food_weight * f_rel + (1.0 - food_weight) * clip_support
            score += 0.02 / max(1, f_rank.get(label, 99))
            source = "hybrid-confirmed" if label in z_map else "food-101"

        merged.append(
            HybridPrediction(
                label=label,
                score=float(min(1.0, max(0.0, score))),
                zero_shot_score=float(z_score),
                food101_confidence=float(f_conf),
                source=source,
            )
        )

    # Protect Food-101 winner whenever the Iranian gate did not pass.
    if top_food_label and not iranian_override:
        merged = [
            HybridPrediction(
                label=item.label,
                score=1.0 if item.label == top_food_label else min(item.score, 0.82),
                zero_shot_score=item.zero_shot_score,
                food101_confidence=item.food101_confidence,
                source=item.source,
            )
            for item in merged
        ]

    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]


def _merge_family_refinement(
    food101: list[Prediction],
    zero_shot: list[tuple[str, float]],
    *,
    family: str,
    top_k: int = 8,
) -> list[HybridPrediction]:
    """Refine a confirmed distinctive family without any Iranian competition."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    family_food101 = DISTINCTIVE_FOOD101_FAMILIES.get(family, frozenset())
    refiners = FAMILY_REFINEMENT_LABELS.get(family, frozenset())
    allowed = family_food101 | refiners

    f_map = {
        p.label: max(0.0, float(p.confidence))
        for p in food101
        if p.label in allowed
    }
    z_map = {
        label: max(0.0, float(score))
        for label, score in zero_shot
        if label in allowed
    }

    if not f_map and not z_map:
        return []

    top_food = food101[0] if food101 else None
    top_food_label = top_food.label if top_food else ""
    top_food_conf = float(top_food.confidence) if top_food else 0.0
    f_best = max(f_map.values(), default=0.0) or 1.0
    z_best = max(z_map.values(), default=0.0) or 1.0

    # Determine whether a same-family expanded label clearly beats the CLIP
    # support for Food-101's own top anchor.
    best_refiner_label = ""
    best_refiner_score = 0.0
    for label in refiners:
        score = z_map.get(label, 0.0)
        if score > best_refiner_score:
            best_refiner_label, best_refiner_score = label, score

    top_anchor_score = z_map.get(top_food_label, 0.0)
    ratio = best_refiner_score / max(top_anchor_score, 1e-9) if best_refiner_label else 0.0

    if top_food_conf >= 0.75:
        refiner_override = ratio >= 1.55 and best_refiner_score >= 0.08
    elif top_food_conf >= 0.45:
        refiner_override = ratio >= 1.25 and best_refiner_score >= 0.06
    else:
        refiner_override = ratio >= 1.08 and best_refiner_score >= 0.04

    union = set(f_map) | set(z_map)
    merged: list[HybridPrediction] = []
    for label in union:
        f_conf = f_map.get(label, 0.0)
        z_score = z_map.get(label, 0.0)
        f_rel = f_conf / f_best if f_best else 0.0
        z_rel = z_score / z_best if z_best else 0.0

        if label in refiners:
            score = 0.72 * z_rel
            source = "family-refiner"
            if refiner_override and label == best_refiner_label:
                score = 1.0
        else:
            score = 0.64 * f_rel + 0.36 * z_rel
            source = "family-confirmed"

        merged.append(
            HybridPrediction(
                label=label,
                score=float(min(1.0, max(0.0, score))),
                zero_shot_score=float(z_score),
                food101_confidence=float(f_conf),
                source=source,
            )
        )

    # If no same-family refiner clearly won, keep Food-101's top result first.
    if top_food_label and not refiner_override:
        merged = [
            HybridPrediction(
                label=item.label,
                score=1.0 if item.label == top_food_label else min(item.score, 0.86),
                zero_shot_score=item.zero_shot_score,
                food101_confidence=item.food101_confidence,
                source=item.source,
            )
            for item in merged
        ]

    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]


# Backward-compatible name used by the existing tests/UI helper imports.
def _merge_rankings(
    food101: list[Prediction],
    zero_shot: list[tuple[str, float]],
    *,
    top_k: int = 8,
    iranian_labels: frozenset[str] = IRANIAN_LABELS,
) -> list[HybridPrediction]:
    family = _detect_distinctive_family(food101)
    if family:
        # Zero-shot input supplied directly to this helper may contain family
        # refinement labels.  Route to the family merger so tests can exercise
        # the exact same guard as runtime inference.
        return _merge_family_refinement(food101, zero_shot, family=family, top_k=top_k)
    return _merge_iranian_rankings(
        food101,
        zero_shot,
        top_k=top_k,
        iranian_labels=iranian_labels,
    )


class HybridFoodClassifier:
    """Food-101 expert + route-aware CLIP refinement / Iranian specialist."""

    def __init__(
        self,
        *,
        zero_shot_model_name: str = ZERO_SHOT_MODEL_NAME,
        food101_classifier: FoodClassifier | None = None,
    ) -> None:
        self.zero_shot_model_name = zero_shot_model_name
        self.food101 = food101_classifier or FoodClassifier()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device: str = "cpu"
        self._iranian_labels: list[str] = []
        self._iranian_text_features: Any | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "برای تشخیص Hybrid کتابخانه‌های torch و transformers لازم هستند. "
                "دستور pip install -r requirements.txt را اجرا کنید."
            ) from exc

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = CLIPProcessor.from_pretrained(self.zero_shot_model_name)
        self._model = CLIPModel.from_pretrained(self.zero_shot_model_name)
        self._model.to(self._device)
        self._model.eval()

    def _load_iranian_features(self) -> None:
        self._load_model()
        if self._iranian_text_features is not None:
            return

        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        torch = self._torch

        self._iranian_labels = [item.label for item in IRANIAN_FOODS]
        prompts = [
            get_clip_prompt(label, display_name=to_persian(label))
            for label in self._iranian_labels
        ]
        inputs = self._processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            text_features = self._model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._iranian_text_features = text_features

    def _encode_text_features(self, labels: list[str]) -> Any:
        self._load_model()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        torch = self._torch

        prompts = [get_clip_prompt(label, display_name=to_persian(label)) for label in labels]
        inputs = self._processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _score_candidates(
        self,
        image: Image.Image,
        candidate_labels: list[str],
        text_features: Any,
        *,
        top_k: int,
    ) -> list[tuple[str, float]]:
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        torch = self._torch

        rgb_image = image.convert("RGB")
        inputs = self._processor(images=rgb_image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(0)
            probs = torch.softmax(similarities / 0.025, dim=-1)
            values, indices = torch.topk(probs, k=min(top_k, len(candidate_labels)))

        return [
            (candidate_labels[int(index)], float(value))
            for value, index in zip(values.detach().cpu(), indices.detach().cpu())
        ]

    def _predict_family_refiner(
        self,
        image: Image.Image,
        food101_predictions: list[Prediction],
        *,
        family: str,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Run CLIP only inside an already-established global food family."""
        self._load_model()

        family_labels = set(DISTINCTIVE_FOOD101_FAMILIES.get(family, frozenset()))
        family_labels.update(FAMILY_REFINEMENT_LABELS.get(family, frozenset()))
        # Keep only Food-101 labels from this same family plus explicit refiners.
        candidate_labels = _unique_labels(
            [p.label for p in food101_predictions if p.label in family_labels]
            + sorted(FAMILY_REFINEMENT_LABELS.get(family, frozenset()))
        )
        if not candidate_labels:
            return []

        text_features = self._encode_text_features(candidate_labels)
        return self._score_candidates(
            image,
            candidate_labels,
            text_features,
            top_k=min(top_k, len(candidate_labels)),
        )

    def _encode_fine_grain_group_features(self, labels: list[str]) -> Any:
        """Build one averaged text embedding per label from discriminative prompts."""
        self._load_model()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        torch = self._torch

        label_features = []
        for label in labels:
            prompts = list(get_fine_grain_clip_prompts(label))
            inputs = self._processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self._model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                mean_feature = features.mean(dim=0, keepdim=True)
                mean_feature = mean_feature / mean_feature.norm(dim=-1, keepdim=True)
            label_features.append(mean_feature)

        return torch.cat(label_features, dim=0)

    def _predict_iranian_fine_grain(
        self,
        image: Image.Image,
        *,
        group: str,
    ) -> list[tuple[str, float]]:
        """Second-stage comparison inside a known confusing Iranian family."""
        labels = sorted(IRANIAN_FINE_GRAIN_GROUPS.get(group, frozenset()))
        if not labels:
            return []
        features = self._encode_fine_grain_group_features(labels)
        return self._score_candidates(
            image,
            labels,
            features,
            top_k=len(labels),
        )

    def _predict_iranian_specialist(
        self,
        image: Image.Image,
        food101_predictions: list[Prediction],
        *,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        self._load_iranian_features()
        torch = self._torch
        assert torch is not None
        assert self._iranian_text_features is not None

        anchor_labels = _unique_labels(p.label for p in food101_predictions[:8])
        anchor_features = self._encode_text_features(anchor_labels) if anchor_labels else None

        candidate_labels = list(self._iranian_labels)
        text_features = self._iranian_text_features
        if anchor_labels and anchor_features is not None:
            candidate_labels.extend(anchor_labels)
            text_features = torch.cat([text_features, anchor_features], dim=0)

        return self._score_candidates(
            image,
            candidate_labels,
            text_features,
            top_k=min(top_k, len(candidate_labels)),
        )

    def predict(self, image: Image.Image, top_k: int = 8) -> list[HybridPrediction]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        food_predictions = self.food101.predict(image, top_k=max(8, top_k))
        family = _detect_distinctive_family(food_predictions)

        if family:
            # Distinctive global route: zero Iranian candidates are evaluated.
            family_predictions = self._predict_family_refiner(
                image,
                food_predictions,
                family=family,
                top_k=max(16, top_k * 3),
            )
            return _merge_family_refinement(
                food_predictions,
                family_predictions,
                family=family,
                top_k=top_k,
            )

        # Ambiguous/general route: CLIP acts as Iranian specialist only.
        specialist_predictions = self._predict_iranian_specialist(
            image,
            food_predictions,
            top_k=max(24, top_k * 4),
        )
        merged = _merge_iranian_rankings(
            food_predictions,
            specialist_predictions,
            top_k=max(top_k, 12),
        )

        # If the main Iranian route already chose one of the known confusing
        # stew families, run a second, much narrower visual comparison.  This
        # is intentionally an *intra-Iranian* correction only; it cannot hijack
        # a protected pasta/burger/pizza/etc. result.
        group = _fine_grain_group_for_labels([merged[0].label] if merged else [])
        if group:
            refinement = self._predict_iranian_fine_grain(image, group=group)
            merged = _apply_fine_grain_group_result(
                merged,
                refinement,
                group=group,
            )

        return merged[:top_k]
