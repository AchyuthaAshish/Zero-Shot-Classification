"""Ambiguity Detection Layer for Industrial Defect Classification.

Conforms to Phase 1 Step 1.4:
1. Implements a dedicated, explainable ambiguity-detection layer.
2. Distinguishes:
   - Insufficient evidence (Unknown, not ambiguous)
   - Genuine competing evidence (ambiguous between specific defect categories)
   - Clearly supported single-category evidence (unambiguous)
3. Never implements a simplistic "confidence < X -> ambiguous" rule.
4. Uses multiple available signals:
   - calibrated probability & raw model score
   - top-2 probability/score margin
   - top-2 predicted categories
   - category-specific evidence/signals (English, Telugu script, Telugu-English)
   - input quality & UnknownDetector analysis
   - Gemini qualitative reliability (without fake numerical probabilities)
5. Fully explainable with structured AmbiguityAssessment output.
"""

from typing import Dict, List, Optional, Tuple, Any

from core.schemas import PreprocessedInput, ConfidenceAssessment, AmbiguityAssessment
from core.unknown_detector import get_unknown_detector, UnknownDetector
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository


# ---------------------------------------------------------------------------
# Centrally Defined Ambiguity Thresholds & Parameters (with justification)
# ---------------------------------------------------------------------------

# In an 8-class calibrated probability distribution (where uniform chance is 0.125),
# a top-2 difference under 0.12 (e.g., 0.38 vs 0.30) indicates that the model's top
# two hypotheses are too close to be statistically separated without decisive physical evidence.
AMBIGUITY_MARGIN_THRESHOLD: float = 0.12

# When multiple categories have matching keywords, if the top category has at least
# 3x the evidence signals of the runner-up (e.g. 3 signals vs 1), the evidence is
# considered decisively dominant rather than ambiguous.
COMPETING_DOMINANCE_RATIO: float = 3.0


class AmbiguityDetector:
    """Evaluates and explains whether defect classification is subject to ambiguity."""

    def __init__(self, unknown_detector: Optional[UnknownDetector] = None, taxonomy_repo=None):
        self.unknown_detector = unknown_detector or get_unknown_detector()
        self.taxonomy_repo = taxonomy_repo or get_taxonomy_repository()

    def evaluate_ambiguity(
        self,
        raw_or_preprocessed: Any,
        candidate_category: str,
        confidence: Optional[float] = None,
        confidence_assessment: Optional[ConfidenceAssessment] = None,
        top2_margin: Optional[float] = None,
        class_probabilities: Optional[Dict[str, float]] = None,
        model_source: str = "local_ml",
        verification_feedback: Optional[Dict[str, Any]] = None
    ) -> AmbiguityAssessment:
        """
        Evaluates ambiguity across multiple evidence and model signals.

        Decision Policy:
        Case 1: Insufficient evidence / vague / unrelated input -> NOT ambiguous.
        Case 2: Single-category uncontested evidence -> NOT ambiguous.
        Case 3: Genuine competing textual evidence between 2+ categories -> AMBIGUOUS.
        Case 4: Borderline model prediction (margin < 0.12) without decisive evidence -> AMBIGUOUS.
        Case 5: Decisive model margin (>= 0.12) with uncontested evidence -> NOT ambiguous.
        Case 6: Gemini external model qualitative dispute -> AMBIGUOUS.
        """
        # Resolve text
        if hasattr(raw_or_preprocessed, "raw_text"):
            text = raw_or_preprocessed.raw_text
        elif hasattr(raw_or_preprocessed, "normalized_text"):
            text = raw_or_preprocessed.normalized_text
        else:
            text = str(raw_or_preprocessed or "")

        ca = confidence_assessment
        effective_margin = ca.top2_margin if (ca and ca.top2_margin is not None) else top2_margin

        # Extract evidence using UnknownDetector
        evidence = self.unknown_detector.analyze_evidence(text)

        # Active categories with matching physical/equipment/symptom signals
        active_evidence_cats = {
            cat: matches for cat, matches in evidence.category_evidence.items()
            if len(matches) > 0
        }

        # -------------------------------------------------------------------
        # Case 1: Insufficient Evidence / Vague / Unrelated / Trivial Input
        # (This is Step 1.3 Unknown, NOT an ambiguity case!)
        # -------------------------------------------------------------------
        if evidence.is_unrelated_or_non_defect or (evidence.is_trivial_length and evidence.total_evidence_count == 0):
            return AmbiguityAssessment(
                is_ambiguous=False,
                reason="Unrelated or non-defect input does not present competing defect evidence.",
                top_category=None,
                competing_category=None,
                margin=None,
                evidence_summary="No industrial defect evidence detected.",
                method="non_defect_not_ambiguous"
            )

        if evidence.is_vague and evidence.total_evidence_count == 0:
            return AmbiguityAssessment(
                is_ambiguous=False,
                reason="Insufficient evidence to evaluate competing categories (classified as Unknown).",
                top_category=None,
                competing_category=None,
                margin=None,
                evidence_summary="Vague failure description without specific technical signals.",
                method="insufficient_evidence_not_ambiguous"
            )

        # -------------------------------------------------------------------
        # Case 2: Genuine Competing Textual Evidence Between 2+ Categories
        # (e.g. "The motor is hot and the network connection is failing")
        # -------------------------------------------------------------------
        if len(active_evidence_cats) >= 2:
            sorted_by_count = sorted(
                active_evidence_cats.keys(),
                key=lambda c: len(active_evidence_cats[c]),
                reverse=True
            )
            top_cat = sorted_by_count[0]
            comp_cat = sorted_by_count[1]
            top_count = len(active_evidence_cats[top_cat])
            comp_count = len(active_evidence_cats[comp_cat])

            # Check if top category overwhelmingly dominates
            if top_count < COMPETING_DOMINANCE_RATIO * comp_count:
                top_signals = ", ".join(active_evidence_cats[top_cat][:2])
                comp_signals = ", ".join(active_evidence_cats[comp_cat][:2])
                summary = f"{top_cat} ({top_signals}) vs {comp_cat} ({comp_signals})"
                return AmbiguityAssessment(
                    is_ambiguous=True,
                    reason=f"Genuine competing evidence detected between {top_cat} and {comp_cat}.",
                    top_category=top_cat,
                    competing_category=comp_cat,
                    margin=effective_margin,
                    evidence_summary=summary,
                    method="competing_textual_evidence"
                )

        # -------------------------------------------------------------------
        # Case 3: Clearly Supported Single-Category Evidence (Uncontested)
        # (e.g. "The conveyor motor is making a grinding noise.")
        # -------------------------------------------------------------------
        if len(active_evidence_cats) == 1:
            supported_cat = list(active_evidence_cats.keys())[0]
            signals = ", ".join(active_evidence_cats[supported_cat][:3])

            # If the candidate category aligns with the sole evidence category
            if candidate_category == supported_cat or candidate_category == "Unknown":
                return AmbiguityAssessment(
                    is_ambiguous=False,
                    reason=f"Unambiguous evidence: clear technical signals ({signals}) uniquely support {supported_cat}.",
                    top_category=supported_cat,
                    competing_category=None,
                    margin=effective_margin,
                    evidence_summary=f"{supported_cat}: {signals}",
                    method="single_category_evidence_uncontested"
                )
            else:
                # Evidence points to one category, but model proposed a different category!
                return AmbiguityAssessment(
                    is_ambiguous=True,
                    reason=f"Evidence conflict: physical signals point to {supported_cat} ({signals}), but model proposed {candidate_category}.",
                    top_category=supported_cat,
                    competing_category=candidate_category,
                    margin=effective_margin,
                    evidence_summary=f"Signals for {supported_cat} conflict with model hypothesis {candidate_category}.",
                    method="model_evidence_conflict"
                )

        # -------------------------------------------------------------------
        # Case 4: Borderline Model Prediction (Local ML)
        # (Where text has no direct keyword matches, but model probability distribution is available)
        # -------------------------------------------------------------------
        # Determine top 2 categories from class_probabilities if available
        top_model_cat = candidate_category
        comp_model_cat = None
        if class_probabilities and len(class_probabilities) >= 2:
            sorted_probs = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
            top_model_cat = sorted_probs[0][0]
            comp_model_cat = sorted_probs[1][0]
            calc_margin = round(float(sorted_probs[0][1] - sorted_probs[1][1]), 4)
            if effective_margin is None:
                effective_margin = calc_margin

        if "local" in model_source.lower() and effective_margin is not None:
            if effective_margin < AMBIGUITY_MARGIN_THRESHOLD:
                runner_up = comp_model_cat or "competing category"
                return AmbiguityAssessment(
                    is_ambiguous=True,
                    reason=f"Borderline model prediction: narrow margin ({effective_margin:.4f} < {AMBIGUITY_MARGIN_THRESHOLD}) between {top_model_cat} and {runner_up}.",
                    top_category=top_model_cat,
                    competing_category=comp_model_cat,
                    margin=effective_margin,
                    evidence_summary=f"Model probabilities overlap near decision boundary (margin: {effective_margin:.4f}).",
                    method="borderline_top2_margin"
                )
            else:
                return AmbiguityAssessment(
                    is_ambiguous=False,
                    reason=f"Decisive model separation: margin ({effective_margin:.4f}) satisfies clarity threshold.",
                    top_category=top_model_cat,
                    competing_category=comp_model_cat,
                    margin=effective_margin,
                    evidence_summary=f"Model hypothesis clearly separated (margin: {effective_margin:.4f}).",
                    method="decisive_model_margin"
                )

        # -------------------------------------------------------------------
        # Case 5: External Provider / Gemini Handling
        # (Qualitative signals only; no fake numerical probabilities)
        # -------------------------------------------------------------------
        if "gemini" in model_source.lower() or "external" in model_source.lower() or "aimlapi" in model_source.lower():
            if verification_feedback and not verification_feedback.get("verified", True):
                alt = verification_feedback.get("alternative_category")
                return AmbiguityAssessment(
                    is_ambiguous=True,
                    reason=f"Verification disputed proposal '{candidate_category}'" + (f" favoring '{alt}'." if alt else " due to insufficient evidence."),
                    top_category=candidate_category,
                    competing_category=alt,
                    margin=None,
                    evidence_summary="Secondary verification dispute.",
                    method="verification_dispute"
                )

            # Check qualitative reliability
            level = ca.level if ca else "Medium"
            if level == "Low":
                return AmbiguityAssessment(
                    is_ambiguous=True,
                    reason=f"External provider reported Low qualitative reliability for {candidate_category}.",
                    top_category=candidate_category,
                    competing_category=None,
                    margin=None,
                    evidence_summary="Low qualitative confidence reported by model.",
                    method="gemini_qualitative_uncertainty"
                )

            return AmbiguityAssessment(
                is_ambiguous=False,
                reason=f"External provider qualitatively confirmed {candidate_category} without competing evidence.",
                top_category=candidate_category,
                competing_category=None,
                margin=None,
                evidence_summary="Uncontested external qualitative classification.",
                method="gemini_qualitative_uncontested"
            )

        # Fallback default (clean unambiguous assignment)
        return AmbiguityAssessment(
            is_ambiguous=False,
            reason=f"Classification as {candidate_category} is uncontested.",
            top_category=candidate_category,
            competing_category=None,
            margin=effective_margin,
            evidence_summary="No competing evidence detected.",
            method="uncontested_default"
        )


# Global singleton instance
_ambiguity_detector: Optional[AmbiguityDetector] = None


def get_ambiguity_detector() -> AmbiguityDetector:
    """Returns singleton AmbiguityDetector instance."""
    global _ambiguity_detector
    if _ambiguity_detector is None:
        _ambiguity_detector = AmbiguityDetector()
    return _ambiguity_detector
