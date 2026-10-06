"""Unknown Detection and Evidence Verification Layer for Industrial Defect Classification.

Conforms to Phase 1 Step 1.3:
1. Evaluates technical evidence supporting one of the 7 approved fault categories.
2. Prevents forcing vague, unrelated, insufficient, or unsupported descriptions
   into a specific fault category.
3. Distinguishes:
   A. Specific supported defect -> return appropriate category
   B. Vague / insufficient evidence -> Unknown
   C. Unrelated / non-defect input -> Unknown
   D. Ambiguous evidence between categories -> do NOT automatically force a category
   E. Valid explicit defect with sufficient evidence -> retain specific category
4. Uses available signals: calibrated probabilities, raw model score, top-2 margin,
   confidence level, ambiguity state, input quality, multilingual tokens.
5. Strictly enforces the authoritative 8-category taxonomy.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any

from core.schemas import PreprocessedInput, ConfidenceAssessment
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository


# --- 1. CATEGORY-SPECIFIC EVIDENCE SIGNALS (English, Telugu script, Romanized Telugu) ---

CATEGORY_EVIDENCE_KEYWORDS: Dict[str, Set[str]] = {
    "Mechanical Fault": {
        # Equipment & Components
        "motor", "bearing", "shaft", "gear", "gearbox", "conveyor", "pump", "impeller",
        "roller", "belt", "coupling", "cylinder", "spindle", "valve", "compressor",
        "fan", "flange", "piston", "crankshaft", "pulley", "chain", "bushing",
        "మోటార్", "పంప్", "బేరింగ్", "గేర్", "షాఫ్ట్",
        # Symptoms & Failure Modes
        "vibration", "vibrating", "vibrate", "wobble", "shaking", "shake", "rattle",
        "rattling", "grinding", "chatter", "chattering", "knocking", "knock",
        "misalignment", "misaligned", "loose bolt", "cracked", "fractured", "wear",
        "worn", "seizure", "seized", "jam", "jammed", "binding", "friction",
        "unusual sound", "noise", "abnormal sound", "strange noise",
        "వైబ్రేట్", "రాపిడి", "జామ్", "శబ్దం", "వణుకు", "సౌండ్"
    },
    "Electrical Fault": {
        # Equipment & Components
        "wiring", "wire", "harness", "terminal", "terminal block", "cable", "conductor",
        "winding", "stator", "armature", "contactor", "relay", "fuse", "breaker",
        "circuit breaker", "busbar", "circuit", "capacitor", "resistor", "switchboard",
        "panel", "electrical panel", "junction box", "spark plug", "transformer winding",
        "ఫ్యూజ్", "వైరింగ్", "సర్క్యూట్",
        # Symptoms & Failure Modes
        "short circuit", "short", "open circuit", "sparking", "spark", "sparks", "arc",
        "arcing", "flashover", "blown fuse", "tripped breaker", "loose terminal",
        "ground fault", "insulation failure", "burnt winding", "scorched", "burning smell",
        "voltage drop", "leakage current", "high resistance", "overcurrent", "current spike",
        "burnt", "smoking",
        "షార్ట్ సర్క్యూట్", "స్పార్క్", "కాలిపోయి", "మంటలు", "పొగ"
    },
    "Sensor Fault": {
        # Equipment & Components
        "sensor", "transducer", "transmitter", "probe", "thermocouple", "rtd",
        "encoder", "proximity switch", "flow meter", "pressure sensor", "temperature sensor",
        "level sensor", "load cell", "tachometer", "accelerometer", "photocell",
        "సెన్సార్", "రీడింగ్",
        # Symptoms & Failure Modes
        "incorrect reading", "wrong reading", "false reading", "erratic reading",
        "inaccurate", "calibration drift", "drift", "stuck at", "frozen value",
        "no signal", "invalid reading", "transducer failure", "analog input error",
        "out of calibration", "reading fluctuation", "erratic measurement", "unstable reading",
        "zero reading", "signal jump", "offset error",
        "తప్పుడు రీడింగ్", "రీడింగ్ తప్పు", "సిగ్నల్ లేదు"
    },
    "Temperature Fault": {
        # Equipment & Components
        "heat exchanger", "chiller", "cooling jacket", "radiator", "cooling fan",
        "thermal probe", "heater", "heat sink", "cooling line", "thermostat",
        # Symptoms & Failure Modes
        "overheat", "overheating", "overheated", "temperature", "excessive heat",
        "running hot", "high temp", "thermal runaway", "cooling failure", "thermal overload",
        "temperature spiked", "heat accumulation", "dangerously hot", "very hot",
        "extremely hot", "boiling", "heat rise", "high temperature", "thermal breakdown",
        "వేడెక్కి", "ఓవర్‌హీట్", "ఎక్కువ వేడి", "హీట్"
    },
    "Software Fault": {
        # Equipment & Components
        "software", "application", "firmware", "plc program", "hmi", "controller logic",
        "embedded system", "scada", "runtime", "driver", "code", "software app", "user interface",
        "సాఫ్ట్‌వేర్", "ప్రోగ్రామ్",
        # Symptoms & Failure Modes
        "crash", "crashed", "crashing", "freeze", "frozen", "hung", "hanging",
        "bug", "exception", "null pointer", "memory leak", "reboot loop", "watchdog reset",
        "compilation error", "segmentation fault", "software lockup", "unresponsive interface",
        "error code", "unhandled error", "runtime error", "blue screen", "infinite loop",
        "logic error", "system halt",
        "క్రాష్", "ఆగిపోవడం", "హ్యాంగ్"
    },
    "Power Supply Fault": {
        # Equipment & Components
        "power supply", "smps", "ups", "battery", "dc bus", "transformer",
        "inverter", "power rail", "generator", "rectifier", "charger", "mains",
        "auxiliary supply", "power source", "incoming line", "feed line",
        "పవర్", "బ్యాటరీ", "విద్యుత్",
        # Symptoms & Failure Modes
        "no power", "power outage", "power loss", "power failure", "blackout",
        "brownout", "voltage sag", "low supply voltage", "zero voltage",
        "power supply dead", "battery drained", "breaker tripped on main supply",
        "unpowered", "power cut", "current collapse", "supply instability",
        "supply fluctuation", "lost incoming power",
        "పవర్ పోయింది", "పవర్ లేదు", "వోల్టేజ్ పడిపోయింది"
    },
    "Communication Fault": {
        # Equipment & Components
        "communication", "ethernet", "wifi", "network", "can bus", "modbus",
        "profibus", "rs485", "rs232", "lan", "gateway", "router", "switch",
        "serial port", "mqtt", "data link", "comm port", "fieldbus", "profinet",
        "కమ్యూనికేషన్", "నెట్‌వర్క్", "కనెక్షన్", "సిగ్నల్",
        # Symptoms & Failure Modes
        "disconnected", "connection lost", "timeout", "packet loss", "communication loss",
        "communication failure", "link down", "unreachable", "transmission error",
        "frame error", "dropped packets", "no handshake", "network failure",
        "connection refused", "protocol error", "crc error", "comm error",
        "కనెక్షన్ కట్", "సిగ్నల్ రాలేదు", "డిస్‌కనెక్ట్"
    }
}


# --- 2. VAGUE DEFECT PATTERNS (Without specific technical symptoms) ---

VAGUE_PATTERNS = [
    r"\bsomething\s+is\s+wrong\b",
    r"\bmachine\s+has\s+an\s+issue\b",
    r"\bproblem\s+occurred\b",
    r"\bsystem\s+is\s+not\s+working\s+properly\b",
    r"\bnot\s+working\s+properly\b",
    r"\bnot\s+working\b",
    r"\bequipment\s+is\s+not\s+working\b",
    r"\bmachine\s+stopped\b",
    r"\bunit\s+stopped\b",
    r"\bline\s+\d+\s+halt\b",
    r"\bplant\s+line\s+\d+\s+halt\b",
    r"\broutine\s+maintenance\s+check\b",
    r"\bcheck\s+unit\s+\d+\b",
    r"\bgeneral\s+equipment\s+malfunction\b",
    r"\bgeneral\s+malfunction\b",
    r"\bunspecified\s+(?:problem|disturbance|reasons?|equipment\s+anomaly|issue)\b",
    r"\boperator\s+noticed\s+an\s+issue\b",
    r"\boperator\s+raised\s+ticket\b",
    r"\bworkstation\s+stopped\b",
    r"\bceased\s+normal\s+operation\b",
    r"\bunexpected\s+behavior\b",
    r"\bprocess\s+interrupted\b",
    r"\bequipment\s+tripped\s+but\s+reason\s+unknown\b",
    r"\bunknown\s+trouble\b",
    r"\brequires?\s+immediate\s+technical\s+inspection\b",
    r"\bnot\s+responding\s+to\s+start\s+sequence\b",
    r"\bshut\s+off\s+unexpectedly\b",
    r"\bshut\s+down\s+automatically\s+without\s+diagnostic\b",
    r"\bdevice\s+stopped\b",
    r"\bissue\s+reported\b",
    r"\bproblem\s+reported\b",
    r"\bstatus\s+abnormal\b",
    r"\bcheck\s+cheyandi\b",
    r"\bedo\s+issue\b",
    r"\bedo\s+problem\b",
    r"\bproper\s+ga\s+work\s+avvatledu\b",
    r"\bwork\s+avvatledu\b",
    r"\brun\s+avvatledu\b",
    r"\breason\s+teleedu\b",
    r"\bsudden\s+ga\s+stop\b",
    r"\baagipoindi\b",
    r"ఏదో\s+సమస్య",
    r"సరిగ్గా\s+పనిచేయడం\s+లేదు",
    r"లైన్\s+ఆగిపోయింది",
    r"సమస్య\s+ఉందని",
    r"నడవట్లేదు",
    r"ఇబ్బంది\s+ఉంది",
    r"కారణం\s+తెలియదు",
    r"పని\s+చేయడం\s+ఆగిపోయింది",
    r"తేడాగా\s+ఉంది",
    r"తెలియని\s+సమస్య",
    r"తేడా\s+కనిపించి"
]

VAGUE_REGEX = re.compile("|".join(VAGUE_PATTERNS), re.IGNORECASE)


# --- 3. UNRELATED / NON-DEFECT PATTERNS ---

UNRELATED_PATTERNS = [
    r"\b(?:hello|hi|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening)\b",
    r"\b(?:need\s+help|help\s+me|please\s+help|can\s+you\s+help|assist\s+me)\b",
    r"\b(?:who\s+are\s+you|what\s+is\s+your\s+name|how\s+are\s+you)\b",
    r"^\s*(?:test|testing|sample|asdf|12345?|ok|okay)\s*[.!?]*\s*$",
    r"\b(?:what\s+is\s+the\s+capital\s+of|tell\s+me\s+a\s+joke|weather\s+today)\b",
    r"\b(?:thank\s+you|thanks|bye|goodbye)\b"
]

UNRELATED_REGEX = re.compile("|".join(UNRELATED_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceAnalysis:
    """Detailed evidence extraction result for defect text."""
    text: str
    has_equipment: bool
    category_evidence: Dict[str, List[str]]
    total_evidence_count: int
    is_vague: bool
    vague_matches: List[str]
    is_unrelated_or_non_defect: bool
    is_trivial_length: bool


@dataclass(frozen=True)
class UnknownDetectionResult:
    """Outcome of Unknown Detection analysis."""
    is_unknown: bool
    final_category: str
    decision_rule: str
    reason: str
    status: str
    confidence_assessment: ConfidenceAssessment


class UnknownDetector:
    """Decision engine layer evaluating technical evidence and Unknown designation."""

    def __init__(self, taxonomy_repo=None):
        self.taxonomy_repo = taxonomy_repo or get_taxonomy_repository()
        self.approved_categories = self.taxonomy_repo.get_categories()

    def analyze_evidence(self, text: str) -> EvidenceAnalysis:
        """Inspects text for equipment, category-specific defect signals, and vague/unrelated patterns."""
        safe_text = (text or "").strip()
        lower = safe_text.lower()
        words = re.findall(r"\w+", lower)

        # Check trivial length (< 2 words)
        is_trivial = len(words) < 2

        # Check unrelated / conversational
        is_unrelated = bool(UNRELATED_REGEX.search(safe_text))

        # Check vague patterns
        vague_matches = VAGUE_REGEX.findall(safe_text)
        is_vague = len(vague_matches) > 0

        # Scan for category-specific evidence
        cat_evidence: Dict[str, List[str]] = {}
        total_ev = 0
        for cat, keywords in CATEGORY_EVIDENCE_KEYWORDS.items():
            matches = []
            for kw in keywords:
                if re.match(r"^[a-zA-Z0-9\s_-]+$", kw):
                    if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                        matches.append(kw)
                else:
                    if kw in lower:
                        matches.append(kw)
            cat_evidence[cat] = matches
            total_ev += len(matches)

        has_equipment = any(
            eq in lower for eq in [
                "motor", "pump", "conveyor", "sensor", "bearing", "shaft", "gear",
                "plc", "controller", "cable", "wiring", "panel", "breaker", "battery",
                "chiller", "radiator", "ethernet", "wifi", "మోటార్", "పంప్", "సెన్సార్"
            ]
        )

        return EvidenceAnalysis(
            text=safe_text,
            has_equipment=has_equipment,
            category_evidence=cat_evidence,
            total_evidence_count=total_ev,
            is_vague=is_vague,
            vague_matches=vague_matches,
            is_unrelated_or_non_defect=is_unrelated,
            is_trivial_length=is_trivial
        )

    def evaluate_unknown(
        self,
        raw_or_preprocessed: Any,
        candidate_category: str,
        confidence: Optional[float] = None,
        confidence_assessment: Optional[ConfidenceAssessment] = None,
        top2_margin: Optional[float] = None,
        class_probabilities: Optional[Dict[str, float]] = None,
        model_source: str = "local_ml"
    ) -> UnknownDetectionResult:
        """
        Executes the authoritative Unknown Detection policy.

        Decision Policy:
        Rule 1: Primary classifier already returned 'Unknown' -> Retain Unknown.
        Rule 2: Input is non-defect / conversational / trivial length -> Unknown.
        Rule 3: Input matches vague failure patterns with zero specific category evidence -> Unknown.
        Rule 4: Ambiguous evidence: top2_margin < 0.10 and candidate lacks decisive dominance -> Unknown (Local) / Ambiguity flag.
        Rule 5: Candidate category supported by technical evidence -> Retain candidate category.
        Rule 6: Candidate category lacks evidence with low confidence (< 0.50) -> Unknown.
        Rule 7: Strict taxonomy validation guard.
        """
        # Resolve text
        if hasattr(raw_or_preprocessed, "raw_text"):
            text = raw_or_preprocessed.raw_text
        elif hasattr(raw_or_preprocessed, "normalized_text"):
            text = raw_or_preprocessed.normalized_text
        else:
            text = str(raw_or_preprocessed or "")

        evidence = self.analyze_evidence(text)
        ca = confidence_assessment

        raw_score = ca.raw_score if ca else confidence
        calibrated_prob = ca.calibrated_prob if ca else None
        effective_margin = ca.top2_margin if (ca and ca.top2_margin is not None) else top2_margin

        # Helper to construct Unknown ConfidenceAssessment
        def make_unknown_ca() -> ConfidenceAssessment:
            return ConfidenceAssessment(
                level="Uncertain",
                approximate_range="Insufficient Evidence",
                raw_score=raw_score,
                calibrated_prob=None,
                top2_margin=effective_margin,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=True
            )

        # ---------------------------------------------------------
        # Rule 1: Unrelated / Non-defect / Trivial input
        # ---------------------------------------------------------
        if evidence.is_unrelated_or_non_defect or (evidence.is_trivial_length and evidence.total_evidence_count == 0):
            return UnknownDetectionResult(
                is_unknown=True,
                final_category="Unknown",
                decision_rule="unrelated_non_defect",
                reason="Input does not describe an industrial equipment defect or malfunction; description appears to be unrelated or conversational text.",
                status="unknown",
                confidence_assessment=make_unknown_ca()
            )

        # ---------------------------------------------------------
        # Rule 2: Vague / Insufficient Technical Evidence
        # (Matches vague patterns and zero category-specific evidence)
        # ---------------------------------------------------------
        if evidence.is_vague and evidence.total_evidence_count == 0:
            return UnknownDetectionResult(
                is_unknown=True,
                final_category="Unknown",
                decision_rule="vague_insufficient_evidence",
                reason="Description is too vague and lacks specific technical evidence to substantiate a specific fault category.",
                status="unknown",
                confidence_assessment=make_unknown_ca()
            )

        # ---------------------------------------------------------
        # Rule 3: Candidate is already 'Unknown'
        # ---------------------------------------------------------
        if candidate_category == "Unknown":
            return UnknownDetectionResult(
                is_unknown=True,
                final_category="Unknown",
                decision_rule="candidate_already_unknown",
                reason="The description does not contain sufficient technical evidence to assign a specific fault category.",
                status="unknown",
                confidence_assessment=make_unknown_ca()
            )

        # ---------------------------------------------------------
        # Rule 4: Ambiguous evidence between categories
        # (For local model: top2_margin < 0.10 and candidate lacks strong dominant evidence)
        # ---------------------------------------------------------
        candidate_matches = evidence.category_evidence.get(candidate_category, [])
        competing_categories = [
            cat for cat, matches in evidence.category_evidence.items()
            if cat != candidate_category and len(matches) > 0
        ]

        if "local" in model_source.lower():
            if effective_margin is not None and effective_margin < 0.10 and len(candidate_matches) == 0:
                return UnknownDetectionResult(
                    is_unknown=True,
                    final_category="Unknown",
                    decision_rule="ambiguous_evidence",
                    reason=f"Ambiguous technical evidence with insufficient model margin ({effective_margin:.4f}) between candidate categories.",
                    status="unknown",
                    confidence_assessment=make_unknown_ca()
                )

        # ---------------------------------------------------------
        # Rule 5: Candidate category supported by technical evidence
        # ---------------------------------------------------------
        if len(candidate_matches) > 0:
            # Explicit physical/technical evidence is present
            is_low_conf = False
            if calibrated_prob is not None and calibrated_prob < 0.60:
                is_low_conf = True
            elif raw_score is not None and raw_score < 0.60:
                is_low_conf = True

            rule_name = "explicit_defect_retained_with_low_confidence" if is_low_conf else "specific_defect_supported"

            # If ca was passed, retain its calibration status, level, etc.
            final_ca = ca or ConfidenceAssessment(
                level="Low" if is_low_conf else "High",
                approximate_range=f"~{round(calibrated_prob * 100)}%" if (calibrated_prob is not None and ca and ca.is_calibrated) else "Qualitative / N/A",
                raw_score=raw_score,
                calibrated_prob=calibrated_prob,
                top2_margin=effective_margin,
                is_calibrated=bool(ca and ca.is_calibrated),
                calibration_method=ca.calibration_method if ca else None,
                is_ambiguous=is_low_conf or bool(effective_margin is not None and effective_margin < 0.15)
            )

            return UnknownDetectionResult(
                is_unknown=False,
                final_category=candidate_category,
                decision_rule=rule_name,
                reason=f"Technical evidence ({', '.join(candidate_matches[:3])}) confirms {candidate_category}.",
                status="success",
                confidence_assessment=final_ca
            )

        # ---------------------------------------------------------
        # Rule 6: Candidate lacks specific evidence with low confidence (< 0.50)
        # ---------------------------------------------------------
        score_to_check = calibrated_prob if calibrated_prob is not None else raw_score
        if score_to_check is not None and score_to_check < 0.50:
            return UnknownDetectionResult(
                is_unknown=True,
                final_category="Unknown",
                decision_rule="insufficient_evidence_for_candidate",
                reason=f"The input does not contain sufficient specific evidence to substantiate a {candidate_category} classification.",
                status="unknown",
                confidence_assessment=make_unknown_ca()
            )

        # ---------------------------------------------------------
        # Rule 7: High-confidence model prediction or Gemini proposal
        # ---------------------------------------------------------
        # Ensure category is strictly in approved taxonomy
        if candidate_category not in REQUIRED_APPROVED_CATEGORIES:
            return UnknownDetectionResult(
                is_unknown=True,
                final_category="Unknown",
                decision_rule="unapproved_taxonomy_rejection",
                reason=f"Proposed category '{candidate_category}' is not an approved taxonomy defect.",
                status="validation_error",
                confidence_assessment=make_unknown_ca()
            )

        final_ca = ca or ConfidenceAssessment(
            level="Medium",
            approximate_range="Qualitative / N/A",
            raw_score=raw_score,
            calibrated_prob=calibrated_prob,
            top2_margin=effective_margin,
            is_calibrated=bool(ca and ca.is_calibrated),
            calibration_method=ca.calibration_method if ca else None,
            is_ambiguous=False
        )

        return UnknownDetectionResult(
            is_unknown=False,
            final_category=candidate_category,
            decision_rule="model_supported_classification",
            reason=f"Classified as {candidate_category} based on model prediction and operational indicators.",
            status="success",
            confidence_assessment=final_ca
        )


# Global singleton instance
_unknown_detector: Optional[UnknownDetector] = None


def get_unknown_detector() -> UnknownDetector:
    """Returns singleton UnknownDetector instance."""
    global _unknown_detector
    if _unknown_detector is None:
        _unknown_detector = UnknownDetector()
    return _unknown_detector
