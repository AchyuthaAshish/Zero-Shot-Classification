"""Multi-Defect Detection Layer for Industrial Defect Classification.

Conforms to Phase 2 Step 2.1:
1. Detects whether an input report likely contains one defect or multiple distinct defect signals.
2. Identifies the presence of multiple distinct defect signals/evidence groups across clauses.
3. Preserves original input text without destructive segmentation or premature per-defect splitting.
4. Distinguishes:
   - Single Defect: Exactly one defect signal / evidence group.
   - Multi-Defect: Two or more distinct, co-occurring defect signals (e.g. across different categories
     or across distinct, separate equipment items with independent symptoms).
   - Connected Single Defect: Single defect failure chain with causal/explanatory connectives
     (e.g. "motor grinding noise because the bearing is damaged").
   - Ambiguity (Phase 1 Step 1.4): Uncertainty/competing evidence about which category ONE defect belongs to,
     or disjunctive alternative hypotheses ("or").
   - Unknown (Phase 1 Step 1.3): Vague, unrelated, or non-defect inputs with 0 identifiable defect signals.
5. Strictly adheres to the authoritative 8-category taxonomy.
6. Multilingual support for English, Telugu-English code-switched, and native Telugu script.
"""

import re
from typing import Dict, List, Optional, Tuple, Set, Any

from core.schemas import PreprocessedInput, MultiDefectAssessment, DefectSignal
from core.unknown_detector import (
    get_unknown_detector,
    UnknownDetector,
    CATEGORY_EVIDENCE_KEYWORDS,
    VAGUE_REGEX,
    UNRELATED_REGEX
)
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository


# ---------------------------------------------------------------------------
# Equipment and Symptom Mappings for Defect Signal Disambiguation
# ---------------------------------------------------------------------------

CATEGORY_EQUIPMENT_KEYWORDS: Dict[str, Set[str]] = {
    "Mechanical Fault": {
        "conveyor motor", "conveyor belt", "conveyor roller", "conveyor shaft", "conveyor",
        "motor bearing", "motor shaft", "motor", "bearing", "shaft", "gearbox", "gear",
        "centrifugal pump", "pump impeller", "pump", "impeller", "roller", "belt",
        "cylinder", "spindle", "valve", "compressor", "fan", "exhaust fan", "cooling fan",
        "flange", "piston", "crankshaft", "pulley", "chain", "bushing", "coupling",
        "మోటార్", "పంప్", "బేరింగ్", "గేర్", "షాఫ్ట్", "ఫ్యాన్"
    },
    "Electrical Fault": {
        "wiring", "wire", "harness", "terminal block", "terminal", "cable", "conductor",
        "winding", "stator", "armature", "contactor", "relay", "fuse", "circuit breaker",
        "breaker", "busbar", "circuit", "capacitor", "resistor", "switchboard",
        "electrical panel", "junction box", "spark plug", "transformer winding",
        "ఫ్యూజ్", "వైరింగ్", "సర్క్యూట్"
    },
    "Sensor Fault": {
        "temperature sensor", "pressure sensor", "flow meter", "level sensor", "load cell",
        "thermocouple", "rtd", "transducer", "transmitter", "probe", "proximity switch",
        "encoder", "tachometer", "accelerometer", "photocell", "sensor",
        "సెన్సార్"
    },
    "Temperature Fault": {
        "heat exchanger", "chiller", "cooling jacket", "radiator", "heater",
        "heat sink", "cooling line", "thermostat", "thermal jacket"
    },
    "Software Fault": {
        "display software", "plc program", "controller logic", "software app",
        "embedded system", "software", "application", "firmware", "hmi", "scada",
        "runtime", "driver", "code", "user interface", "ui",
        "సాఫ్ట్‌వేర్", "ప్రోగ్రామ్"
    },
    "Power Supply Fault": {
        "power supply", "smps", "ups", "battery", "dc bus", "transformer", "inverter",
        "power rail", "generator", "rectifier", "charger", "mains", "auxiliary supply",
        "power source", "incoming line", "feed line",
        "పవర్", "బ్యాటరీ", "విద్యుత్"
    },
    "Communication Fault": {
        "communication", "ethernet", "wifi", "network", "can bus", "modbus",
        "profibus", "rs485", "rs232", "lan", "gateway", "router", "switch",
        "serial port", "mqtt", "data link", "comm port", "fieldbus", "profinet",
        "plc", "controller",
        "కమ్యూనికేషన్", "నెట్‌వర్క్", "కనెక్షన్", "సిగ్నల్"
    }
}

CATEGORY_SYMPTOM_KEYWORDS: Dict[str, Set[str]] = {
    "Mechanical Fault": {
        "grinding noise", "grinding", "vibration", "vibrating", "vibrate", "wobble",
        "wobbling", "shaking", "shake", "rattle", "rattling", "knocking", "knock",
        "chatter", "chattering", "misalignment", "misaligned", "loose bolt", "cracked",
        "fractured", "wear", "worn", "seizure", "seized", "jam", "jammed", "binding",
        "friction", "unusual sound", "abnormal sound", "strange noise", "noise", "damaged",
        "వైబ్రేట్", "రాపిడి", "జామ్", "శబ్దం", "వణుకు", "సౌండ్"
    },
    "Electrical Fault": {
        "short circuit", "short", "open circuit", "sparking", "spark", "sparks", "arc",
        "arcing", "flashover", "blown fuse", "tripped breaker", "loose terminal",
        "ground fault", "insulation failure", "burnt winding", "scorched", "burning smell",
        "voltage drop", "high resistance", "overcurrent", "current spike", "burnt", "smoking",
        "smoke", "fire",
        "షార్ట్ సర్క్యూట్", "స్పార్క్", "కాలిపోయి", "మంటలు", "పొగ"
    },
    "Sensor Fault": {
        "incorrect reading", "incorrect readings", "wrong reading", "wrong readings",
        "false reading", "false readings", "erratic reading", "erratic readings",
        "inaccurate reading", "inaccurate", "calibration drift", "drift", "stuck at",
        "frozen value", "no signal", "invalid reading", "transducer failure",
        "analog input error", "out of calibration", "reading fluctuation",
        "erratic measurement", "unstable reading", "zero reading", "signal jump", "offset error",
        "faulty reading",
        "తప్పుడు రీడింగ్", "రీడింగ్ తప్పు", "సిగ్నల్ లేదు"
    },
    "Temperature Fault": {
        "overheating", "overheat", "overheated", "excessive heat", "running hot",
        "high temp", "thermal runaway", "cooling failure", "thermal overload",
        "temperature spiked", "heat accumulation", "dangerously hot", "very hot",
        "extremely hot", "boiling", "heat rise", "thermal breakdown", "high temperature",
        "వేడెక్కి", "ఓవర్‌హీట్", "ఎక్కువ వేడి", "హీట్"
    },
    "Software Fault": {
        "crash", "crashed", "crashes", "crashing", "freeze", "frozen", "freezes",
        "hung", "hanging", "bug", "exception", "null pointer", "memory leak",
        "reboot loop", "watchdog reset", "compilation error", "segmentation fault",
        "software lockup", "unresponsive interface", "unresponsive", "error code",
        "unhandled error", "runtime error", "blue screen", "infinite loop", "logic error",
        "system halt",
        "క్రాష్", "ఆగిపోవడం", "హ్యాంగ్"
    },
    "Power Supply Fault": {
        "no power", "power outage", "power loss", "power failure", "blackout",
        "brownout", "voltage sag", "low supply voltage", "zero voltage",
        "power supply dead", "battery drained", "breaker tripped on main supply",
        "unpowered", "power cut", "current collapse", "supply instability",
        "supply fluctuation", "lost incoming power", "voltage drop", "voltage dropping",
        "voltage is dropping", "dropping voltage", "voltage keeps dropping",
        "పవర్ పోయింది", "పవర్ లేదు", "వోల్టేజ్ పడిపోయింది"
    },
    "Communication Fault": {
        "disconnected", "connection lost", "timeout", "packet loss", "communication loss",
        "communication failure", "link down", "unreachable", "transmission error",
        "frame error", "dropped packets", "no handshake", "network failure",
        "connection refused", "protocol error", "crc error", "comm error",
        "lost communication", "lost connection", "connection drop", "comm drop",
        "కనెక్షన్ కట్", "సిగ్నల్ రాలేదు", "డిస్‌కనెక్ట్"
    }
}


# ---------------------------------------------------------------------------
# Conjunction & Connective Patterns
# ---------------------------------------------------------------------------

# Causal / Explanatory Connectives (Indicate single connected defect mechanism)
CAUSAL_CONNECTIVES = [
    r"\bbecause\s+of\b",
    r"\bbecause\b",
    r"\bdue\s+to\b",
    r"\bcaused\s+by\b",
    r"\bas\s+a\s+result\s+of\b",
    r"\bresulting\s+in\b",
    r"\bleading\s+to\b",
    r"\bowing\s+to\b",
    r"\breason\s+is\b",
    r"\bvalla\b",
    r"\bkaaranangaa\b",
    r"\banduvalana\b",
    r"వల్ల",
    r"కారణంగా",
    r"అందువలన"
]
CAUSAL_REGEX = re.compile("|".join(CAUSAL_CONNECTIVES), re.IGNORECASE)

# Disjunctive Connectives (Indicate ambiguity/alternative hypotheses rather than multi-defect)
DISJUNCTIVE_CONNECTIVES = [
    r"\beither\s+.*?\s+or\b",
    r"\bor\b",
    r"\bleda\b",
    r"లేదా"
]
DISJUNCTIVE_REGEX = re.compile("|".join(DISJUNCTIVE_CONNECTIVES), re.IGNORECASE)

# Coordinating Conjunctions (Connecting independent defect clauses)
COORDINATING_PATTERNS = [
    r"\band\s+also\b",
    r"\bas\s+well\s+as\b",
    r"\bin\s+addition\s+to\b",
    r"\balong\s+with\b",
    r"\badditionally\b",
    r"\bfurthermore\b",
    r"\band\b",
    r"\bplus\b",
    r"\balso\b",
    r"\bmariyu\b",
    r"\binkaa\b",
    r"మరియు",
    r"అలాగే",
    r"కూడా"
]
COORDINATING_REGEX = re.compile("|".join(COORDINATING_PATTERNS), re.IGNORECASE)


class MultiDefectDetector:
    """Dedicated detector evaluating whether defect descriptions contain multiple defect signals."""

    def __init__(self, unknown_detector: Optional[UnknownDetector] = None, taxonomy_repo=None):
        self.unknown_detector = unknown_detector or get_unknown_detector()
        self.taxonomy_repo = taxonomy_repo or get_taxonomy_repository()

    def detect_multi_defect(
        self,
        raw_or_preprocessed: Any,
        candidate_category: Optional[str] = None
    ) -> MultiDefectAssessment:
        """Evaluates whether the input text contains multiple defect signals.

        Returns a structured MultiDefectAssessment.
        """
        # 1. Resolve raw text
        if hasattr(raw_or_preprocessed, "raw_text"):
            text = raw_or_preprocessed.raw_text
        elif hasattr(raw_or_preprocessed, "normalized_text"):
            text = raw_or_preprocessed.normalized_text
        else:
            text = str(raw_or_preprocessed or "")

        clean_text = text.strip()
        if not clean_text:
            return MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=0,
                signals=[],
                detection_reason="Empty input text contains no defect signals.",
                method="rule_based_evidence_partitioning",
                primary_candidate_category="Unknown"
            )

        # 2. Check for Non-Defect / Unrelated Input (Step 1.3 Unknown)
        ev_analysis = self.unknown_detector.analyze_evidence(clean_text)
        if ev_analysis.is_unrelated_or_non_defect:
            return MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=0,
                signals=[],
                detection_reason="Unrelated or non-defect input does not contain industrial defect signals.",
                method="rule_based_evidence_partitioning",
                primary_candidate_category="Unknown"
            )

        # 3. Check for Vague Malfunction lacking physical symptoms (Step 1.3 Unknown)
        if ev_analysis.is_vague and ev_analysis.total_evidence_count == 0:
            return MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=0,
                signals=[],
                detection_reason="Vague description lacking technical evidence does not contain identifiable defect signals.",
                method="rule_based_evidence_partitioning",
                primary_candidate_category="Unknown"
            )

        # 4. Check for Disjunctive Alternatives ("or" / Ambiguity)
        # Note: If clauses are connected with disjunctive "or", it represents alternative hypotheses
        # for a single incident (Ambiguity), NOT multiple co-occurring defects!
        has_disjunctive = bool(DISJUNCTIVE_REGEX.search(clean_text))
        has_coordinating = bool(COORDINATING_REGEX.search(clean_text)) or ("," in clean_text) or (";" in clean_text)
        has_causal = bool(CAUSAL_REGEX.search(clean_text))

        if has_disjunctive and not has_coordinating:
            primary_cat = candidate_category if (candidate_category and candidate_category != "Unknown") else self._best_category_for_text(clean_text)
            return MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=1,
                signals=[
                    DefectSignal(
                        signal_id=1,
                        category=primary_cat,
                        text_span=clean_text,
                        symptom="alternative hypothesis (disjunctive)",
                        keywords=self._extract_matched_keywords(clean_text, primary_cat)
                    )
                ],
                detection_reason="Input describes alternative hypothetical causes (ambiguity) rather than multiple co-occurring defects.",
                method="rule_based_evidence_partitioning",
                has_coordinating_conjunction=False,
                has_causal_relation=False,
                primary_candidate_category=primary_cat
            )

        # 5. Partition Text into Candidate Defect Clauses
        raw_spans = self._partition_clauses(clean_text)

        # 6. Extract Defect Signals from candidate spans
        extracted_signals: List[DefectSignal] = []
        signal_id_counter = 1

        for span in raw_spans:
            sig = self._analyze_span(span, signal_id_counter)
            if sig is not None:
                extracted_signals.append(sig)
                signal_id_counter += 1

        # 7. Post-Processing & Boundary Analysis
        # If no valid defect signals found (e.g. trivial/unknown text)
        if not extracted_signals:
            primary_cat = candidate_category or "Unknown"
            count = 1 if primary_cat != "Unknown" else 0
            return MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=count,
                signals=[],
                detection_reason="No distinct defect evidence groups identified in input.",
                method="rule_based_evidence_partitioning",
                has_coordinating_conjunction=has_coordinating,
                has_causal_relation=has_causal,
                primary_candidate_category=primary_cat
            )

        # If exactly one defect signal found
        if len(extracted_signals) == 1:
            return MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=1,
                signals=extracted_signals,
                detection_reason=f"Single defect signal identified for {extracted_signals[0].category}.",
                method="rule_based_evidence_partitioning",
                has_coordinating_conjunction=has_coordinating,
                has_causal_relation=has_causal,
                primary_candidate_category=extracted_signals[0].category
            )

        # 8. Evaluation of Multiple Signals (Causal vs Independent)
        # Check if signals are causally linked (e.g. "motor grinding noise because bearing damaged")
        categories_in_signals = [s.category for s in extracted_signals]
        distinct_categories = set(categories_in_signals)
        equipment_in_signals = [s.equipment for s in extracted_signals if s.equipment]
        distinct_equipment = set(equipment_in_signals)

        # Case A: Same category with Causal Relation
        # E.g. "The motor is making a grinding noise because the bearing is damaged."
        # Connected single defect chain within the same subsystem.
        if len(distinct_categories) == 1 and has_causal:
            cat = list(distinct_categories)[0]
            merged_signal = DefectSignal(
                signal_id=1,
                category=cat,
                text_span=clean_text,
                equipment=extracted_signals[0].equipment,
                symptom="; ".join(s.symptom for s in extracted_signals if s.symptom),
                keywords=list(set(kw for s in extracted_signals for kw in s.keywords))
            )
            return MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=1,
                signals=[merged_signal],
                detection_reason=f"Connected single defect: secondary clause provides causal explanation within {cat}.",
                method="rule_based_evidence_partitioning",
                has_coordinating_conjunction=has_coordinating,
                has_causal_relation=True,
                primary_candidate_category=cat
            )

        # Case B: Same category, same equipment, multiple descriptive symptoms without causal connective
        # E.g. "The motor is making a grinding noise and motor is vibrating heavily."
        # Multiple descriptive symptoms for the same machine defect.
        if len(distinct_categories) == 1 and len(distinct_equipment) <= 1:
            cat = list(distinct_categories)[0]
            merged_signal = DefectSignal(
                signal_id=1,
                category=cat,
                text_span=clean_text,
                equipment=extracted_signals[0].equipment,
                symptom="; ".join(s.symptom for s in extracted_signals if s.symptom),
                keywords=list(set(kw for s in extracted_signals for kw in s.keywords))
            )
            return MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=1,
                signals=[merged_signal],
                detection_reason=f"Single defect with multiple descriptive symptoms for the same equipment ({cat}).",
                method="rule_based_evidence_partitioning",
                has_coordinating_conjunction=has_coordinating,
                has_causal_relation=has_causal,
                primary_candidate_category=cat
            )

        # Case C: Same category, but clearly distinct, separate physical equipment
        # E.g. "The conveyor motor is vibrating and the exhaust fan has a loose bolt."
        # Two independent mechanical defects on distinct machines!
        if len(distinct_categories) == 1 and len(distinct_equipment) >= 2 and not has_causal:
            equip_desc = " and ".join(f"{s.equipment} ('{s.text_span}')" for s in extracted_signals if s.equipment)
            return MultiDefectAssessment(
                is_multi_defect=True,
                defect_signal_count=len(extracted_signals),
                signals=extracted_signals,
                detection_reason=f"Detected {len(extracted_signals)} distinct defect signals across separate equipment: {equip_desc}.",
                method="rule_based_evidence_partitioning",
                has_coordinating_conjunction=has_coordinating,
                has_causal_relation=False,
                primary_candidate_category=candidate_category or list(distinct_categories)[0]
            )

        # Case D: Multiple distinct categories (True Multi-Defect!)
        # E.g. "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        # E.g. "Voltage is dropping and the PLC lost communication with the controller."
        # E.g. "The machine is overheating, the bearing is vibrating, and the display software crashes."
        cat_descriptions = [f"{s.category} ('{s.text_span}')" for s in extracted_signals]
        reason = f"Detected {len(extracted_signals)} distinct defect signals across multiple categories: {', '.join(cat_descriptions)}."

        primary_cat = candidate_category if (candidate_category and candidate_category != "Unknown") else extracted_signals[0].category

        return MultiDefectAssessment(
            is_multi_defect=True,
            defect_signal_count=len(extracted_signals),
            signals=extracted_signals,
            detection_reason=reason,
            method="rule_based_evidence_partitioning",
            has_coordinating_conjunction=has_coordinating,
            has_causal_relation=has_causal,
            primary_candidate_category=primary_cat
        )

    # -----------------------------------------------------------------------
    # Clause Partitioning & Linguistic Helpers
    # -----------------------------------------------------------------------

    def _partition_clauses(self, text: str) -> List[str]:
        """Partitions input text into candidate defect clauses based on delimiters,
        coordinating conjunctions, and causal markers.
        """
        # Split on sentence terminals and semicolons first
        major_chunks = re.split(r'[;\n]+', text)
        clauses: List[str] = []

        split_pattern = (
            r'(?:\s*,\s*(?:and\s+also|as\s+well\s+as|in\s+addition\s+to|along\s+with|and|plus|also|mariyu|inkaa|మరియు|అలాగే)?\s*'
            r'|\s+(?:and\s+also|as\s+well\s+as|in\s+addition\s+to|along\s+with|and|plus|also|mariyu|inkaa|మరియు|అలాగే)\s+'
            r'|\s+(?:because\s+of|because|due\s+to|caused\s+by|as\s+a\s+result\s+of|resulting\s+in|leading\s+to|owing\s+to|valla|kaaranangaa|anduvalana|వల్ల|కారణంగా|అందువలన)\s+)'
        )

        for chunk in major_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # Split by comma-conjunction, coordinating conjunction, or causal connective
            sub_spans = re.split(split_pattern, chunk, flags=re.IGNORECASE)
            for s in sub_spans:
                s_clean = s.strip().strip(",").strip(".")
                if s_clean and len(s_clean) >= 3:
                    clauses.append(s_clean)

        return clauses if clauses else [text.strip()]

    def _analyze_span(self, span: str, signal_id: int) -> Optional[DefectSignal]:
        """Analyzes a candidate text span to determine if it contains a valid defect signal."""
        span_lower = span.lower()

        # 1. Detect equipment mentioned
        found_equipment: Optional[str] = None
        longest_equipment_len = 0

        for cat, equip_set in CATEGORY_EQUIPMENT_KEYWORDS.items():
            for equip in equip_set:
                if equip in span_lower and len(equip) > longest_equipment_len:
                    found_equipment = equip
                    longest_equipment_len = len(equip)

        # 2. Detect symptoms mentioned
        found_symptom: Optional[str] = None
        longest_symptom_len = 0
        symptom_cat: Optional[str] = None

        for cat, sym_set in CATEGORY_SYMPTOM_KEYWORDS.items():
            for sym in sym_set:
                if sym in span_lower and len(sym) > longest_symptom_len:
                    found_symptom = sym
                    longest_symptom_len = len(sym)
                    symptom_cat = cat

        # 3. Detect general category keywords
        cat_scores: Dict[str, int] = {cat: 0 for cat in REQUIRED_APPROVED_CATEGORIES if cat != "Unknown"}
        cat_matched_keywords: Dict[str, List[str]] = {cat: [] for cat in cat_scores}

        for cat, kw_set in CATEGORY_EVIDENCE_KEYWORDS.items():
            if cat not in cat_scores:
                continue
            for kw in kw_set:
                if kw in span_lower:
                    cat_scores[cat] += len(kw.split()) + 1
                    cat_matched_keywords[cat].append(kw)

        # Bonus weight if a symptom matched this category
        if symptom_cat and symptom_cat in cat_scores:
            cat_scores[symptom_cat] += 4

        # Filter categories with positive evidence
        active_cats = [(cat, score) for cat, score in cat_scores.items() if score > 0]
        if not active_cats:
            return None

        # Sort by score
        active_cats.sort(key=lambda x: x[1], reverse=True)
        best_cat, best_score = active_cats[0]

        # A valid defect signal MUST have either:
        # (a) an explicit symptom, OR
        # (b) a strong category evidence score >= 3 (e.g. multi-word defect phrase like "short circuit")
        if found_symptom is None and best_score < 3:
            return None

        matched_kws = cat_matched_keywords[best_cat]
        if found_symptom and found_symptom not in matched_kws:
            matched_kws.append(found_symptom)
        if found_equipment and found_equipment not in matched_kws:
            matched_kws.append(found_equipment)

        return DefectSignal(
            signal_id=signal_id,
            category=best_cat,
            text_span=span.strip(),
            equipment=found_equipment,
            symptom=found_symptom or "reported anomaly",
            keywords=sorted(list(set(matched_kws)))
        )

    def _best_category_for_text(self, text: str) -> str:
        """Finds the best matching canonical category for the text based on evidence keywords."""
        lower = text.lower()
        best_cat = "Unknown"
        best_score = 0

        for cat, kw_set in CATEGORY_EVIDENCE_KEYWORDS.items():
            score = sum(1 for kw in kw_set if kw in lower)
            if score > best_score:
                best_score = score
                best_cat = cat

        return best_cat

    def _extract_matched_keywords(self, text: str, category: str) -> List[str]:
        """Extracts matched keywords for a given category."""
        lower = text.lower()
        keywords = CATEGORY_EVIDENCE_KEYWORDS.get(category, set())
        return [kw for kw in keywords if kw in lower]


# ---------------------------------------------------------------------------
# Global Singleton & Public Service Function
# ---------------------------------------------------------------------------

_multi_defect_detector_instance: Optional[MultiDefectDetector] = None


def get_multi_defect_detector() -> MultiDefectDetector:
    """Returns singleton MultiDefectDetector instance."""
    global _multi_defect_detector_instance
    if _multi_defect_detector_instance is None:
        _multi_defect_detector_instance = MultiDefectDetector()
    return _multi_defect_detector_instance


def detect_multi_defect(
    raw_or_preprocessed: Any,
    candidate_category: Optional[str] = None
) -> MultiDefectAssessment:
    """Public convenience function to evaluate multi-defect presence on input text."""
    detector = get_multi_defect_detector()
    return detector.detect_multi_defect(raw_or_preprocessed, candidate_category=candidate_category)
