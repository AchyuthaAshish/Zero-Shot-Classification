"""Dynamic, Input-Aware Explanation Generator for Defect Classification.

Produces concise, varied, evidence-based natural language explanations
directly grounded in the user's actual defect description without calling external LLM APIs.
Conforms strictly to the 8-category taxonomy and never invents unmentioned component failures.
"""

import re
import random
from typing import Optional, NamedTuple


class EvidenceMatch(NamedTuple):
    symptom: str
    equipment: Optional[str]
    detail: str


def extract_evidence(text: str, category: str) -> EvidenceMatch:
    """
    Extracts the specific physical symptom, affected equipment, and technical
    interpretation directly from the user's defect description.
    """
    lower = text.lower()

    # --- 1. Equipment Extraction ---
    equipment = None
    if "conveyor shaft" in lower:
        equipment = "the conveyor shaft"
    elif "conveyor motor" in lower:
        equipment = "the conveyor motor"
    elif "conveyor roller" in lower:
        equipment = "the conveyor roller"
    elif "conveyor belt" in lower:
        equipment = "the conveyor belt"
    elif "conveyor" in lower:
        equipment = "the conveyor"
    elif "motor bearing" in lower:
        equipment = "the motor bearing"
    elif "motor shaft" in lower:
        equipment = "the motor shaft"
    elif "motor" in lower or "మోటార్" in lower:
        equipment = "the motor"
    elif "bearing" in lower:
        equipment = "the bearing"
    elif "shaft" in lower:
        equipment = "the shaft"
    elif "gearbox" in lower or "gear" in lower:
        equipment = "the gearbox"
    elif "centrifugal pump" in lower or "pump impeller" in lower or "pump" in lower or "పంప్" in lower:
        equipment = "the pump"
    elif "plc" in lower and "controller" in lower:
        equipment = "the PLC and controller"
    elif "plc" in lower:
        equipment = "the PLC"
    elif "temperature sensor" in lower:
        equipment = "the temperature sensor"
    elif "pressure sensor" in lower or "pressure transmitter" in lower:
        equipment = "the pressure sensor"
    elif "sensor" in lower or "సెన్సార్" in lower:
        equipment = "the sensor"
    elif "firmware" in lower and "device" in lower:
        equipment = "the firmware on the device"
    elif "firmware" in lower:
        equipment = "the firmware"
    elif "controller" in lower:
        equipment = "the controller"
    elif "power supply" in lower or "smps" in lower:
        equipment = "the power supply"
    elif "battery" in lower or "ups" in lower:
        equipment = "the battery backup"
    elif "transformer" in lower:
        equipment = "the transformer"
    elif "circuit breaker" in lower or "breaker" in lower:
        equipment = "the circuit breaker"
    elif "machine" in lower or "మెషిన్" in lower:
        equipment = "the machine"
    elif "device" in lower:
        equipment = "the device"

    # --- 2. Category-Specific Symptom & Detail Extraction ---
    symptom = None
    detail = None

    if category == "Mechanical Fault":
        if any(w in lower for w in ["grinding noise", "grinding sound", "grinding", "రాపిడి"]):
            symptom = "grinding noise"
            detail = "abnormal friction in a moving mechanical component"
        elif any(w in lower for w in ["loud rattling sound", "rattling sound", "rattling noise", "rattling"]):
            symptom = "rattling sound"
            detail = "mechanical looseness or component collision"
        elif any(w in lower for w in ["misaligned", "misalignment"]):
            symptom = "shaft misalignment" if "shaft" in lower else "mechanical misalignment"
            detail = "a physical alignment error in the mechanical assembly"
        elif any(w in lower for w in ["vibrat", "వైబ్రేట్", "wobble", "shaking"]):
            symptom = "heavy vibration" if any(w in lower for w in ["heavy", "excessive", "loud"]) else "abnormal vibration"
            detail = "rotational imbalance or mechanical wear"
        elif any(w in lower for w in ["bearing wear", "worn bearing"]):
            symptom = "bearing wear"
            detail = "mechanical degradation of the bearing"
        elif any(w in lower for w in ["jam", "seiz", "stuck", "జామ్", "binding"]):
            symptom = "mechanical binding"
            detail = "physical obstruction or motion arrest"
        else:
            symptom = "mechanical irregularity"
            detail = "abnormal physical operation"

    elif category == "Electrical Fault":
        if any(w in lower for w in ["short circuit", "షార్ట్ సర్క్యూట్", "ground fault"]):
            symptom = "short circuit"
            detail = "an abnormal connection between electrical conductors"
        elif any(w in lower for w in ["spark", "arc", "స్పార్క్", "flashover"]):
            symptom = "sparking"
            detail = "electrical discharge or dielectric failure"
        elif any(w in lower for w in ["tripped", "breaker", "fuse", "ఫ్యూజ్"]):
            symptom = "circuit breaker trip" if "breaker" in lower else "fuse blowout"
            detail = "an electrical overcurrent or protection trip"
        elif any(w in lower for w in ["burn", "smoke", "కాలిపోయి", "scorch"]):
            symptom = "burning smell or scorched insulation"
            detail = "severe electrical overheating or insulation damage"
        elif any(w in lower for w in ["loose wire", "loose terminal", "wiring"]):
            symptom = "loose wiring connection"
            detail = "poor contact or terminal degradation"
        else:
            symptom = "electrical anomaly"
            detail = "an abnormal condition in the electrical circuit"

    elif category == "Sensor Fault":
        if any(w in lower for w in ["incorrect reading", "incorrect value", "wrong reading", "inaccurate", "erratic", "తప్పుడు"]):
            symptom = "inaccurate measurements"
            detail = "a measurement discrepancy or transducer inaccuracy"
        elif any(w in lower for w in ["drift", "calibration"]):
            symptom = "calibration drift"
            detail = "deviation in the sensor's measurement baseline"
        elif any(w in lower for w in ["no signal", "signal lost", "stuck at", "frozen", "output locked"]):
            symptom = "loss of sensor signal"
            detail = "a sensing element failure or frozen output"
        else:
            symptom = "sensor reading issue"
            detail = "a problem with sensor measurement or instrumentation"

    elif category == "Temperature Fault":
        if any(w in lower for w in ["overheat", "ఓవర్‌హీట్"]):
            symptom = "overheating"
            detail = "excessive thermal buildup beyond safe operating limits"
        elif any(w in lower for w in ["very hot", "extremely hot", "dangerously hot", "running hot", "వేడెక్కి"]):
            symptom = "elevated temperature"
            detail = "abnormal heat accumulation"
        elif any(w in lower for w in ["cooling failure", "chiller", "heat exchanger"]):
            symptom = "cooling failure"
            detail = "inadequate heat dissipation"
        else:
            symptom = "thermal excess"
            detail = "an abnormal temperature condition"

    elif category == "Software Fault":
        if any(w in lower for w in ["crash", "క్రాష్"]):
            if any(w in lower for w in ["whenever the device starts", "startup", "boot"]):
                symptom = "startup crash"
            else:
                symptom = "software crash"
            detail = "an unhandled software exception or fatal application error"
        elif any(w in lower for w in ["freez", "froze", "hung", "hangs", "not responding", "ఫ్రీజ్"]):
            symptom = "system freeze"
            detail = "a process deadlock or unresponsive execution state"
        elif any(w in lower for w in ["watchdog", "timeout"]):
            symptom = "watchdog timeout"
            detail = "task starvation or firmware execution hang"
        elif any(w in lower for w in ["runtime error", "null pointer", "segmentation fault"]):
            symptom = "runtime exception"
            detail = "a logic defect or invalid memory access"
        else:
            symptom = "software abnormality"
            detail = "abnormal program or firmware behavior"

    elif category == "Power Supply Fault":
        if "voltage" in lower and any(w in lower for w in ["drop", "sag", "keeps dropping", "పడిపోయి", "dropped"]):
            symptom = "repeated voltage drop" if "keeps dropping" in lower else "voltage drop"
            detail = "instability in the electrical power delivery rail"
        elif any(w in lower for w in ["unstable voltage", "fluctuat", "voltage is unstable"]):
            symptom = "unstable supply voltage"
            detail = "power rail fluctuation"
        elif any(w in lower for w in ["battery", "ups"]):
            symptom = "battery backup issue"
            detail = "power storage or auxiliary supply depletion"
        elif any(w in lower for w in ["smps", "power supply unit"]):
            symptom = "power supply unit failure"
            detail = "an internal power converter fault"
        else:
            symptom = "power supply irregularity"
            detail = "insufficient or unstable electrical supply"

    elif category == "Communication Fault":
        if any(w in lower for w in ["lost communication", "no communication", "cannot communicate", "కమ్యూనికేషన్"]):
            symptom = "loss of communication"
            detail = "a disrupted data link between interconnected devices"
        elif any(w in lower for w in ["timeout", "socket"]):
            symptom = "connection timeout"
            detail = "inability to maintain active network response"
        elif any(w in lower for w in ["link down", "disconnected", "cable disconnected"]):
            symptom = "link disconnection"
            detail = "interrupted network cabling or port failure"
        elif any(w in lower for w in ["packet loss", "crc", "transmission error"]):
            symptom = "packet transmission loss"
            detail = "data corruption across the network interface"
        else:
            symptom = "communication breakdown"
            detail = "a network protocol or data transmission failure"

    else:
        # Unknown
        symptom = "general operational issue"
        detail = "unspecified machine trouble"

    return EvidenceMatch(symptom=symptom, equipment=equipment, detail=detail)


def generate_local_explanation(
    text: str,
    category: str,
    confidence: Optional[float] = None,
    raw_text: Optional[str] = None
) -> str:
    """
    Generates a dynamic, evidence-grounded natural-language explanation
    derived from the actual user input.
    
    Provides varied phrasing without inventing unsupported component failures.
    Operates 100% offline without external API calls.
    """
    input_text = raw_text or text or ""
    ev = extract_evidence(input_text, category)
    eq = ev.equipment
    symptom = ev.symptom
    detail = ev.detail

    # --- Unknown Category ---
    if category == "Unknown":
        unknown_templates = [
            "The description does not provide enough specific technical evidence to associate the issue with a defined fault category.",
            "The reported symptom is too vague to determine the underlying fault type reliably.",
            "There is insufficient technical detail in the description to make an authoritative classification.",
            "The input describes general machine trouble without detailing the physical or operational symptoms needed for an exact category.",
            "A general operational problem was reported without mentioning specific technical symptoms or affected components."
        ]
        return random.choice(unknown_templates)

    # --- Category Explanations with Varied Sentence Structures ---
    templates = []

    if category == "Mechanical Fault":
        if eq and symptom:
            templates = [
                f"The {symptom} coming from {eq} suggests {detail}, supporting a Mechanical Fault classification.",
                f"The reported {symptom} on {eq} indicates a physical problem affecting the machine's moving parts.",
                f"The description highlights a {symptom} from {eq}, pointing toward abnormal mechanical behavior.",
                f"Observing a {symptom} in {eq} indicates {detail}, consistent with a Mechanical Fault.",
                f"Evidence of {symptom} associated with {eq} reflects physical component wear or friction.",
                f"The defect description focuses on a {symptom} affecting {eq}, which confirms a Mechanical Fault."
            ]
        elif eq:
            templates = [
                f"The physical symptom reported for {eq} points toward abnormal mechanical behavior in the assembly.",
                f"The description reports a physical issue affecting {eq}, supporting a Mechanical Fault classification.",
                f"Abnormal mechanical operation on {eq} indicates physical component wear or friction.",
                f"Physical irregularities affecting {eq} reflect a mechanical operational fault."
            ]
        elif symptom:
            templates = [
                f"The {symptom} described in the input suggests {detail}, characteristic of a Mechanical Fault.",
                f"Reporting a {symptom} points toward abnormal physical friction or movement in the machinery.",
                f"The mention of a {symptom} indicates an issue with moving mechanical components.",
                f"Observing a {symptom} reflects mechanical wear or physical movement disruption."
            ]
        else:
            templates = [
                "The physical symptoms described in the input indicate abnormal behavior in the machine's mechanical components.",
                "The description describes physical movement or friction consistent with a Mechanical Fault."
            ]

    elif category == "Electrical Fault":
        if eq and symptom:
            templates = [
                f"The {symptom} reported in {eq} indicates {detail}, pointing to an Electrical Fault.",
                f"Evidence of {symptom} affecting {eq} suggests an electrical circuit abnormality or insulation issue.",
                f"The description highlights {symptom} in {eq}, which points directly to an electrical malfunction.",
                f"Reporting {symptom} associated with {eq} indicates dielectric breakdown or an electrical circuit defect.",
                f"The electrical symptom observed in {eq} reflects {detail}, characteristic of an Electrical Fault."
            ]
        elif eq:
            templates = [
                f"The electrical abnormality observed in {eq} indicates a circuit or wiring issue.",
                f"The description reports an electrical malfunction affecting {eq}, supporting an Electrical Fault classification.",
                f"Circuit irregularity reported on {eq} reflects an electrical fault condition."
            ]
        elif symptom:
            templates = [
                f"The mention of {symptom} indicates {detail}, supporting an Electrical Fault classification.",
                f"The reported {symptom} points toward an electrical circuit failure or connection breakdown.",
                f"Observing {symptom} reflects electrical current leakage or component insulation failure."
            ]
        else:
            templates = [
                "The reported symptoms indicate an electrical circuit or wiring problem.",
                "The description describes an electrical abnormality consistent with an Electrical Fault."
            ]

    elif category == "Sensor Fault":
        if eq and symptom:
            templates = [
                f"{eq.capitalize()} is reported to produce {symptom}, which directly indicates a Sensor Fault.",
                f"The {symptom} observed from {eq} point to {detail}.",
                f"The description highlights {symptom} regarding {eq}, indicating an instrumentation discrepancy rather than a mechanical failure.",
                f"Reporting {symptom} on {eq} suggests a transducer signal discrepancy or calibration issue.",
                f"An observation of {symptom} from {eq} confirms an instrumentation measurement fault."
            ]
        elif eq:
            templates = [
                f"The instrumentation reading reported from {eq} indicates a sensor measurement issue.",
                f"The description points to an inaccurate signal from {eq}, characteristic of a Sensor Fault.",
                f"A transducer signal error on {eq} reflects an instrumentation calibration defect."
            ]
        elif symptom:
            templates = [
                f"The {symptom} described in the input indicates {detail}, supporting a Sensor Fault.",
                f"Reporting {symptom} points to an instrumentation or transducer measurement defect.",
                f"An observation of {symptom} reflects sensor measurement inaccuracies or signal loss."
            ]
        else:
            templates = [
                "The input describes an inaccurate or missing sensor reading, indicating a Sensor Fault.",
                "The reported symptom points to measurement inaccuracy or sensor instrumentation failure."
            ]

    elif category == "Temperature Fault":
        if eq and symptom:
            templates = [
                f"The {symptom} reported on {eq} indicates {detail}, supporting a Temperature Fault classification.",
                f"Evidence of {symptom} affecting {eq} points to abnormal thermal conditions exceeding safe limits.",
                f"The description highlights {symptom} on {eq}, indicating severe thermal excess or cooling deficiency.",
                f"Reporting that {eq} shows {symptom} indicates elevated heat accumulation requiring thermal inspection.",
                f"Thermal excess described on {eq} reflects an abnormal temperature condition beyond standard operating thresholds."
            ]
        elif eq:
            templates = [
                f"The thermal condition described on {eq} indicates temperature elevation beyond normal limits.",
                f"Elevated operating heat reported for {eq} points toward a Temperature Fault.",
                f"High thermal readings on {eq} reflect excessive heat accumulation or cooling failure."
            ]
        elif symptom:
            templates = [
                f"The {symptom} described in the input reflects {detail}, characteristic of a Temperature Fault.",
                f"Reporting {symptom} points directly to abnormal thermal elevation or cooling failure.",
                f"Severe thermal buildup described as {symptom} supports a Temperature Fault classification."
            ]
        else:
            templates = [
                "The thermal excess reported in the description indicates a Temperature Fault.",
                "The input describes elevated temperatures exceeding standard operational safety boundaries."
            ]

    elif category == "Software Fault":
        if eq and symptom:
            templates = [
                f"The {symptom} affecting {eq} indicates {detail}, characteristic of a Software Fault.",
                f"The reported {symptom} in {eq} points directly to a program execution or firmware failure.",
                f"The description highlights a {symptom} on {eq}, reflecting abnormal software behavior.",
                f"Experiencing a {symptom} on {eq} points to an unhandled exception or software logic defect.",
                f"Evidence of a {symptom} during device operation confirms a software or firmware malfunction."
            ]
        elif eq:
            templates = [
                f"The software problem reported for {eq} indicates an unhandled runtime error or application crash.",
                f"A software failure affecting {eq} points toward a control program or firmware issue.",
                f"Application execution failure on {eq} reflects program logic termination or deadlock."
            ]
        elif symptom:
            templates = [
                f"The {symptom} described in the input indicates {detail}, supporting a Software Fault classification.",
                f"Reporting a {symptom} points to a software execution exception or logic error.",
                f"An occurrence of a {symptom} indicates firmware crash or application lockup."
            ]
        else:
            templates = [
                "The software abnormality reported in the description indicates a program or firmware fault.",
                "The input describes an unhandled software crash or system lockup."
            ]

    elif category == "Power Supply Fault":
        if eq and symptom:
            templates = [
                f"The {symptom} affecting {eq} points to {detail}, supporting a Power Supply Fault classification.",
                f"The description highlights a {symptom} supplied to {eq}, indicating power source instability rather than an internal device fault.",
                f"Reporting a {symptom} on {eq} reflects an electrical power delivery failure.",
                f"A {symptom} affecting {eq} indicates an issue with the electrical power rail or incoming voltage supply.",
                f"Evidence of a {symptom} concerning {eq} points directly to supply power instability."
            ]
        elif eq:
            templates = [
                f"The power irregularity affecting {eq} indicates incoming electrical supply instability.",
                f"The description reports unstable power delivery to {eq}, characteristic of a Power Supply Fault.",
                f"Electrical supply fluctuations on {eq} reflect auxiliary or source power rail deficiency."
            ]
        elif symptom:
            templates = [
                f"The {symptom} described in the input points to {detail}, indicating a Power Supply Fault.",
                f"Reporting a {symptom} reflects electrical power delivery instability or auxiliary supply depletion.",
                f"An observation of {symptom} points directly to supply voltage collapse or power interruption."
            ]
        else:
            templates = [
                "The reported symptoms indicate electrical power instability or supply failure.",
                "The description describes an incoming power delivery problem consistent with a Power Supply Fault."
            ]

    elif category == "Communication Fault":
        if eq and symptom:
            templates = [
                f"The {symptom} between {eq} points directly to a communication-related problem.",
                f"The reported {symptom} involving {eq} indicates {detail}, characteristic of a Communication Fault.",
                f"The description highlights a {symptom} affecting {eq}, indicating a network data transmission failure.",
                f"A {symptom} between {eq} indicates an interrupted data link or industrial bus disruption.",
                f"The failure of data exchange involving {eq} confirms an industrial communication link fault."
            ]
        elif eq:
            templates = [
                f"The communication issue reported for {eq} points to a network link or protocol disruption.",
                f"A data exchange failure involving {eq} indicates an industrial network fault.",
                f"Network transmission errors affecting {eq} reflect a communication protocol breakdown."
            ]
        elif symptom:
            templates = [
                f"The {symptom} described in the input points to {detail}, supporting a Communication Fault classification.",
                f"Reporting a {symptom} indicates an industrial network disruption or packet transmission failure.",
                f"Data loss described as {symptom} confirms a communication link failure."
            ]
        else:
            templates = [
                "The reported symptom indicates a loss of data communication or network link disruption.",
                "The description describes an industrial protocol or data link failure."
            ]

    elif category == "Unknown":
        return "Insufficient evidence to assign a specific defect category."
    else:
        templates = [
            f"The description was classified as {category} based on observed physical and operational symptoms."
        ]

    return random.choice(templates)
