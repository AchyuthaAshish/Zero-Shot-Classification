"""Classifier View for Industrial Defect Classification.

Streamlit frontend component implementing one simple presentable screen.
Supports Local ML, Hybrid, and Gemini classification modes.
Displays model source and confidence alongside category and explanation.
"""

import streamlit as st
from classification.classifier import classify_defect
from config.settings import get_settings


from persistence.repository import (
    save_defect_report,
    save_multi_defect_report,
    is_supabase_configured,
    DatabaseError,
    MultiDefectPersistenceError
)


def render_classifier_view() -> None:
    """Renders the single-screen industrial defect classifier interface."""
    st.title("Industrial Defect Classifier")
    st.markdown(
        "Multi-lingual AI classification of industrial machine and equipment defects. "
        "Powered by **Free Local ML** with optional **Gemini Zero-Shot Fallback**."
    )

    current_settings = get_settings()
    ext_provider_label = "AI/ML API (GLM 5 Turbo)" if current_settings.llm_provider == "aimlapi" else "Gemini"

    # Clean Mode Selector
    mode_options = {
        f"Hybrid (Local ML + {ext_provider_label} Fallback)": "hybrid",
        "Local ML (100% Free Offline)": "local",
        f"{ext_provider_label} (Cloud Zero-Shot)": "gemini"
    }

    current_env_mode = current_settings.classification_mode.lower()
    default_idx = 0
    if current_env_mode == "local":
        default_idx = 1
    elif current_env_mode in ("gemini", "external"):
        default_idx = 2

    selected_label = st.radio(
        label="Classification Mode",
        options=list(mode_options.keys()),
        index=default_idx,
        horizontal=True,
        help=f"Local ML runs 100% offline on your device without API calls. Hybrid uses Local ML first and {ext_provider_label} as fallback for uncertain cases."
    )
    active_mode = mode_options[selected_label]

    col_rep, col_emp = st.columns(2)
    with col_rep:
        reporter_name = st.text_input(
            label="Reporter Name",
            placeholder="e.g. Demo Employee",
            key="input_reporter_name"
        )
    with col_emp:
        employee_id = st.text_input(
            label="Employee ID",
            placeholder="e.g. DEMO-001",
            key="input_employee_id"
        )

    st.caption("Type the defect in your own words. English, Telugu, and Telugu-English are supported. You can enter any new or unseen defect description.")

    defect_text = st.text_area(
        label="Defect Description",
        placeholder="Enter any defect description in plain language (e.g., 'motor lo unusual sound ostundhi', 'pump is making a rattling sound', 'machine is getting very hot')...",
        height=130,
        help="Type any machine or equipment defect description in your own words."
    )

    classify_clicked = st.button("Classify / Submit Report", type="primary", use_container_width=True)

    if classify_clicked:
        clean_rep = reporter_name.strip()
        clean_emp = employee_id.strip()
        clean_text = defect_text.strip()

        if not clean_rep:
            st.warning("Please enter the Reporter Name before submitting.")
            return
        if not clean_emp:
            st.warning("Please enter the Employee ID before submitting.")
            return
        if not clean_text:
            st.warning("Please enter the Defect Description before submitting.")
            return

        with st.spinner("Classifying defect description..."):
            try:
                result = classify_defect(clean_text, mode=active_mode)
            except Exception:
                st.error("System Error: An unexpected issue occurred while processing the request. Please try again.")
                return

        _render_result_card(result, ext_provider_label)

        # Only after successful classification, save the complete report to Supabase
        if result.status in ("success", "unknown", "low_confidence"):
            submission_sig = f"{clean_rep}|{clean_emp}|{clean_text}|{result.category}"
            if st.session_state.get("last_saved_signature") != submission_sig:
                if is_supabase_configured():
                    source_raw = getattr(result, "model_source", "local_ml") or "local_ml"
                    if source_raw == "local_ml":
                        actual_provider = "local"
                        actual_model = "Local ML (TF-IDF / MiniLM)"
                    elif "aimlapi" in source_raw:
                        actual_provider = "aimlapi"
                        actual_model = getattr(current_settings, "llm_model", "glm-5-turbo")
                    elif "gemini" in source_raw:
                        actual_provider = "gemini"
                        actual_model = getattr(current_settings, "llm_model", "gemini-3.8-flash")
                    else:
                        actual_provider = getattr(current_settings, "llm_provider", "local")
                        actual_model = getattr(current_settings, "llm_model", None)

                    try:
                        md_res = getattr(result, "multi_defect_classification", None)
                        if md_res and getattr(md_res, "is_multi_defect", False) and getattr(md_res, "defect_count", 0) >= 2:
                            save_multi_defect_report(
                                reporter_name=clean_rep,
                                employee_id=clean_emp,
                                classification_result=result
                            )
                        else:
                            ca = getattr(result, "confidence_assessment", None)
                            val_res = getattr(result, "validation_result", None)
                            val_status = getattr(val_res, "status", "VALID") if val_res else "VALID"
                            save_defect_report(
                                reporter_name=clean_rep,
                                employee_id=clean_emp,
                                defect_description=clean_text,
                                category=result.category,
                                confidence=result.confidence,
                                reliability=result.reliability,
                                explanation=result.reason,
                                classification_mode=active_mode,
                                provider=actual_provider,
                                model=actual_model,
                                confidence_level=ca.level if ca else result.reliability,
                                confidence_range=ca.approximate_range if ca else None,
                                raw_score=ca.raw_score if ca else result.confidence,
                                calibrated_score=ca.calibrated_prob if ca else None,
                                is_multi_defect=False,
                                defect_count=1 if result.category != "Unknown" else 0,
                                validation_status=val_status
                            )
                        st.session_state["last_saved_signature"] = submission_sig
                        st.success("Defect report saved successfully.")
                    except (DatabaseError, MultiDefectPersistenceError) as e:
                        msg = getattr(e, "message", str(e))
                        st.warning(f"Classification completed, but the report could not be saved to the database: {msg}")
                    except Exception:
                        st.warning("Classification completed, but the report could not be saved to the database.")
                else:
                    st.info("Classification completed. (Database storage is offline: SUPABASE_URL and SUPABASE_KEY are not configured).")
            else:
                st.success("Defect report saved successfully.")


def _render_result_card(result, ext_provider_label: str = "External LLM") -> None:
    """Renders the result card based on the classification outcome and status state."""
    status = getattr(result, "status", "system_error")

    # Error State Handling: Displays clear, safe messages without raw stack traces or API internals
    if status == "input_error":
        error_msg = getattr(result, "error_message", None) or getattr(result, "reason", "Invalid input.")
        st.warning(f"**Input Error:** {error_msg}")
        return

    if status == "validation_error":
        st.error("**Validation Error:** Unable to produce a valid taxonomy category for this description.")
        return

    if status == "configuration_error":
        st.error(f"**Configuration Error:** {ext_provider_label} API key is missing or invalid. Please check your local configuration or switch to Local ML mode.")
        return

    if status == "model_error":
        raw_msg = (getattr(result, "error_message", None) or getattr(result, "reason", "")).lower()
        if any(marker in raw_msg for marker in ["rate limit", "quota", "resource_exhausted", "429", "503", "unavailable", "busy", "exhausted"]):
            st.error(f"{ext_provider_label} is temporarily busy or the request quota has been reached. Please switch to Local ML mode or wait a few minutes.")
        else:
            st.error("Unable to classify the defect right now. Please try again later or use Local ML mode.")
        return

    if status == "system_error":
        st.error("**System Error:** An internal system error occurred while processing the classification. Please try again.")
        return

    # Check for multi-defect result (Step 3)
    md_res = getattr(result, "multi_defect_classification", None)
    is_multi = bool(md_res and getattr(md_res, "is_multi_defect", False) and getattr(md_res, "defect_count", 0) >= 2)
    if is_multi:
        _render_multi_defect_result(result, md_res, ext_provider_label)
        return

    # Outcome States: success, unknown, or low_confidence (Single Defect Workflow)
    with st.container(border=True):
        st.subheader("Classification Result")

        if status == "unknown" or getattr(result, "category", "") == "Unknown":
            st.info(
                "**Outcome:** The description does not contain sufficient technical evidence "
                "to assign an authoritative defect category."
            )
        elif status == "low_confidence":
            st.warning(
                "**Low confidence:** the local model is uncertain about this classification. "
                "Review the result carefully."
            )

        amb = getattr(result, "ambiguity_assessment", None)
        if amb and amb.is_ambiguous:
            st.warning(
                f"**Ambiguity Detected:** {amb.reason}"
            )

        # Format Model Source Display
        source_raw = getattr(result, "model_source", "local_ml") or "local_ml"
        if source_raw == "local_ml":
            source_display = "Local ML"
        elif source_raw == "local_ml_fallback_aimlapi":
            source_display = "Local ML → AI/ML API fallback"
        elif source_raw == "local_ml_fallback_gemini":
            source_display = "Local ML → Gemini fallback"
        elif source_raw == "local_ml_fallback":
            source_display = "Local ML fallback"
        elif source_raw == "aimlapi":
            source_display = "AI/ML API (GLM 5 Turbo)"
        elif source_raw == "gemini":
            source_display = "Gemini"
        else:
            source_display = str(source_raw)

        ca = getattr(result, "confidence_assessment", None)
        level_display = ca.level.upper() if ca and ca.level else (result.reliability.upper() if result.reliability else "UNCERTAIN")
        range_display = ca.approximate_range if ca and ca.approximate_range else "N/A"

        # Two-column layout with ample space to prevent any text clipping or truncation
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Category**")
            st.markdown(f"### {result.category}")
            st.markdown("**Model Source**")
            st.markdown(f"{source_display}")

        with col_right:
            st.markdown("**Confidence Level**")
            st.markdown(f"### {level_display}")
            if ca and ca.is_calibrated:
                st.markdown("**Calibrated Confidence**")
            else:
                st.markdown("**Approximate Confidence**")
            st.markdown(f"{range_display}")

        # Diagnostic metrics: Raw Model Score, Calibrated Probability, Top-2 Margin
        metric_items = []
        if ca and ca.is_calibrated and ca.calibrated_prob is not None:
            metric_items.append(("Calibrated Prob", f"{ca.calibrated_prob:.4f}"))
        if ca and ca.raw_score is not None:
            metric_items.append(("Raw Model Score", f"{ca.raw_score:.4f}"))
        if ca and ca.is_calibrated and ca.calibration_method:
            metric_items.append(("Calibration Method", ca.calibration_method))
        if ca and ca.top2_margin is not None:
            metric_items.append(("Top-2 Margin", f"{ca.top2_margin:.4f}"))
        if amb:
            metric_items.append(("Ambiguity", "AMBIGUOUS" if amb.is_ambiguous else "CLEAR"))

        if metric_items:
            m_cols = st.columns(len(metric_items))
            for i, (m_label, m_val) in enumerate(metric_items):
                with m_cols[i]:
                    st.caption(m_label)
                    st.markdown(f"**{m_val}**")

        st.markdown(f"**Detected Language:** `{result.language}`")
        st.markdown("---")
        st.markdown("**Explanation**")
        st.write(result.reason)


def _render_multi_defect_result(result, md_res, ext_provider_label: str = "External LLM") -> None:
    """Renders the comprehensive multi-defect classification view and child breakdown."""
    defects = getattr(md_res, "defects", [])
    defect_count = getattr(md_res, "defect_count", len(defects))

    val_res = getattr(md_res, "validation_result", None) or getattr(result, "validation_result", None)
    val_status = getattr(val_res, "status", "VALID") if val_res else "VALID"

    with st.container(border=True):
        st.subheader("Multi-Defect Classification Result")

        # 1. Overall Summary Banner (Step 3)
        st.info(
            f"**Multi-Defect Detected:** This report contains **{defect_count} distinct defects** "
            "that have been segmented and classified independently."
        )

        col_count, col_val, col_mode = st.columns(3)
        with col_count:
            st.metric("Total Defects", defect_count)
        with col_val:
            st.metric("Validation Status", val_status)
        with col_mode:
            st.metric("Overall Outcome", getattr(md_res, "overall_status", "success").upper())

        # 2. Segment Visualization (Step 4)
        st.markdown("---")
        st.markdown("#### Input Text & Segment Breakdown")
        st.markdown(f"**Full Report:** *\"{result.original_description}\"*")

        for d in defects:
            d_idx = getattr(d, "defect_id", None) or getattr(d, "segment_id", 1)
            span_info = ""
            start_c = getattr(d, "source_start_char", None)
            end_c = getattr(d, "source_end_char", None)
            if start_c is not None and end_c is not None and (start_c > 0 or end_c > 0):
                span_info = f" *(chars {start_c}–{end_c})*"

            st.markdown(
                f"- **Defect {d_idx}**{span_info}: \"*{d.text}*\" → **{d.category}**"
            )

        st.markdown("---")
        st.markdown(f"#### Individual Defect Details ({defect_count})")

        # 3. Separate Result Expander / Card for EACH defect (Step 3 & Step 5)
        for idx, d in enumerate(defects, start=1):
            d_num = getattr(d, "defect_id", idx)
            d_cat = getattr(d, "category", "Unknown")
            d_text = getattr(d, "text", "")
            d_ca = getattr(d, "confidence_assessment", None)
            d_amb = getattr(d, "ambiguity_assessment", None)

            with st.expander(f"Defect #{d_num}: {d_cat} — \"{d_text[:50]}...\"", expanded=True):
                st.markdown("**Defect Text Segment:**")
                st.info(f"\"{d_text}\"")

                # Individual Unknown Warning (Step 5)
                if d_cat == "Unknown" or getattr(d, "status", "") == "unknown":
                    st.warning(
                        "**Outcome:** The segment does not contain sufficient technical evidence "
                        "to assign an authoritative defect category."
                    )

                # Individual Ambiguity Warning (Step 5)
                if d_amb and getattr(d_amb, "is_ambiguous", False):
                    amb_reason = getattr(d_amb, "reason", "Competing evidence between categories.")
                    st.warning(f"**Ambiguity Detected:** {amb_reason}")

                # Confidence and Category info (Step 3)
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("**Category**")
                    st.markdown(f"### {d_cat}")
                    st.markdown("**Reliability**")
                    st.markdown(f"{getattr(d, 'reliability', 'N/A')}")
                    mode_info = getattr(d, "classification_mode", "local")
                    prov_info = getattr(d, "provider", None) or "local"
                    model_info = getattr(d, "model", None) or "Local ML"
                    st.markdown("**Classification Mode / Provider**")
                    st.markdown(f"`{mode_info}` | `{prov_info}` (`{model_info}`)")

                with col_right:
                    level_disp = d_ca.level.upper() if d_ca and d_ca.level else (getattr(d, "reliability", "UNCERTAIN").upper())
                    range_disp = d_ca.approximate_range if d_ca and d_ca.approximate_range else "N/A"
                    st.markdown("**Confidence Level**")
                    st.markdown(f"### {level_disp}")
                    if d_ca and getattr(d_ca, "is_calibrated", False):
                        st.markdown("**Calibrated Confidence**")
                    else:
                        st.markdown("**Approximate Confidence**")
                    st.markdown(f"{range_disp}")

                # Diagnostic Metrics: Calibrated probability, raw score, top-2 margin (Step 3)
                metric_items = []
                cal_p = getattr(d, "calibrated_prob", None)
                if cal_p is None and d_ca:
                    cal_p = getattr(d_ca, "calibrated_prob", None)
                if cal_p is not None:
                    metric_items.append(("Calibrated Prob", f"{float(cal_p):.4f}"))

                raw_s = getattr(d, "raw_score", None)
                if raw_s is None and d_ca:
                    raw_s = getattr(d_ca, "raw_score", None)
                if raw_s is None and getattr(d, "confidence", None) is not None:
                    raw_s = getattr(d, "confidence", None)
                if raw_s is not None:
                    metric_items.append(("Raw Model Score", f"{float(raw_s):.4f}"))

                top2_m = getattr(d, "top2_margin", None)
                if top2_m is None and d_ca:
                    top2_m = getattr(d_ca, "top2_margin", None)
                if top2_m is not None:
                    metric_items.append(("Top-2 Margin", f"{float(top2_m):.4f}"))

                if d_amb:
                    metric_items.append(("Ambiguity", "AMBIGUOUS" if getattr(d_amb, "is_ambiguous", False) else "CLEAR"))

                if metric_items:
                    m_cols = st.columns(len(metric_items))
                    for i, (m_label, m_val) in enumerate(metric_items):
                        with m_cols[i]:
                            st.caption(m_label)
                            st.markdown(f"**{m_val}**")

                st.markdown("---")
                st.markdown("**Explanation**")
                st.write(getattr(d, "explanation", "No explanation provided."))
