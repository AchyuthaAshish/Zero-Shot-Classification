"""History View for Stored Defect Reports.

Displays persistent defect reports stored in Supabase with filtering, sorting,
and interactive child defect inspection for multi-defect records.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import streamlit as st
import pandas as pd

from persistence.repository import (
    get_defect_reports,
    get_defect_report_items,
    is_supabase_configured,
    DatabaseError
)
from taxonomy.repository import get_taxonomy_repository


def render_history_view() -> None:
    """Renders the historical records view connected to Supabase."""
    st.title("Defect Report History")
    st.markdown(
        "Persistent audit log of industrial defect reports and classification outcomes "
        "stored securely in Supabase."
    )

    if not is_supabase_configured():
        st.warning(
            "**Supabase Storage Offline:** `SUPABASE_URL` and `SUPABASE_KEY` are not configured in your `.env` file.\n\n"
            "To view and persist historical records, please set your Supabase credentials in `.env`:\n"
            "```text\n"
            "SUPABASE_URL=https://your-project.supabase.co\n"
            "SUPABASE_KEY=your-anon-publishable-key\n"
            "```"
        )
        return

    # Filter Section
    st.subheader("Filter Reports")
    tax_repo = get_taxonomy_repository()
    categories = ["All"] + tax_repo.get_categories()

    col_cat, col_emp, col_btn = st.columns([2, 2, 1])
    with col_cat:
        selected_cat = st.selectbox("Filter by Category", options=categories, index=0)
    with col_emp:
        emp_filter = st.text_input("Filter by Employee ID", placeholder="e.g. EMP-1042")
    with col_btn:
        st.write("")
        st.write("")
        refresh_clicked = st.button("Refresh", use_container_width=True)

    # Fetch Reports from Supabase via Repository Layer (Step 6)
    with st.spinner("Fetching reports from Supabase..."):
        try:
            reports = get_defect_reports(
                limit=100,
                category=selected_cat if selected_cat != "All" else None,
                employee_id=emp_filter if emp_filter.strip() else None
            )
        except DatabaseError as e:
            st.error(f"Failed to query reports: {e.message}")
            return
        except Exception:
            st.error("An unexpected database error occurred while querying records.")
            return

    if not reports:
        st.info("No defect reports found matching the specified filters.")
        return

    st.caption(f"Displaying **{len(reports)}** most recent reports (sorted newest first):")

    # Format records for overview display (Step 6 & Step 8)
    display_rows = []
    for r in reports:
        raw_ts = r.get("created_at")
        formatted_ts = "N/A"
        if raw_ts:
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                formatted_ts = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                formatted_ts = str(raw_ts)[:19]

        conf_lvl = r.get("confidence_level") or r.get("reliability") or "N/A"
        conf_rng = r.get("confidence_range") or ("Qualitative / N/A" if not r.get("confidence") else "Legacy uncalibrated")
        model_src = r.get("provider") or r.get("classification_mode") or "N/A"
        raw_sc = r.get("raw_score")
        if raw_sc is None and r.get("confidence") is not None:
            raw_sc = r.get("confidence")
        raw_sc_str = f"{float(raw_sc):.4f}" if raw_sc is not None else "N/A"

        is_multi = bool(r.get("is_multi_defect"))
        d_cnt = r.get("defect_count")
        if d_cnt is None:
            d_cnt = 2 if is_multi else 1
        rep_type = f"Multi-Defect • {d_cnt} defects" if is_multi else "Single Defect"
        val_stat = r.get("validation_status") or ("VALID" if not is_multi else "N/A")

        display_rows.append({
            "Date/Time": formatted_ts,
            "Employee ID": r.get("employee_id", "N/A"),
            "Reporter": r.get("reporter_name", "N/A"),
            "Report Type": rep_type,
            "Category": r.get("category", "N/A"),
            "Validation Status": val_stat,
            "Confidence Level": conf_lvl,
            "Confidence Range": conf_rng,
            "Raw Model Score": raw_sc_str,
            "Model Source": model_src,
            "Mode": r.get("classification_mode", "N/A"),
            "Defect Description": r.get("defect_description", "N/A"),
            "Explanation": r.get("explanation", "N/A")
        })

    df = pd.DataFrame(display_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detailed Report Inspection / Child Items Section (Step 6 & Step 8)
    st.markdown("---")
    st.subheader("Report Inspection & Defect Breakdown")

    report_map = {}
    options = []
    for idx, r in enumerate(reports, start=1):
        ts_short = (r.get("created_at") or "")[:19].replace("T", " ")
        emp = r.get("employee_id") or "Unknown"
        is_m = bool(r.get("is_multi_defect"))
        cnt = r.get("defect_count", 2)
        desc_preview = (r.get("defect_description") or "")[:45]
        type_badge = f"[MULTI • {cnt} defects]" if is_m else f"[{r.get('category', 'Single')}]"
        label = f"#{idx} | {ts_short} | {emp} | {type_badge} - {desc_preview}..."
        options.append(label)
        report_map[label] = r

    selected_label = st.selectbox(
        "Select a Report to Inspect Details & Child Items",
        options=options,
        index=0,
        help="Select any report from the list above to view full details and child defect segments."
    )

    if selected_label and selected_label in report_map:
        selected_report = report_map[selected_label]
        _render_history_detail(selected_report)


def _render_history_detail(r: Dict[str, Any]) -> None:
    """Renders the detailed view of a selected history report, retrieving child defect items if multi-defect."""
    is_multi = bool(r.get("is_multi_defect"))
    defect_count = r.get("defect_count") or (2 if is_multi else 1)
    val_status = r.get("validation_status") or "N/A"
    report_id = r.get("id")

    with st.container(border=True):
        st.markdown("### Report Details")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption("Reporter")
            st.markdown(f"**{r.get('reporter_name', 'N/A')}**")
        with col2:
            st.caption("Employee ID")
            st.markdown(f"**{r.get('employee_id', 'N/A')}**")
        with col3:
            st.caption("Report Type")
            st.markdown(f"**{'Multi-Defect' if is_multi else 'Single Defect'}**")
        with col4:
            st.caption("Validation Status")
            st.markdown(f"**{val_status}**")

        st.markdown("**Original Defect Description:**")
        st.info(f"\"{r.get('defect_description', 'N/A')}\"")

        if is_multi:
            st.markdown(f"#### Child Defect Items ({defect_count} detected)")

            child_items = []
            if report_id:
                try:
                    child_items = get_defect_report_items(report_id)
                except DatabaseError as e:
                    st.warning(f"Could not retrieve child defect items from database: {e.message}")
                except Exception:
                    st.warning("Could not retrieve child defect items from database.")

            if child_items:
                for item in child_items:
                    d_idx = item.get("defect_index") or item.get("defect_id") or 1
                    cat = item.get("category", "Unknown")
                    d_text = item.get("defect_text", "N/A")
                    conf_lvl = item.get("confidence_level") or item.get("reliability") or "N/A"
                    conf_rng = item.get("confidence_range") or "N/A"
                    raw_sc = item.get("raw_score")
                    raw_sc_str = f"{float(raw_sc):.4f}" if raw_sc is not None else "N/A"
                    cal_sc = item.get("calibrated_score")
                    cal_sc_str = f"{float(cal_sc):.4f}" if cal_sc is not None else "N/A"
                    margin = item.get("top2_margin")
                    margin_str = f"{float(margin):.4f}" if margin is not None else "N/A"
                    is_amb = bool(item.get("is_ambiguous"))
                    amb_reason = item.get("ambiguity_reason")

                    with st.expander(f"Defect #{d_idx}: {cat} — \"{d_text[:50]}...\"", expanded=True):
                        st.markdown("**Defect Text Segment:**")
                        st.info(f"\"{d_text}\"")

                        if cat == "Unknown":
                            st.warning("**Outcome:** Insufficient technical evidence to assign an authoritative defect category.")

                        if is_amb:
                            st.warning(f"**Ambiguity:** {amb_reason or 'Competing evidence between categories.'}")

                        c_left, c_right = st.columns(2)
                        with c_left:
                            st.markdown("**Category**")
                            st.markdown(f"### {cat}")
                            st.markdown("**Reliability**")
                            st.markdown(f"{item.get('reliability', 'N/A')}")
                            st.markdown("**Source**")
                            st.markdown(f"`{item.get('classification_mode', 'N/A')}` | `{item.get('provider', 'N/A')}` (`{item.get('model', 'N/A')}`)")
                        with c_right:
                            st.markdown("**Confidence Level**")
                            st.markdown(f"### {conf_lvl}")
                            st.markdown("**Approximate Confidence**")
                            st.markdown(f"{conf_rng}")

                        metric_entries = []
                        if cal_sc is not None:
                            metric_entries.append(("Calibrated Prob", cal_sc_str))
                        if raw_sc is not None:
                            metric_entries.append(("Raw Score", raw_sc_str))
                        if margin is not None:
                            metric_entries.append(("Top-2 Margin", margin_str))
                        metric_entries.append(("Ambiguity", "AMBIGUOUS" if is_amb else "CLEAR"))

                        if metric_entries:
                            m_cols = st.columns(len(metric_entries))
                            for i, (m_label, m_val) in enumerate(metric_entries):
                                with m_cols[i]:
                                    st.caption(m_label)
                                    st.markdown(f"**{m_val}**")

                        st.markdown("---")
                        st.markdown("**Explanation**")
                        st.write(item.get("explanation", "No explanation available."))
            else:
                st.info("No child defect items found for this record in defect_report_items (legacy record or child items not present).")
                st.markdown(f"**Primary Category:** `{r.get('category', 'N/A')}`")
                st.markdown(f"**Confidence Level:** `{r.get('confidence_level') or r.get('reliability') or 'N/A'}`")
                st.markdown(f"**Explanation:** {r.get('explanation', 'N/A')}")
        else:
            # Single-defect detailed view
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("**Category**")
                st.markdown(f"### {r.get('category', 'N/A')}")
                st.markdown("**Reliability**")
                st.markdown(f"{r.get('reliability', 'N/A')}")
                st.markdown("**Source / Mode**")
                st.markdown(f"`{r.get('provider') or r.get('classification_mode') or 'N/A'}` (`{r.get('model', 'N/A')}`)")
            with c_right:
                conf_lvl = r.get("confidence_level") or r.get("reliability") or "N/A"
                conf_rng = r.get("confidence_range") or ("Qualitative / N/A" if not r.get("confidence") else "Legacy uncalibrated")
                st.markdown("**Confidence Level**")
                st.markdown(f"### {conf_lvl}")
                st.markdown("**Approximate Confidence**")
                st.markdown(f"{conf_rng}")

            raw_sc = r.get("raw_score")
            if raw_sc is None and r.get("confidence") is not None:
                raw_sc = r.get("confidence")
            raw_sc_str = f"{float(raw_sc):.4f}" if raw_sc is not None else "N/A"
            cal_sc = r.get("calibrated_score")
            cal_sc_str = f"{float(cal_sc):.4f}" if cal_sc is not None else "N/A"

            metric_entries = []
            if cal_sc is not None:
                metric_entries.append(("Calibrated Prob", cal_sc_str))
            if raw_sc is not None:
                metric_entries.append(("Raw Model Score", raw_sc_str))
            metric_entries.append(("Mode", r.get("classification_mode", "N/A")))

            m_cols = st.columns(len(metric_entries))
            for i, (m_label, m_val) in enumerate(metric_entries):
                with m_cols[i]:
                    st.caption(m_label)
                    st.markdown(f"**{m_val}**")

            st.markdown("---")
            st.markdown("**Explanation**")
            st.write(r.get("explanation", "N/A"))
