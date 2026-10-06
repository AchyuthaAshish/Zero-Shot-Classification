"""Streamlit Application Entrypoint for Zero-Shot Industrial Defect Classifier.

Conforms to MVP Frontend Requirements:
1. Launches the Streamlit application.
2. Delegates UI rendering to ui.classifier_view.
3. Connects solely to classification.classifier.classify_defect.
"""

import streamlit as st
from ui.classifier_view import render_classifier_view
from ui.history_view import render_history_view
from ui.evaluation_view import render_evaluation_dashboard


def main():
    st.set_page_config(
        page_title="Industrial Defect Classifier",
        page_icon="🏭",
        layout="centered"
    )

    # Sidebar Navigation
    st.sidebar.title("🏭 Defect System")
    nav_selection = st.sidebar.radio(
        "Navigation",
        options=["Classify Defect", "Report History", "AI Evaluation Dashboard"],
        index=0
    )

    if nav_selection == "Classify Defect":
        render_classifier_view()
    elif nav_selection == "Report History":
        render_history_view()
    else:
        render_evaluation_dashboard()


if __name__ == "__main__":
    main()
