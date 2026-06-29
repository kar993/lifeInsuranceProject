"""Streamlit UI for the Life Insurance Risk Analysis & Advisory Agent."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")
load_dotenv()

from agents.workflow_agent import run_structured_analysis
from app.adk_runner import AdvisoryAgentRunner
from config.paths import REPORT_DIR
from services.pdf_report_generator import PDFReportGenerator
from domain.recommended_product import RecommendedProduct
from domain.report_data import ReportData
from state.state_keys import PDF_PATH, RECOMMENDATION_RESULT, REPORT_TEXT

st.set_page_config(
    page_title="Life Insurance Advisory Agent",
    page_icon="🛡️",
    layout="wide",
)

st.title("Life Insurance Risk Analysis & Advisory Agent")
st.caption(
    "Agentic AI advisory system with ADK multi-agent workflow, session state, "
    "graph-based analysis, and explainable recommendations."
)


def _format_currency(value: float) -> str:
    return f"₹{value:,.0f}"


def _ensure_api_key_warning() -> bool:
    if os.getenv("GOOGLE_API_KEY"):
        return True
    st.warning(
        "Set `GOOGLE_API_KEY` in `backend/.env` to use the conversational AI chat tab. "
        "The structured analysis tab works without an API key."
    )
    return False


def _load_occupation_options() -> list[str]:
    occupation_file = BACKEND_DIR / "data" / "01_occupation_risk_table.csv"
    with occupation_file.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["occupation"] for row in reader]


def _default_profile() -> dict:
    return {
        "age": 35,
        "gender": "Male",
        "annual_income": 1_200_000,
        "marital_status": "Married",
        "dependents": 2,
        "existing_life_coverage": 500_000,
        "outstanding_loans": 2_500_000,
        "height_cm": 175,
        "weight_kg": 78,
        "occupation": "Software Engineer",
        "smoker_status": False,
        "alcohol_use": "Moderate",
        "exercise_level": "Medium",
        "diabetes": False,
        "hypertension": False,
        "heart_disease": False,
        "respiratory_disease": False,
        "cancer_history": False,
        "desired_coverage": 15_000_000,
        "payment_mode_preference": "Annual",
        "autopay_preference": True,
    }


def _render_recommendation_details(state: dict) -> None:
    recommendation = state.get(RECOMMENDATION_RESULT)
    if not recommendation:
        st.info("Run an analysis to see recommendation details.")
        return

    col1, col2, col3 = st.columns(3)
    mortality = recommendation.get("mortality_prediction", {})
    coverage = recommendation.get("coverage_recommendation", {})

    col1.metric("Mortality Category", mortality.get("mortality_category", "N/A"))
    col2.metric(
        "Recommended Coverage",
        _format_currency(coverage.get("recommended_coverage", 0)),
    )
    col3.metric(
        "Mortality Score",
        f"{mortality.get('mortality_score', 0):.1f}",
    )

    products = recommendation.get("recommended_products", [])
    if products:
        st.subheader("Ranked Product Recommendations")
        rows = []
        for product in products[:5]:
            rows.append(
                {
                    "Product": product.get("product_name"),
                    "Annual Premium": _format_currency(product.get("annual_premium", 0)),
                    "Lapse Category": product.get("lapse_category"),
                    "Suitability Score": product.get("suitability_score"),
                }
            )
        st.dataframe(rows, use_container_width=True)


def _generate_pdf_from_state(state: dict) -> Path | None:
    report_data_dict = state.get("report_data")
    recommendation = state.get(RECOMMENDATION_RESULT)
    if not report_data_dict or not recommendation:
        return None

    recommended_products = [
        RecommendedProduct(**product)
        for product in recommendation.get("recommended_products", [])
    ]

    report_data = ReportData(
        customer_name=report_data_dict.get("customer_name", "Customer"),
        eligibility_status=report_data_dict.get("eligibility_status", "UNKNOWN"),
        mortality_probability=report_data_dict.get("mortality_probability", 0.0),
        mortality_category=report_data_dict.get("mortality_category", "Unknown"),
        total_protection_need=report_data_dict.get("total_protection_need", 0.0),
        recommended_coverage=report_data_dict.get("recommended_coverage", 0.0),
        stretch_coverage=report_data_dict.get("stretch_coverage", 0.0),
        recommended_products=recommended_products,
        advisory_summary=state.get(REPORT_TEXT, ""),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORT_DIR / "streamlit_advisory_report.pdf"
    PDFReportGenerator().generate(report_data, str(pdf_path))
    return pdf_path


tab_chat, tab_form, tab_state = st.tabs(
    ["AI Advisor Chat", "Structured Analysis", "Session State"]
)

with tab_form:
    st.subheader("Structured Profile Analysis")
    st.write(
        "Runs the deterministic ADK graph workflow and ML pipeline without requiring "
        "a Gemini API call."
    )

    profile = _default_profile()
    with st.form("customer_profile_form"):
        c1, c2, c3 = st.columns(3)
        profile["age"] = c1.number_input("Age", 18, 85, profile["age"])
        profile["gender"] = c2.selectbox("Gender", ["Male", "Female"], index=0)
        profile["annual_income"] = c3.number_input(
            "Annual Income (₹)",
            min_value=100_000,
            value=profile["annual_income"],
            step=50_000,
        )

        c4, c5, c6 = st.columns(3)
        profile["marital_status"] = c4.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced", "Widowed"],
            index=1,
        )
        profile["dependents"] = c5.number_input("Dependents", 0, 10, profile["dependents"])
        occupation_options = _load_occupation_options()
        occupation_index = (
            occupation_options.index(profile["occupation"])
            if profile["occupation"] in occupation_options
            else 0
        )
        profile["occupation"] = c6.selectbox(
            "Occupation",
            occupation_options,
            index=occupation_index,
        )

        c7, c8, c9 = st.columns(3)
        profile["height_cm"] = c7.number_input(
            "Height (cm)", min_value=120.0, max_value=220.0, value=float(profile["height_cm"])
        )
        profile["weight_kg"] = c8.number_input(
            "Weight (kg)", min_value=35.0, max_value=180.0, value=float(profile["weight_kg"])
        )
        profile["desired_coverage"] = c9.number_input(
            "Desired Coverage (₹)",
            min_value=100_000,
            value=profile["desired_coverage"],
            step=100_000,
        )

        c10, c11 = st.columns(2)
        profile["existing_life_coverage"] = c10.number_input(
            "Existing Life Coverage (₹)",
            min_value=0,
            value=profile["existing_life_coverage"],
            step=100_000,
        )
        profile["outstanding_loans"] = c11.number_input(
            "Outstanding Loans (₹)",
            min_value=0,
            value=profile["outstanding_loans"],
            step=100_000,
        )

        c12, c13, c14 = st.columns(3)
        profile["alcohol_use"] = c12.selectbox(
            "Alcohol Use",
            ["No Alcohol", "Occasional", "Moderate", "Heavy"],
            index=2,
        )
        profile["exercise_level"] = c13.selectbox(
            "Exercise Level",
            ["Low", "Medium", "High"],
            index=1,
        )
        profile["payment_mode_preference"] = c14.selectbox(
            "Payment Mode",
            ["Annual", "Semi-Annual", "Quarterly", "Monthly"],
            index=0,
        )

        c15, c16, c17, c18, c19 = st.columns(5)
        profile["smoker_status"] = c15.checkbox("Smoker", value=profile["smoker_status"])
        profile["diabetes"] = c16.checkbox("Diabetes", value=profile["diabetes"])
        profile["hypertension"] = c17.checkbox("Hypertension", value=profile["hypertension"])
        profile["heart_disease"] = c18.checkbox("Heart Disease", value=profile["heart_disease"])
        profile["respiratory_disease"] = c19.checkbox(
            "Respiratory Disease",
            value=profile["respiratory_disease"],
        )
        profile["cancer_history"] = st.checkbox(
            "Cancer History",
            value=profile["cancer_history"],
        )
        profile["autopay_preference"] = st.checkbox(
            "Autopay Preference",
            value=profile["autopay_preference"],
        )

        submitted = st.form_submit_button("Run Recommendation Analysis", type="primary")

    if submitted:
        with st.spinner("Running eligibility, mortality, coverage, premium, and persistency models..."):
            try:
                result = run_structured_analysis(profile)
                st.session_state["analysis_state"] = result["session_state"]
                st.session_state["analysis_message"] = result["message"]
                st.success("Analysis completed.")
                st.markdown(result["message"])
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    if "analysis_state" in st.session_state:
        _render_recommendation_details(st.session_state["analysis_state"])

        if st.button("Generate PDF Report"):
            pdf_path = _generate_pdf_from_state(st.session_state["analysis_state"])
            if pdf_path and pdf_path.exists():
                st.session_state["analysis_state"][PDF_PATH] = str(pdf_path)
                st.success(f"PDF generated: {pdf_path.name}")
                st.download_button(
                    "Download Advisory PDF",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                )

with tab_chat:
    st.subheader("Conversational AI Advisor")
    _ensure_api_key_warning()

    if "chat_runner" not in st.session_state:
        st.session_state.chat_runner = AdvisoryAgentRunner()
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = AdvisoryAgentRunner.new_session_id()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about life insurance coverage, products, or share your profile...")
    if prompt and os.getenv("GOOGLE_API_KEY"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Advisor is thinking..."):
                try:
                    response = st.session_state.chat_runner.send_message_sync(
                        message=prompt,
                        session_id=st.session_state.chat_session_id,
                    )
                    st.markdown(response.text)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": response.text}
                    )
                    st.session_state["chat_state"] = response.session_state
                except Exception as exc:
                    if "Session not found" in str(exc):
                        try:
                            # Session was lost (e.g. server restarted). Create a new one and retry.
                            st.session_state.chat_session_id = AdvisoryAgentRunner.new_session_id()
                            response = st.session_state.chat_runner.send_message_sync(
                                message=prompt,
                                session_id=st.session_state.chat_session_id,
                            )
                            st.markdown(response.text)
                            st.session_state.chat_messages.append(
                                {"role": "assistant", "content": response.text}
                            )
                            st.session_state["chat_state"] = response.session_state
                        except Exception as retry_exc:
                            st.error(f"Agent error after session reset: {retry_exc}")
                    elif any(
                        marker in str(exc).upper()
                        for marker in ("503", "UNAVAILABLE", "HIGH DEMAND")
                    ):
                        st.error(
                            "The Gemini API is temporarily overloaded. "
                            "Please wait a minute and try again. "
                            "You can also set `GEMINI_MODEL=gemini-2.5-pro` in `backend/.env`."
                        )
                    else:
                        st.error(f"Agent error: {exc}")


    chat_col1, chat_col2 = st.columns(2)
    if chat_col1.button("Start New Chat Session"):
        st.session_state.chat_session_id = AdvisoryAgentRunner.new_session_id()
        st.session_state.chat_messages = []
        st.session_state.pop("chat_state", None)
        st.rerun()

    if chat_col2.button("Show Chat Session State") and st.session_state.get("chat_state"):
        st.json(st.session_state["chat_state"])

with tab_state:
    st.subheader("ADK Session State Inspector")
    state = st.session_state.get("analysis_state") or st.session_state.get("chat_state")
    if state:
        st.json(state)
        _render_recommendation_details(state)
        pdf_path = state.get(PDF_PATH)
        if pdf_path and Path(pdf_path).exists():
            st.download_button(
                "Download Latest PDF",
                data=Path(pdf_path).read_bytes(),
                file_name=Path(pdf_path).name,
                mime="application/pdf",
            )
    else:
        st.info("Run structured analysis or chat with the advisor to populate session state.")

with st.sidebar:
    st.header("Project Status")
    models_dir = BACKEND_DIR / "models"
    mortality_model = models_dir / "mortality_prediction_model_v1.joblib"
    persistency_model = models_dir / "persistency_prediction_model_v1.joblib"
    st.write("Mortality model:", "✅" if mortality_model.exists() else "❌")
    st.write("Persistency model:", "✅" if persistency_model.exists() else "❌")
    has_api_key = bool(os.getenv("GOOGLE_API_KEY"))
    st.write("Gemini API key:", "✅" if has_api_key else "❌")

    if not has_api_key:
        st.subheader("🔑 Set Gemini API Key")
        st.markdown(
            "Get a free API key from [Google AI Studio](https://aistudio.google.com/)."
        )
        api_key_input = st.text_input("Enter Gemini API Key", type="password", key="gemini_api_key_input")
        if st.button("Save API Key", type="primary"):
            if api_key_input:
                env_path = BACKEND_DIR / ".env"
                try:
                    env_path.write_text(f'GOOGLE_API_KEY="{api_key_input}"\n', encoding="utf-8")
                    os.environ["GOOGLE_API_KEY"] = api_key_input
                    os.environ["GEMINI_API_KEY"] = api_key_input
                    st.success("API key saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving API key: {e}")
            else:
                st.warning("Please enter a valid API key.")


    st.divider()
    st.markdown(
        """
**Architecture**
- Coordinator agent routes intake, recommendation, and report specialists
- Graph workflow runs deterministic ML pipeline
- ADK session state stores recommendation artifacts
- Memory service stores completed advisory sessions
        """
    )

    if not mortality_model.exists() or not persistency_model.exists():
        st.code("cd backend\npython scripts/train_models.py", language="bash")

    st.code("streamlit run streamlit_app.py", language="bash")
    st.code("cd backend\nadk web --port 8000", language="bash")
