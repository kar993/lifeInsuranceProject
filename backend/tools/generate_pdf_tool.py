from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from google.adk.tools import ToolContext

from domain.report_data import ReportData
from domain.recommended_product import RecommendedProduct
from services.pdf_report_generator import PDFReportGenerator
from state.state_keys import PDF_PATH, REPORT_DATA, REPORT_TEXT
from config.paths import REPORT_DIR


pdf_generator = PDFReportGenerator()


def generate_pdf_tool(
    report_text: str,
    tool_context: ToolContext,
) -> dict:
    """Generate a PDF advisory report from session state and narrative text."""

    report_data_dict = tool_context.state.get(REPORT_DATA)
    if not report_data_dict:
        return {
            "status": "error",
            "message": "No recommendation data found. Run generate_recommendation_tool first.",
        }

    recommended_products = [
        RecommendedProduct(**product)
        for product in report_data_dict.get("recommended_products", [])
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
        advisory_summary=report_text or report_data_dict.get("advisory_summary", ""),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORT_DIR / f"advisory_report_{timestamp}.pdf"
    pdf_generator.generate(report_data, str(pdf_path))

    tool_context.state[PDF_PATH] = str(pdf_path)
    tool_context.state[REPORT_TEXT] = report_text

    return {
        "status": "success",
        "pdf_path": str(pdf_path),
        "report_data": asdict(report_data),
    }
