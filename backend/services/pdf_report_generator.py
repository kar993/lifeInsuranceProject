# services/pdf_report_generator.py

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from domain.report_data import ReportData


class PDFReportGenerator:

    def generate(
        self,
        report_data: ReportData,
        output_path: str,
    ) -> None:

        doc = SimpleDocTemplate(output_path)

        styles = getSampleStyleSheet()

        elements = []

        # ==================================================
        # TITLE
        # ==================================================

        elements.append(
            Paragraph(
                "Life Insurance Advisory Report",
                styles["Title"],
            )
        )

        elements.append(Spacer(1,20,))

        elements.append(
            Paragraph(
                f"Customer: {report_data.customer_name}",
                styles["Normal"],
            )
        )

        elements.append(Spacer(1,20,))

        # ==================================================
        # CUSTOMER / ELIGIBILITY SUMMARY
        # ==================================================

        elements.append(
            Paragraph(
                "Executive Summary",
                styles["Heading1"],
            )
        )

        summary_table = Table(
            [
                [
                    "Eligibility Status",
                    report_data.eligibility_status,
                ],
                [
                    "Mortality Category",
                    report_data.mortality_category,
                ],
                [
                    "Mortality Probability",
                    (
                        f"{report_data.mortality_probability:.2%}"
                    ),
                ],
                [
                    "Recommended Coverage",
                    (
                        f"₹{report_data.recommended_coverage:,.0f}"
                    ),
                ],
            ]
        )

        summary_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        1,
                        colors.black,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.lightgrey,
                    ),
                ]
            )
        )

        elements.append(summary_table)

        elements.append(Spacer(1,20,))

        # ==================================================
        # COVERAGE ANALYSIS
        # ==================================================

        elements.append(
            Paragraph(
                "Coverage Analysis",
                styles["Heading1"],
            )
        )

        coverage_table = Table(
            [
                ["Total Protection Need",(f"₹{report_data.total_protection_need:,.0f}"),],

                ["Recommended Coverage",(f"₹{report_data.recommended_coverage:,.0f}"),],

                ["Stretch Coverage",(f"₹{report_data.stretch_coverage:,.0f}"),],
            ]
        )

        coverage_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        1,
                        colors.black,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.lightgrey,
                    ),
                ]
            )
        )

        elements.append(coverage_table)

        elements.append(Spacer(1,20,))

        # ==================================================
        # PRODUCT RECOMMENDATIONS
        # ==================================================

        elements.append(
            Paragraph(
                "Recommended Products",
                styles["Heading1"],
            )
        )

        product_rows = [
            [
                "Product",
                "Type",
                "Score",
                "Annual Premium",
                "Lapse Risk",
            ]
        ]

        for product in (report_data.recommended_products):
            product_rows.append(
                [
                    product.product_name,
                    product.product_type,
                    f"{product.suitability_score:.1f}",
                    f"INR{product.annual_premium:,.0f}",
                    (
                        f"{product.lapse_category}"
                        f" ({product.lapse_probability:.1%})"
                    ),
                ]
            )

        product_table = Table(
            product_rows,
            colWidths=[180,80,60,100,100],
        )

        product_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        1,
                        colors.black,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                ]
            )
        )

        elements.append(product_table)

        elements.append(Spacer(1,20,))

        elements.append(
            Paragraph(
                "Recommended Product Details",
                styles['Heading1']
            )
        )

        for product in report_data.recommended_products:
            elements.append(
                Paragraph(
                    f"<b>{product.product_name}</b>",
                    styles["Heading2"],
                )
            )
            elements.append(
                Paragraph(
                    (
                        f"Suitability Score: "
                        f"{product.suitability_score:.1f}"
                    ),
                    styles["BodyText"],
                )
            )
            elements.append(
                Paragraph(
                    (
                        f"Annual Premium: "
                        f"₹{product.annual_premium:,.0f}"
                    ),
                    styles["BodyText"],
                )
            )
            elements.append(
                Paragraph(
                    (
                        f"Expected Lapse Risk: "
                        f"{product.lapse_category}"
                    ),
                    styles["BodyText"],
                )
            )
            for reason in product.rationale:

                elements.append(
                    Paragraph(
                        f"• {reason}",
                        styles["BodyText"],
                    )
                )
            elements.append(
                Spacer(1, 10)
            )
        # ==================================================
        # ADVISORY SUMMARY
        # ==================================================

        elements.append(
            Paragraph(
                "Advisory Summary",
                styles["Heading1"],
            )
        )

        elements.append(
            Paragraph(
                report_data.advisory_summary,
                styles["BodyText"],
            )
        )

        elements.append(Spacer(1,20,))

        # ==================================================
        # DISCLAIMER
        # ==================================================

        elements.append(
            Paragraph(
                "Important Notes",
                styles["Heading1"],
            )
        )

        disclaimer = """
        This report is advisory in nature and has been generated
        using internal business rules and predictive models.

        Final underwriting decisions remain subject to insurer
        review and approval.

        Premiums shown are indicative estimates and may change
        following underwriting assessment.
        """

        elements.append(
            Paragraph(
                disclaimer,
                styles["BodyText"],
            )
        )

        # ==================================================
        # BUILD PDF
        # ==================================================

        doc.build(elements)