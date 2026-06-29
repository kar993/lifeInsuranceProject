from google.adk.sessions import Session

from state.state_keys import (
    CUSTOMER_PROFILE,
    RECOMMENDATION_RESULT,
    REPORT_DATA,
    REPORT_TEXT,
    PDF_PATH,
    RECOMMENDATION_GENERATED,
)


class StateManager:
    """
    Helper methods for reading/writing
    ADK session state.
    """

    @staticmethod
    def set_customer_profile(session: Session, profile: dict):
        session.state[CUSTOMER_PROFILE] = profile

    @staticmethod
    def get_customer_profile(session: Session):
        return session.state.get(CUSTOMER_PROFILE)

    @staticmethod
    def set_recommendation_result(
        session: Session,
        recommendation_result: dict,
    ):
        session.state[RECOMMENDATION_RESULT] = recommendation_result
        session.state[RECOMMENDATION_GENERATED] = True

    @staticmethod
    def get_recommendation_result(session: Session):
        return session.state.get(RECOMMENDATION_RESULT)

    @staticmethod
    def set_report_data(
        session: Session,
        report_data: dict,
    ):
        session.state[REPORT_DATA] = report_data

    @staticmethod
    def get_report_data(session: Session):
        return session.state.get(REPORT_DATA)

    @staticmethod
    def set_report_text(
        session: Session,
        report_text: str,
    ):
        session.state[REPORT_TEXT] = report_text

    @staticmethod
    def get_report_text(session: Session):
        return session.state.get(REPORT_TEXT)

    @staticmethod
    def set_pdf_path(
        session: Session,
        pdf_path: str,
    ):
        session.state[PDF_PATH] = pdf_path

    @staticmethod
    def get_pdf_path(session: Session):
        return session.state.get(PDF_PATH)

    @staticmethod
    def recommendation_exists(session: Session):
        return session.state.get(
            RECOMMENDATION_GENERATED,
            False,
        )