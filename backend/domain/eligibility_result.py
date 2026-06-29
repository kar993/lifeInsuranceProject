from dataclasses import dataclass

@dataclass
class EligibilityResult:

    decision: str      # ELIGIBLE / REFER / DECLINE

    triggered_rules: list[str]

    explanations: list[str]