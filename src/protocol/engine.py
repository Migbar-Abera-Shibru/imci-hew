"""
IMCI Young Infant Protocol Engine

Deterministic traversal of the IMCI decision tree.
No ML. No LLM. Pure logic.

Input:  a set of observed signs (keys from sign_schema.json)
Output: a structured clinical decision with classifications + actions
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field




# data structure
@dataclass
class Classification:
    name: str
    priority: int
    severity: str
    matched_signs: list[str]
    actions: dict[str, Any]
    is_default: bool = False


@dataclass
class ClinicalDecision:
    input_signs: list[str]
    classifications: list[Classification] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # nodes with no classification

    def to_dict(self) -> dict:
        return{
            "input_signs": self.input_signs,
            "classifcations": [asdict(c) for c in self.Classifications],
            "unresolved_nodes": self.unresolved,
        }

    def highest_severity(self) -> str:
        """ return the highest severity among all classification."""
        order = {"critical": 0, "severe": 1, "moderate": 2, "low": 3, "none": 4}
        if not self.classifications:
            return "none"
        return min(
            (c.severity for c in self.classifications),
            key=lambda s: order.get(s, 99)
        )