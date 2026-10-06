"""
IMCI Young Infant Protocol Engine

Deterministic traversal of the IMCI decision tree.
No ML. No LLM. Pure logic.

Input:  a set of observed signs (keys from sign_schema.json)
Output: a structured clinical decision with classifications + actions
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any




# data structure
@dataclass
class Classification:
    name: str
    node_id: str
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


# engine 

class IMCIEngine:
    def __init__(self, protocol_path: str | Path):
        with open(protocol_path, "r", encoding="utf-8") as f:
            self.protocol = json.load(f)
        self.nodes = self.protocol["decision_nodes"]

        # public api
    def evaluate(self, observed_signs: list[str]) -> ClinicalDecision:
        """
        Run the full initial-visit assessment against the observed signs.

        observed_signs: list of sign keys from sign_schema.json

        Returns a ClinicalDecision with all classifications triggered.
        """
        observed = set(observed_signs)
        decision = ClinicalDecision(input_signs=sorted(observed))

        # Assess in the protocol's specified order, skipping the entry-point node
        entry = self.nodes["assess_young_infant"]
        for node_key in entry["sequence"]:
            node = self.nodes.get(node_key)
            if node is None:
                # Node defined in entry sequence but not yet implemented
                decision.unresolved.append(node_key)
                continue

            if node.get("type") != "classification":
                # Flowcharts (plan_c) and others handled separately
                continue

            classification = self._classify_node(node, observed)
            if classification is not None:
                decision.classifications.append(classification)

        # Sort by priority (lower number = higher priority)
        decision.classifications.sort(key=lambda c: c.priority)
        return decision

    def evaluate_referral_refused(self, observed_signs: list[str]) -> ClinicalDecision:
        """
        When PSBI was triggered but referral is refused/not feasible,
        reclassify using the referral-refused node.
        """
        observed = set(observed_signs)
        decision = ClinicalDecision(input_signs=sorted(observed))

        node = self.nodes["psbi_referral_refused"]
        classification = self._classify_node(node, observed)
        if classification is not None:
            decision.classifications.append(classification)

        return decision

    def get_plan_c_path(self, answers: dict[str, bool]) -> dict[str, Any]:
        """
        Traverse the Plan C flowchart.

        answers: dict of question_key -> bool
        keys: "iv_immediate", "iv_nearby", "trained_ng_tube", "infant_can_drink"

        Returns the action list for the reached node.
        """
        path = self.nodes["plan_c"]["decision_path"][0]

        if answers.get("iv_immediate"):
            return {"branch": "iv_immediate", "actions": path["yes"]["actions"]}

        q2 = path["no"]
        if answers.get("iv_nearby"):
            return {"branch": "iv_nearby", "actions": q2["yes"]["actions"]}

        q3 = q2["no"]
        if answers.get("trained_ng_tube"):
            return {"branch": "trained_ng_tube", "actions": q3["yes"]["actions"]}

        q4 = q3["no"]
        if answers.get("infant_can_drink"):
            return {"branch": "oral_rehydration", "actions": q4["yes"]["actions"]}

        return {"branch": "refer_urgently", "actions": q4["no"]["actions"]}


        # internal classification logic

    def _classify_node(self, node: dict, observed: set[str]) -> Classification | None:
        """Apply a classification node's rules to observed signs."""
        for classification in node["classifications"]:
            if classification.get("default"):
                continue

            criteria = classification.get("criteria_any_of")
            if criteria is not None:
                matched = [s for s in criteria if s in observed]
                if matched:
                    return self._build_classification(node, classification, matched)
                continue

            # Diarrhoea-style: count_at_least
            criteria_obj = classification.get("criteria")
            if criteria_obj is not None:
                signs = criteria_obj.get("signs", [])
                required = criteria_obj.get("count_at_least", 1)
                matched = [s for s in signs if s in observed]
                if len(matched) >= required:
                    return self._build_classification(node, classification, matched)
                continue

        # Fall through to default
        for classification in node["classifications"]:
            if classification.get("default"):
                return self._build_classification(node, classification, [], is_default=True)

        return None


    def _build_classification(
        self,
        node: dict,
        classification: dict,
        matched_signs: list[str],
        is_default: bool = False,
    ) -> Classification:
        return Classification(
            name=classification["name"],
            node_id=node["node_id"],
            priority=classification.get("priority", 99),
            severity=classification.get("severity", "none" if is_default else "unknown"),
            matched_signs=matched_signs,
            actions=classification.get("actions", {}),
            is_default=is_default,
        )
    @staticmethod
    def _extract_criteria_signs(classification: dict) -> list[str]:
        # handle the diarrhoea-style criteria : {count_at_least, signs: [...]}
        criteria = classification.get("criteria")
        if criteria and "signs" in criteria:
            return criteria["signs"]
        return []

# CLI for quick testing
if __name__ == "__main__":
    import sys

    protocol_path = Path(__file__).resolve().parents[2] / "data" / "protocol" / "imci_young_infant.json"
    engine = IMCIEngine(protocol_path)

    # Smoke test: an infant with two PSBI danger signs
    test_signs = ["not_feeding_well", "high_temp_38_plus"]
    decision = engine.evaluate(test_signs)
    print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))
