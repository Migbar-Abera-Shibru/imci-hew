import json
from pathlib import Path

from protocol.engine import IMCIEngine

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "data" / "protocol" / "imci_young_infant.json"

def test_psbi_triggered_on_single_danger_sign():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate(["not_feeding_well"])
    names = [c.name for c in decision.classifications]
    assert "POSSIBLE_SERIOUS_BACTERIAL_INFECTION" in names


def test_psbi_not_triggered_when_no_signs():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate([])
    names = [c.name for c in decision.classifications]
    assert "POSSIBLE_SERIOUS_BACTERIAL_INFECTION" not in names
    assert "INFECTION_UNLIKELY" in names


def test_local_bacterial_infection():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate(["umbilicus_red_or_pus"])
    names = [c.name for c in decision.classifications]
    assert "LOCAL_BACTERIAL_INFECTION" in names
    assert "POSSIBLE_SERIOUS_BACTERIAL_INFECTION" not in names


def test_psbi_overrides_local_infection():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate(["umbilicus_red_or_pus", "convulsions"])
    names = [c.name for c in decision.classifications]
    assert "POSSIBLE_SERIOUS_BACTERIAL_INFECTION" in names


def test_priority_order_correct():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate(["convulsions"])
    assert decision.classifications[0].name == "POSSIBLE_SERIOUS_BACTERIAL_INFECTION"
    assert decision.classifications[0].priority == 1


def test_severe_dehydration_two_signs():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate(["sunken_eyes", "skin_pinch_very_slow"])
    names = [c.name for c in decision.classifications]
    assert "SEVERE_DEHYDRATION" in names


def test_some_dehydration_two_signs():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate(["restless_irritable", "sunken_eyes"])
    names = [c.name for c in decision.classifications]
    assert "SOME_DEHYDRATION" in names


def test_no_dehydration_default():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate([])
    names = [c.name for c in decision.classifications]
    assert "NO_DEHYDRATION" in names


def test_referral_refused_critical_illness():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate_referral_refused(["convulsions"])
    names = [c.name for c in decision.classifications]
    assert "CRITICAL_ILLNESS" in names


def test_referral_refused_clinical_severe():
    engine = IMCIEngine(PROTOCOL_PATH)
    decision = engine.evaluate_referral_refused(["not_feeding_well"])
    names = [c.name for c in decision.classifications]
    assert "CLINICAL_SEVERE_INFECTION" in names


def test_plan_c_iv_immediate():
    engine = IMCIEngine(PROTOCOL_PATH)
    result = engine.get_plan_c_path({"iv_immediate": True})
    assert result["branch"] == "iv_immediate"
    assert any("intravenous fluid" in a.lower() for a in result["actions"])


def test_plan_c_refer_urgently():
    engine = IMCIEngine(PROTOCOL_PATH)
    result = engine.get_plan_c_path({
        "iv_immediate": False,
        "iv_nearby": False,
        "trained_ng_tube": False,
        "infant_can_drink": False,
    })
    assert result["branch"] == "refer_urgently"