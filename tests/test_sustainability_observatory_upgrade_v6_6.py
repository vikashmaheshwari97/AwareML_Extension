from pathlib import Path

import pandas as pd

from awareml.ui_v2.sustainability_observatory_v6_6 import (
    accuracy_cost_table,
    carbon_accounting_figure,
    efficiency_ratio_figure,
    resource_signature_figure,
    sustainability_leaders,
    tradeoff_atlas_figure,
    tradeoff_sentence,
)


def _frame():
    return pd.DataFrame(
        [
            {"Framework":"AutoStreamML","Accuracy":0.712,"Runtime s":12.46,"Energy kWh":0.000216,"CO₂ kg":0.0000902},
            {"Framework":"AutoClass","Accuracy":0.696,"Runtime s":10.49,"Energy kWh":0.000182,"CO₂ kg":0.0000759},
            {"Framework":"EvoAutoML","Accuracy":0.687,"Runtime s":32.99,"Energy kWh":0.000573,"CO₂ kg":0.000239},
            {"Framework":"OAML","Accuracy":0.636,"Runtime s":8.37,"Energy kWh":0.000145,"CO₂ kg":0.0000605},
            {"Framework":"ChaCha","Accuracy":0.594,"Runtime s":14.54,"Energy kWh":0.000253,"CO₂ kg":0.000105},
        ]
    )


def test_research_figures_have_traces():
    frame = _frame()
    assert len(resource_signature_figure(frame).data) >= 15
    assert len(efficiency_ratio_figure(frame).data) >= 3
    assert len(tradeoff_atlas_figure(frame).data) >= 5
    assert len(carbon_accounting_figure(frame).data) >= 5


def test_tradeoff_ledger_ratios_are_transparent():
    ledger = accuracy_cost_table(_frame())
    assert abs(float(ledger["Energy × best"].min()) - 1.0) < 1e-12
    assert abs(float(ledger["Runtime × best"].min()) - 1.0) < 1e-12
    assert abs(float(ledger["CO₂ × best"].min()) - 1.0) < 1e-12


def test_leaders_and_sentence():
    frame = _frame()
    leaders = sustainability_leaders(frame)
    assert leaders["accuracy"] == "AutoStreamML"
    assert leaders["energy"] == "OAML"
    assert leaders["co2"] == "OAML"
    sentence = tradeoff_sentence(frame)
    assert "AutoStreamML" in sentence
    assert "OAML" in sentence
    assert "×" in sentence


def test_page_uses_new_observatory():
    text = Path("awareml/ui_v2/pages_specialist.py").read_text(encoding="utf-8")
    assert "SUSTAINABILITY OBSERVATORY" in text
    assert "r126_sustain_resource_signature" in text
    assert "r126_sustain_tradeoff_atlas" in text
    assert "r126_sustain_carbon_accounting" in text
    assert "Framework sustainability matrix" not in text[text.find("def sustainability_v2_page():"):text.find("def trust_calibration_v2_page():")]
