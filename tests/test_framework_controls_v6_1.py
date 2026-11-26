from pathlib import Path


def test_framework_control_has_no_nested_expander_surface():
    import awareml.ui_v2.framework_control as fc
    src = Path(fc.__file__).read_text(encoding="utf-8")
    assert "st.expander(" not in src
    assert "st.container(border=True)" in src


def test_every_framework_exposes_algorithm_ui_and_oaml_owns_mode():
    import awareml.ui_v2.framework_control as fc
    src = Path(fc.__file__).read_text(encoding="utf-8")
    assert '"Algorithms"' in src
    assert '"Algorithm"' in src
    assert '"Execution mode"' in src
    assert 'framework == "OAML"' in src


def test_catalog_has_oaml_mode_and_all_algorithm_families():
    from awareml.frameworks.catalog import framework_catalog

    cat = framework_catalog()
    assert cat["OAML"]["global"]["mode"]["choices"] == ["online", "gama"]
    assert "EvolutionaryBaggingClassifier" in cat["EvoAutoML"]["algorithms"]
    assert "AutoVW" in cat["ChaCha"]["algorithms"]
    assert "HoeffdingAdaptiveTreeClassifier" in cat["OAML"]["algorithms"]
    assert "GAMA AutoPipeline" in cat["OAML"]["algorithms"]
    assert "HoeffdingAdaptiveTreeClassifier" in cat["AutoClass"]["algorithms"]
    assert "AdaptiveRandomForestClassifier" in cat["AutoStreamML"]["algorithms"]
