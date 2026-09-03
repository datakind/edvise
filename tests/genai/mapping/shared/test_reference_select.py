"""Tests for rules-based GenAI reference selection and run few-shot snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edvise.genai.mapping.shared import reference_pin as rp
from edvise.genai.mapping.shared import reference_select as rs


def _dataset(
    *,
    cols: dict[str, str],
    unique_keys: list[str],
    term: dict | None = None,
) -> dict:
    training: dict = {
        "file_path": "/x.csv",
        "num_rows": 1,
        "num_columns": len(cols),
        "column_normalization": {"original_to_normalized": {}},
        "column_details": [],
    }
    if term is not None:
        training["term_normalization"] = term
    return {
        "normalized_columns": cols,
        "dtypes": {v: "string" for v in cols.values()},
        "non_null_columns": [],
        "unique_keys": unique_keys,
        "null_tokens": [],
        "boolean_map": {},
        "training": training,
    }


def _contract(
    school_id: str,
    datasets: dict[str, dict],
) -> dict:
    return {
        "school_id": school_id,
        "school_name": school_id,
        "datasets": datasets,
    }


def test_jaccard_edge_cases() -> None:
    assert rs.jaccard(set(), set()) == 1.0
    assert rs.jaccard({"a"}, set()) == 0.0
    assert rs.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_score_perfect_match() -> None:
    term = {
        "mode": "single_column",
        "term_extraction": "standard",
        "term_col": "term",
        "clean_spec_term_column": "term",
    }
    ds = {
        "student": _dataset(
            cols={"SID": "student_id", "TERM": "term"},
            unique_keys=["student_id", "term"],
            term=term,
        ),
        "course": _dataset(
            cols={"SID": "student_id", "COURSE": "course_id"},
            unique_keys=["student_id", "course_id"],
        ),
    }
    q = _contract("query", ds)
    c = _contract("cand", ds)
    score = rs.score_reference_contracts(q, c, reference_id="cand")
    assert score.score_datasets == 1.0
    assert score.score_columns == 1.0
    assert score.score_grain == 1.0
    assert score.score_term == 1.0
    assert score.total == pytest.approx(1.0)
    assert score.shared_datasets == ("course", "student")


def test_score_no_shared_datasets_zeros_contextual_signals() -> None:
    q = _contract(
        "q",
        {"student": _dataset(cols={"A": "a"}, unique_keys=["a"])},
    )
    c = _contract(
        "c",
        {"course": _dataset(cols={"A": "a"}, unique_keys=["a"])},
    )
    score = rs.score_reference_contracts(q, c, reference_id="c")
    assert score.score_datasets == 0.0
    assert score.score_columns == 0.0
    assert score.score_grain == 0.0
    assert score.score_term == 0.0
    assert score.total == 0.0


def test_score_uses_normalized_values_not_original_keys() -> None:
    q = _contract(
        "q",
        {
            "student": _dataset(
                cols={"RAW_SID": "student_id", "RAW_TERM": "term"},
                unique_keys=["student_id", "term"],
            )
        },
    )
    # Same cleaned names, different originals → full column match.
    c = _contract(
        "c",
        {
            "student": _dataset(
                cols={"OTHER_SID": "student_id", "OTHER_TERM": "term"},
                unique_keys=["student_id", "term"],
            )
        },
    )
    score = rs.score_reference_contracts(q, c, reference_id="c")
    assert score.score_columns == 1.0


def test_score_term_comparable_only_when_both_present() -> None:
    q = _contract(
        "q",
        {
            "student": _dataset(
                cols={"A": "a"},
                unique_keys=["a"],
                term={
                    "mode": "single_column",
                    "term_extraction": "standard",
                    "term_col": "term",
                    "clean_spec_term_column": "term",
                },
            )
        },
    )
    c = _contract(
        "c",
        {"student": _dataset(cols={"A": "a"}, unique_keys=["a"], term=None)},
    )
    score = rs.score_reference_contracts(q, c, reference_id="c")
    assert score.score_term == 0.0


def test_score_term_disagreement() -> None:
    q = _contract(
        "q",
        {
            "student": _dataset(
                cols={"A": "a"},
                unique_keys=["a"],
                term={
                    "mode": "single_column",
                    "term_extraction": "standard",
                    "term_col": "term",
                    "clean_spec_term_column": "term",
                },
            )
        },
    )
    c = _contract(
        "c",
        {
            "student": _dataset(
                cols={"A": "a"},
                unique_keys=["a"],
                term={
                    "mode": "year_season_columns",
                    "term_extraction": "standard",
                    "year_col": "year",
                    "season_col": "season",
                    "clean_spec_term_column": "year",
                },
            )
        },
    )
    score = rs.score_reference_contracts(q, c, reference_id="c")
    assert score.score_term == 0.0


def test_weights_sum_to_one() -> None:
    assert (
        rs.WEIGHT_DATASETS + rs.WEIGHT_COLUMNS + rs.WEIGHT_GRAIN + rs.WEIGHT_TERM
        == pytest.approx(1.0)
    )


def _patch_reference_current_root(
    monkeypatch: pytest.MonkeyPatch, volumes: Path
) -> None:
    def _current(reference_id: str, *, catalog: str) -> str:
        return str(
            volumes
            / catalog
            / "genai_mapping"
            / "references"
            / reference_id
            / "current"
        )

    monkeypatch.setattr(rp, "genai_reference_current_root", _current)
    monkeypatch.setattr(rs, "genai_reference_current_root", _current)


def _write_pin(
    volumes: Path,
    *,
    catalog: str,
    reference_id: str,
    contract: dict,
    manifest: dict | None = None,
    transformation: dict | None = None,
) -> str:
    current = (
        volumes / catalog / "genai_mapping" / "references" / reference_id / "current"
    )
    current.mkdir(parents=True)
    m_path = current / "manifest_map.json"
    t_path = current / "transformation_map.json"
    c_path = current / "enriched_schema_contract.json"
    m_path.write_text(json.dumps(manifest or {"m": reference_id}), encoding="utf-8")
    t_path.write_text(
        json.dumps(transformation or {"t": reference_id}), encoding="utf-8"
    )
    c_path.write_text(json.dumps(contract), encoding="utf-8")
    content_hash = rp.compute_reference_content_hash(
        {
            "manifest_map.json": m_path,
            "transformation_map.json": t_path,
            "enriched_schema_contract.json": c_path,
        }
    )
    pin = {
        "schema_version": 1,
        "reference_id": reference_id,
        "source_institution_id": reference_id,
        "pinned_at": "2026-01-01T00:00:00Z",
        "content_hash": content_hash,
        "artifacts": [
            "enriched_schema_contract.json",
            "manifest_map.json",
            "transformation_map.json",
        ],
        "source": "active",
        "uc_catalog": catalog,
    }
    (current / rp.GENAI_REFERENCE_PIN_BASENAME).write_text(
        json.dumps(pin, indent=2) + "\n", encoding="utf-8"
    )
    return content_hash


def test_select_picks_higher_column_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = "dev_sst_02"
    volumes = tmp_path / "Volumes"
    _patch_reference_current_root(monkeypatch, volumes)

    query = _contract(
        "new_school",
        {
            "student": _dataset(
                cols={"A": "student_id", "B": "gpa", "C": "term"},
                unique_keys=["student_id", "term"],
            )
        },
    )
    close = _contract(
        "close_ref",
        {
            "student": _dataset(
                cols={"A": "student_id", "B": "gpa", "C": "term"},
                unique_keys=["student_id", "term"],
            )
        },
    )
    far = _contract(
        "far_ref",
        {
            "student": _dataset(
                cols={"A": "student_id", "Z": "unrelated"},
                unique_keys=["student_id"],
            )
        },
    )
    _write_pin(volumes, catalog=catalog, reference_id="close_ref", contract=close)
    _write_pin(volumes, catalog=catalog, reference_id="far_ref", contract=far)

    result = rs.select_reference_id(
        catalog=catalog,
        institution_id="new_school",
        query_contract=query,
        active_pins=[
            {"reference_id": "far_ref"},
            {"reference_id": "close_ref"},
        ],
    )
    assert result.selection_mode == "auto"
    assert result.reference_id == "close_ref"
    assert result.scores[0].reference_id == "close_ref"
    assert result.scores[0].total > result.scores[1].total


def test_select_tie_break_lexicographic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = "dev_sst_02"
    volumes = tmp_path / "Volumes"
    _patch_reference_current_root(monkeypatch, volumes)

    ds = {
        "student": _dataset(
            cols={"A": "student_id"},
            unique_keys=["student_id"],
        )
    }
    query = _contract("new_school", ds)
    identical = _contract("x", ds)
    _write_pin(volumes, catalog=catalog, reference_id="beta_ref", contract=identical)
    _write_pin(volumes, catalog=catalog, reference_id="alpha_ref", contract=identical)

    result = rs.select_reference_id(
        catalog=catalog,
        institution_id="new_school",
        query_contract=query,
        active_pins=[
            {"reference_id": "beta_ref"},
            {"reference_id": "alpha_ref"},
        ],
    )
    assert result.reference_id == "alpha_ref"


def test_select_excludes_self(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = "dev_sst_02"
    volumes = tmp_path / "Volumes"
    _patch_reference_current_root(monkeypatch, volumes)

    ds = {
        "student": _dataset(
            cols={"A": "student_id", "B": "gpa"},
            unique_keys=["student_id"],
        )
    }
    query = _contract("gold_school", ds)
    _write_pin(volumes, catalog=catalog, reference_id="gold_school", contract=query)
    other = _contract(
        "other",
        {
            "student": _dataset(
                cols={"A": "student_id"},
                unique_keys=["student_id"],
            )
        },
    )
    _write_pin(volumes, catalog=catalog, reference_id="other", contract=other)

    result = rs.select_reference_id(
        catalog=catalog,
        institution_id="gold_school",
        query_contract=query,
        active_pins=[
            {"reference_id": "gold_school"},
            {"reference_id": "other"},
        ],
    )
    assert result.reference_id == "other"
    assert any(s["reference_id"] == "gold_school" for s in result.skipped)


def test_select_skips_pin_without_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = "dev_sst_02"
    volumes = tmp_path / "Volumes"
    _patch_reference_current_root(monkeypatch, volumes)

    query = _contract(
        "new_school",
        {"student": _dataset(cols={"A": "a"}, unique_keys=["a"])},
    )
    good = _contract(
        "good",
        {"student": _dataset(cols={"A": "a"}, unique_keys=["a"])},
    )
    _write_pin(volumes, catalog=catalog, reference_id="good", contract=good)

    # Pin without enriched contract: required few-shot only.
    bad_current = volumes / catalog / "genai_mapping" / "references" / "bad" / "current"
    bad_current.mkdir(parents=True)
    m = bad_current / "manifest_map.json"
    t = bad_current / "transformation_map.json"
    m.write_text("{}", encoding="utf-8")
    t.write_text("{}", encoding="utf-8")
    h = rp.compute_reference_content_hash(
        {"manifest_map.json": m, "transformation_map.json": t}
    )
    (bad_current / rp.GENAI_REFERENCE_PIN_BASENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reference_id": "bad",
                "content_hash": h,
                "artifacts": ["manifest_map.json", "transformation_map.json"],
            }
        ),
        encoding="utf-8",
    )

    result = rs.select_reference_id(
        catalog=catalog,
        institution_id="new_school",
        query_contract=query,
        active_pins=[{"reference_id": "bad"}, {"reference_id": "good"}],
    )
    assert result.reference_id == "good"
    assert any(s["reference_id"] == "bad" for s in result.skipped)


def test_ensure_run_few_shot_reuses_without_reselect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = "dev_sst_02"
    volumes = tmp_path / "Volumes"
    _patch_reference_current_root(monkeypatch, volumes)

    contract = _contract(
        "ref_a",
        {"student": _dataset(cols={"A": "a"}, unique_keys=["a"])},
    )
    _write_pin(volumes, catalog=catalog, reference_id="ref_a", contract=contract)

    sma_root = tmp_path / "sma_run"
    sma_root.mkdir()
    query = _contract(
        "new_school",
        {"student": _dataset(cols={"A": "a"}, unique_keys=["a"])},
    )
    snap1, sel1 = rs.ensure_run_few_shot(
        sma_root,
        catalog=catalog,
        institution_id="new_school",
        query_contract=query,
        active_pins=[{"reference_id": "ref_a"}],
    )
    assert sel1 is not None
    assert snap1.reference_id == "ref_a"

    # Second call must reuse even if library listing would differ.
    snap2, sel2 = rs.ensure_run_few_shot(
        sma_root,
        catalog=catalog,
        institution_id="new_school",
        query_contract=query,
        active_pins=[],  # would fail select if re-run
    )
    assert sel2 is None
    assert snap2.reference_id == "ref_a"


def test_materialize_and_reuse_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = "dev_sst_02"
    volumes = tmp_path / "Volumes"
    _patch_reference_current_root(monkeypatch, volumes)

    contract = _contract(
        "ref_a",
        {"student": _dataset(cols={"A": "a"}, unique_keys=["a"])},
    )
    content_hash = _write_pin(
        volumes, catalog=catalog, reference_id="ref_a", contract=contract
    )

    sma_root = tmp_path / "sma_run"
    sma_root.mkdir()
    selection = rs.ReferenceSelectionResult(
        reference_id="ref_a",
        content_hash=content_hash,
        selection_mode="auto",
        scores=[],
        skipped=[],
    )
    snap1 = rs.materialize_run_few_shot_snapshot(
        sma_root,
        catalog=catalog,
        reference_id="ref_a",
        selection=selection,
    )
    assert snap1.manifest_map.is_file()
    assert snap1.transformation_map.is_file()
    assert (snap1.root / rp.GENAI_REFERENCE_PIN_BASENAME).is_file()
    assert (snap1.root / rs.SELECTION_AUDIT_BASENAME).is_file()
    assert snap1.content_hash == content_hash

    # Mutate library current/ — run snapshot must still reuse old bytes.
    lib_manifest = (
        volumes
        / catalog
        / "genai_mapping"
        / "references"
        / "ref_a"
        / "current"
        / "manifest_map.json"
    )
    lib_manifest.write_text('{"m": "CHANGED"}', encoding="utf-8")

    snap2 = rs.materialize_run_few_shot_snapshot(
        sma_root,
        catalog=catalog,
        reference_id="ref_a",
        selection=selection,
    )
    assert snap2.root == snap1.root
    assert snap2.manifest_map.read_text(encoding="utf-8") == json.dumps({"m": "ref_a"})
    resolved = rs.resolve_run_few_shot_snapshot(sma_root)
    assert resolved is not None
    assert resolved.reference_id == "ref_a"


def test_gate2_style_resolve_does_not_touch_library_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gate_2 path: resolve_run_few_shot_snapshot only — library may be gone."""
    catalog = "dev_sst_02"
    volumes = tmp_path / "Volumes"
    _patch_reference_current_root(monkeypatch, volumes)

    contract = _contract(
        "ref_a",
        {"student": _dataset(cols={"A": "a"}, unique_keys=["a"])},
    )
    _write_pin(volumes, catalog=catalog, reference_id="ref_a", contract=contract)

    sma_root = tmp_path / "sma_run"
    sma_root.mkdir()
    rs.materialize_run_few_shot_snapshot(
        sma_root, catalog=catalog, reference_id="ref_a"
    )

    # Delete library current/ entirely.
    import shutil

    shutil.rmtree(
        volumes / catalog / "genai_mapping" / "references" / "ref_a" / "current"
    )

    snap = rs.resolve_run_few_shot_snapshot(sma_root)
    assert snap is not None
    assert snap.manifest_map.is_file()
    assert json.loads(snap.transformation_map.read_text(encoding="utf-8")) == {
        "t": "ref_a"
    }


def test_select_zero_candidates_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = "dev_sst_02"
    volumes = tmp_path / "Volumes"
    _patch_reference_current_root(monkeypatch, volumes)
    query = _contract(
        "new_school",
        {"student": _dataset(cols={"A": "a"}, unique_keys=["a"])},
    )
    with pytest.raises(ValueError, match="No selectable active references"):
        rs.select_reference_id(
            catalog=catalog,
            institution_id="new_school",
            query_contract=query,
            active_pins=[],
        )
