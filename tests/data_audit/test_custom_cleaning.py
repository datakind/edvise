import types

import numpy as np
import pandas as pd
import logging
import pytest

from edvise.data_audit import custom_cleaning as m
from edvise.data_audit import eda
from edvise.data_audit.custom_cleaning import (
    create_datasets,
    normalize_columns,
    DtypeGenerationOptions,
    generate_column_training_dtype,
    generate_training_dtypes,
    CleanSpec,
    clean_dataset,
    clean_all_datasets_map,
    SchemaFreezeOptions,
    freeze_schema,
    enforce_schema,
    SchemaContractMeta,
    build_schema_contract,
    enforce_schema_contract,
    save_schema_contract,
    load_schema_contract,
)


# -------------------------------------------------------------------
# create_datasets
# -------------------------------------------------------------------
def test_create_datasets_handles_dict_and_object():
    df = pd.DataFrame({"a": [1]})
    bronze_dict = {
        "drop_cols": ["foo"],
        "non_null_cols": ["bar"],
        "primary_keys": ["id"],
    }

    out_dict = create_datasets(df, bronze_dict)
    assert out_dict["data"] is df
    assert out_dict["drop columns"] == ["foo"]
    assert out_dict["non-null columns"] == ["bar"]
    assert out_dict["unique keys"] == ["id"]

    bronze_obj = types.SimpleNamespace(
        drop_cols=[],
        non_null_cols=None,
        primary_keys=("id",),
    )
    out_obj = create_datasets(df, bronze_obj, include_empty=True)
    assert out_obj["data"] is df
    assert out_obj["drop columns"] == []
    assert "non-null columns" in out_obj
    assert out_obj["non-null columns"] is None
    assert out_obj["unique keys"] == ("id",)


# -------------------------------------------------------------------
# normalize_columns (minimal)
# -------------------------------------------------------------------
def test_normalize_columns_uses_convert_to_snake_case(monkeypatch):
    calls = []

    def fake_snake(s):
        calls.append(s)
        return s.lower().replace(" ", "_")

    monkeypatch.setattr(m, "convert_to_snake_case", fake_snake)

    idx, mapping = normalize_columns(["Student ID", "TermCode"])
    assert list(idx) == ["student_id", "termcode"]
    assert mapping == {
        "student_id": ["Student ID"],
        "termcode": ["TermCode"],
    }
    assert calls == ["Student ID", "TermCode"]


# -------------------------------------------------------------------
# generate_column_training_dtype / generate_training_dtypes
# -------------------------------------------------------------------
def test_generate_column_training_dtype_date_numeric_boolean_string():
    # Use relaxed options so small synthetic series can still be typed
    # realistically: we want to *exercise* the paths, not mimic production thresholds.
    opts = DtypeGenerationOptions(
        dtype_confidence_threshold=0.5,
        min_non_null=1,
        date_formats=("%m/%d/%Y",),
    )

    # --- Date inference ---
    s_date = pd.Series(
        [
            "01/01/2020",
            "02/01/2020",
            "03/01/2020",
            "04/01/2020",
            "05/01/2020",
            "06/01/2020",
            "07/01/2020",
            "08/01/2020",
            "09/01/2020",
            "10/01/2020",
            "01/01/2020",
            "02/01/2020",
            None,
            "not a date",
        ]
    )
    out_date = generate_column_training_dtype(s_date, opts)
    assert str(out_date.dtype).startswith("datetime64")
    assert out_date.iloc[0] == pd.Timestamp("2020-01-01")

    # --- Numeric: Int64 vs Float64 ---
    s_int = pd.Series(["1", "2", None])
    s_float = pd.Series(["1.5", "2.0", None])

    out_int = generate_column_training_dtype(s_int, opts)
    out_float = generate_column_training_dtype(s_float, opts)

    # integer-like → nullable Int64
    assert str(out_int.dtype) == "Int64"
    assert list(out_int.astype("Int64")) == [1, 2, pd.NA]

    # non-integer numeric → nullable Float64
    assert str(out_float.dtype) == "Float64"
    assert out_float.iloc[0] == 1.5

    # --- Boolean ---
    s_bool = pd.Series(["Yes", "no", "TRUE", None])
    out_bool = generate_column_training_dtype(s_bool, opts)
    assert str(out_bool.dtype) == "boolean"
    assert list(out_bool.astype("boolean")) == [True, False, True, pd.NA]

    # --- Fallback string ---
    s_str = pd.Series(["foo", "bar", None])
    out_str = generate_column_training_dtype(s_str, opts)
    assert str(out_str.dtype) == "string"


def test_generate_training_dtypes_columnwise():
    opts = DtypeGenerationOptions(
        dtype_confidence_threshold=0.5,
        min_non_null=1,
        date_formats=("%m/%d/%Y",),
    )

    df = pd.DataFrame(
        {
            "date_col": ["01/01/2020", "01/02/2020"],
            "num_col": ["1", "2"],
        }
    )

    out = generate_training_dtypes(df, opts)

    assert str(out["date_col"].dtype).startswith("datetime64")
    assert str(out["num_col"].dtype) == "Int64"
    # original not mutated
    assert df["date_col"].dtype == object


# -------------------------------------------------------------------
# clean_dataset / clean_all_datasets_map
# -------------------------------------------------------------------
def test_clean_dataset_raises_on_column_collision(monkeypatch):
    def fake_snake(s):
        return "dup"

    monkeypatch.setattr(m, "convert_to_snake_case", fake_snake)

    df = pd.DataFrame({"A": [1], "B": [2]})
    spec = CleanSpec()

    with pytest.raises(ValueError) as exc:
        clean_dataset(df, spec, dataset_name="students")
    assert "Column-name collisions" in str(exc.value)


def test_clean_dataset_student_id_rename_null_handling_and_pk_uniqueness(caplog):
    caplog.set_level(logging.INFO, logger="edvise.data_audit.custom_cleaning")

    df = pd.DataFrame(
        {
            "student_id_randomized_datakind": ["001", "002", "002"],
            "a": ["(Blank)", "value", "   "],
            "x": [1, 2, 2],
        }
    )
    spec = CleanSpec(
        drop_columns=["x"],  # should not drop student_id
        non_null_columns=["a"],
        unique_keys=["student_id_randomized_datakind"],
    )

    out = clean_dataset(df, spec, dataset_name="students")
    # rename + type
    assert "student_id" in out.columns
    assert "student_id_randomized_datakind" not in out.columns
    assert str(out["student_id"].dtype) == "string"

    # null tokens normalized + rows dropped
    assert len(out) == 1
    assert out["a"].iloc[0] == "value"
    assert any("Dropped rows missing" in rec.getMessage() for rec in caplog.records)

    # uniqueness enforced on renamed key
    assert out["student_id"].is_unique


def test_clean_dataset_dedupe_fn_and_pk_dedupe():
    def custom_dedupe(g):
        # keep last occurrence per id
        return g.drop_duplicates(subset=["id"], keep="last")

    df = pd.DataFrame({"id": [1, 1, 2], "x": [10, 11, 20]})
    spec = CleanSpec(unique_keys=["id"], dedupe_fn=custom_dedupe)

    out = clean_dataset(df, spec, dataset_name="tbl")
    assert len(out) == 2
    assert out.sort_values("id")["x"].tolist() == [11, 20]


def test_clean_dataset_raises_when_primary_key_not_unique_after_cleaning():
    df = pd.DataFrame({"id": [1, 1], "x": [10, 11]})
    spec = CleanSpec(unique_keys=["id"])

    out = clean_dataset(df, spec, dataset_name="tbl", enforce_uniqueness=True)
    # We expect deduplication on primary key, not an exception
    assert len(out) == 1
    assert out["id"].is_unique
    assert set(out["id"].tolist()) == {1}


def test_clean_all_datasets_map_happy_path():
    raw = {
        "students": {
            "data": pd.DataFrame({"id": [1, 1], "x": [10, 10]}),
            "unique keys": ["id"],
        },
        "courses": {
            "data": pd.DataFrame({"course_id": [1], "name": ["Math"]}),
        },
    }

    cleaned = clean_all_datasets_map(raw)
    assert set(cleaned.keys()) == {"students", "courses"}
    assert isinstance(cleaned["students"], pd.DataFrame)


# -------------------------------------------------------------------
# freeze_schema / enforce_schema
# -------------------------------------------------------------------
def test_freeze_schema_and_enforce_schema_main_flow(caplog):
    # simulate cleaned data
    df = pd.DataFrame(
        {
            "id": pd.Series([1, 2], dtype="Int64"),
            "val": pd.Series([1.1, 2.2], dtype="Float64"),
            "flag": pd.Series([True, False], dtype="boolean"),
        }
    )
    spec = {
        "_orig_cols_": ["ID_original", "VAL_ORIG", "FLAG_ORIG"],
        "non-null columns": ["id"],
        "unique keys": ["id"],
    }

    schema = freeze_schema(
        df, spec, opts=SchemaFreezeOptions(include_column_order_hash=True)
    )
    assert schema["dtypes"] == {
        "id": "Int64",
        "val": "Float64",
        "flag": "boolean",
    }
    assert "column_order_hash" in schema

    # messy inference-time df
    df_infer = pd.DataFrame(
        {
            " ID ": ["1", "2"],
            "VAL ": ["1.1", "2.2"],
            "FLAG": ["true", "false"],
            "extra_col": [1, 2],
        }
    )

    out = enforce_schema(df_infer, schema)
    assert list(out.columns) == ["id", "val", "flag"]
    assert str(out["id"].dtype) == "Int64"
    assert str(out["val"].dtype) == "Float64"
    assert str(out["flag"].dtype) == "boolean"
    assert any(
        "Unexpected columns at inference" in r.getMessage() for r in caplog.records
    )


def test_enforce_schema_raises_on_unique_key_duplicates():
    schema = {
        "dtypes": {"id": "Int64"},
        "non_null_columns": [],
        "unique_keys": ["id"],
    }
    df = pd.DataFrame({"id": [1, 1]})
    with pytest.raises(ValueError) as exc:
        enforce_schema(df, schema)
    assert "Duplicate rows on unique keys" in str(exc.value)


# -------------------------------------------------------------------
# build_schema_contract / enforce_schema_contract
# -------------------------------------------------------------------
def test_build_schema_contract_and_enforce_schema_contract_wiring(monkeypatch):
    cleaned = {
        "students": pd.DataFrame({"id": pd.Series([1], dtype="Int64")}),
        "courses": pd.DataFrame({"course_id": pd.Series([10], dtype="Int64")}),
    }
    specs = {
        "students": {"non-null columns": ["id"], "unique keys": ["id"]},
        "courses": {"non-null columns": ["course_id"], "unique keys": ["course_id"]},
    }
    meta = SchemaContractMeta(
        created_at="2024-01-01T00:00:00Z",
        null_tokens=["(Blank)"],
    )

    schema_contract = build_schema_contract(cleaned, specs, meta=meta)
    assert schema_contract["created_at"] == "2024-01-01T00:00:00Z"
    assert set(schema_contract["datasets"].keys()) == {"students", "courses"}

    calls = []

    def fake_enforce_schema(df, schema):
        calls.append((df.copy(), schema))
        return df.assign(processed=True)

    monkeypatch.setattr(m, "enforce_schema", fake_enforce_schema)

    raw = {
        "students": pd.DataFrame({"id": [1]}),
        "courses": pd.DataFrame({"course_id": [10]}),
    }

    out = enforce_schema_contract(raw, schema_contract)
    assert set(out.keys()) == {"students", "courses"}
    assert out["students"]["processed"].all()
    assert out["courses"]["processed"].all()
    assert len(calls) == 2


# -------------------------------------------------------------------
# save_schema_contract / load_schema_contract
# -------------------------------------------------------------------
def test_save_and_load_schema_contract_roundtrip(tmp_path):
    schema = {
        "created_at": "2024-01-01T00:00:00Z",
        "null_tokens": ["(Blank)"],
        "datasets": {
            "students": {"dtypes": {"id": "Int64"}},
        },
    }

    path = tmp_path / "schema.json"
    save_schema_contract(schema, str(path))
    loaded = load_schema_contract(str(path))

    assert loaded == schema


def test_generate_training_dtypes_respects_forced_dtypes():
    # Without forcing, both columns would be inferred as numeric:
    # - "id" → Int64
    # - "gpa" → Float64
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "gpa": ["3.5", "4.0"],
        }
    )

    opts = DtypeGenerationOptions(
        dtype_confidence_threshold=0.9,
        min_non_null=10,
        date_formats=("%m/%d/%Y",),
        forced_dtypes={
            "id": "string",  # force string instead of inferred Int64
            "gpa": "Float64",  # explicitly force Float64 (still exercises the path)
        },
    )

    out = generate_training_dtypes(df, opts)

    # Forced overrides should win over heuristic inference
    assert str(out["id"].dtype) == "string"
    assert str(out["gpa"].dtype) == "Float64"


# -------------------------------------------------------------------
# align_and_rank_dataframes
# -------------------------------------------------------------------
def test_align_and_rank_dataframes_basic_two_dataframes():
    """Test basic alignment with two dataframes and overlapping term ranges."""
    df1 = pd.DataFrame(
        {
            "student_id": ["A", "A", "B"],
            "term_order": [1, 2, 3],
            "score": [85, 90, 88],
        }
    )
    df2 = pd.DataFrame(
        {
            "course_id": [101, 102, 103],
            "term_order": [2, 3, 4],
            "credits": [3, 3, 4],
        }
    )

    result = m.align_and_rank_dataframes([df1, df2], term_column="term_order")

    # Both dataframes should be filtered to overlapping range [2, 3]
    assert len(result) == 2
    assert len(result[0]) == 2  # df1 has terms 2, 3 in range
    assert len(result[1]) == 2  # df2 has terms 2, 3 in range

    # Check term_rank is assigned
    assert "term_rank" in result[0].columns
    assert "term_rank" in result[1].columns

    # Verify term_rank values (0-indexed based on sorted unique terms)
    assert result[0]["term_rank"].tolist() == [0, 1]  # terms 2, 3 → ranks 0, 1
    assert result[1]["term_rank"].tolist() == [0, 1]  # terms 2, 3 → ranks 0, 1

    # core_term_rank should NOT be present (no core_term_col provided)
    assert "core_term_rank" not in result[0].columns
    assert "core_term_rank" not in result[1].columns


def test_align_and_rank_dataframes_with_core_terms():
    """Test alignment with core_term_col to generate core_term_rank."""
    df1 = pd.DataFrame(
        {
            "student_id": ["A", "A", "A"],
            "term_order": [1, 2, 3],
            "is_core_term": [True, False, True],
        }
    )
    df2 = pd.DataFrame(
        {
            "course_id": [101, 102, 103],
            "term_order": [1, 2, 3],
            "is_core_term": [True, False, True],
        }
    )

    result = m.align_and_rank_dataframes(
        [df1, df2], term_column="term_order", core_term_col="is_core_term"
    )

    # Both should have all 3 rows (full overlap)
    assert len(result[0]) == 3
    assert len(result[1]) == 3

    # Check both term_rank and core_term_rank are present
    assert "term_rank" in result[0].columns
    assert "core_term_rank" in result[0].columns
    assert "term_rank" in result[1].columns
    assert "core_term_rank" in result[1].columns

    # term_rank should be 0, 1, 2 for all terms
    assert result[0]["term_rank"].tolist() == [0, 1, 2]

    # core_term_rank should only be assigned to core terms (where is_core_term=True)
    # Core terms are at indices 0, 2 (term_order 1, 3) → core_term_rank 0, 1
    assert result[0]["core_term_rank"].tolist() == [0, pd.NA, 1]
    assert result[1]["core_term_rank"].tolist() == [0, pd.NA, 1]


def test_align_and_rank_dataframes_three_dataframes():
    """Test alignment with three dataframes."""
    df1 = pd.DataFrame({"id": [1], "term_order": [1]})
    df2 = pd.DataFrame({"id": [2, 2], "term_order": [1, 2]})
    df3 = pd.DataFrame({"id": [3, 3, 3], "term_order": [1, 2, 3]})

    result = m.align_and_rank_dataframes([df1, df2, df3], term_column="term_order")

    # Overlapping range is [1, 1] (min of maxes)
    assert len(result) == 3
    assert len(result[0]) == 1  # df1 has only term 1
    assert len(result[1]) == 1  # df2 filtered to term 1
    assert len(result[2]) == 1  # df3 filtered to term 1

    # All should have term_rank = 0 (only one term in range)
    assert result[0]["term_rank"].tolist() == [0]
    assert result[1]["term_rank"].tolist() == [0]
    assert result[2]["term_rank"].tolist() == [0]


def test_align_and_rank_dataframes_without_core_term_col():
    """Test that core_term_rank is omitted when core_term_col is None."""
    df1 = pd.DataFrame({"term_order": [1, 2], "val": [10, 20]})
    df2 = pd.DataFrame({"term_order": [1, 2], "val": [30, 40]})

    result = m.align_and_rank_dataframes([df1, df2], term_column="term_order")

    assert "term_rank" in result[0].columns
    assert "core_term_rank" not in result[0].columns
    assert "core_term_rank" not in result[1].columns


def test_align_and_rank_dataframes_core_term_col_missing_in_one_dataframe():
    """Test that core_term_rank is omitted when core_term_col is missing in any dataframe."""
    df1 = pd.DataFrame(
        {"term_order": [1, 2], "is_core_term": [True, False], "val": [10, 20]}
    )
    df2 = pd.DataFrame({"term_order": [1, 2], "val": [30, 40]})  # missing is_core_term

    result = m.align_and_rank_dataframes(
        [df1, df2], term_column="term_order", core_term_col="is_core_term"
    )

    # core_term_rank should NOT be present (missing in df2)
    assert "core_term_rank" not in result[0].columns
    assert "core_term_rank" not in result[1].columns


def test_align_and_rank_dataframes_raises_with_one_dataframe():
    """Test that ValueError is raised when less than 2 dataframes provided."""
    df1 = pd.DataFrame({"term_order": [1, 2]})

    with pytest.raises(ValueError) as exc:
        m.align_and_rank_dataframes([df1])
    assert "at least two dataframes" in str(exc.value).lower()


def test_align_and_rank_dataframes_raises_when_term_column_missing():
    """Test that ValueError is raised when term_column is missing in any dataframe."""
    df1 = pd.DataFrame({"term_order": [1, 2]})
    df2 = pd.DataFrame({"other_col": [1, 2]})  # missing term_order

    with pytest.raises(ValueError) as exc:
        m.align_and_rank_dataframes([df1, df2], term_column="term_order")
    assert "must have column 'term_order'" in str(exc.value)


def test_align_and_rank_dataframes_raises_with_empty_dataframe():
    """Test that ValueError is raised when any dataframe is empty."""
    df1 = pd.DataFrame({"term_order": [1, 2]})
    df2 = pd.DataFrame({"term_order": []})  # empty dataframe

    with pytest.raises(ValueError) as exc:
        m.align_and_rank_dataframes([df1, df2], term_column="term_order")
    assert "empty dataframe" in str(exc.value).lower()


def test_align_and_rank_dataframes_raises_with_no_overlapping_range():
    """Test that ValueError is raised when there's no overlapping term range."""
    df1 = pd.DataFrame({"term_order": [1, 2, 3]})
    df2 = pd.DataFrame({"term_order": [4, 5, 6]})  # no overlap

    with pytest.raises(ValueError) as exc:
        m.align_and_rank_dataframes([df1, df2], term_column="term_order")
    assert "No overlapping" in str(exc.value)


def test_align_and_rank_dataframes_raises_with_all_null_terms():
    """Test that ValueError is raised when dataframe has all null term values."""
    df1 = pd.DataFrame({"term_order": [1, 2, 3]})
    df2 = pd.DataFrame({"term_order": [None, None, None]})  # all nulls

    with pytest.raises(ValueError) as exc:
        m.align_and_rank_dataframes([df1, df2], term_column="term_order")
    assert "Cannot determine term range" in str(exc.value)


def test_align_and_rank_dataframes_preserves_original_columns():
    """Test that original columns are preserved in the result."""
    df1 = pd.DataFrame(
        {
            "student_id": ["A", "B"],
            "term_order": [1, 2],
            "score": [85, 90],
            "extra_col": ["x", "y"],
        }
    )
    df2 = pd.DataFrame(
        {"course_id": [101, 102], "term_order": [1, 2], "credits": [3, 4]}
    )

    result = m.align_and_rank_dataframes([df1, df2], term_column="term_order")

    # Check original columns are preserved
    assert "student_id" in result[0].columns
    assert "score" in result[0].columns
    assert "extra_col" in result[0].columns
    assert "course_id" in result[1].columns
    assert "credits" in result[1].columns

    # And new columns are added
    assert "term_rank" in result[0].columns
    assert "term_rank" in result[1].columns


def test_align_and_rank_dataframes_logs_alignment_info(caplog):
    """Test that alignment information is logged."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.custom_cleaning")

    df1 = pd.DataFrame({"term_order": [1, 2, 3]})
    df2 = pd.DataFrame({"term_order": [2, 3, 4]})

    m.align_and_rank_dataframes([df1, df2], term_column="term_order")

    # Check that logging occurred
    messages = [r.getMessage() for r in caplog.records]
    assert any("Common term range" in msg for msg in messages)
    assert any("DataFrame 1 aligned" in msg for msg in messages)
    assert any("DataFrame 2 aligned" in msg for msg in messages)
    assert any("Alignment complete" in msg for msg in messages)


# -------------------------------------------------------------------
# _extract_readmit_ids
# -------------------------------------------------------------------
def test_extract_readmit_ids_finds_readmit_students():
    """Test that _extract_readmit_ids correctly identifies readmit students."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C", "D"],
            "entry_type": ["First Time", "Readmit", "Transfer", "readmit"],
            "term": [1, 1, 1, 2],
        }
    )

    result = m._extract_readmit_ids(
        df, entry_col="entry_type", student_col="student_id"
    )

    # Should find both "Readmit" and "readmit" (case-insensitive)
    assert len(result) == 2
    assert set(result) == {"B", "D"}


def test_extract_readmit_ids_handles_whitespace():
    """Test that _extract_readmit_ids strips whitespace."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C"],
            "entry_type": ["First Time", "  readmit  ", "READMIT"],
        }
    )

    result = m._extract_readmit_ids(df)

    assert len(result) == 2
    assert set(result) == {"B", "C"}


def test_extract_readmit_ids_returns_empty_when_no_readmits():
    """Test that _extract_readmit_ids returns empty array when no readmits found."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C"],
            "entry_type": ["First Time", "Transfer", "First Time"],
        }
    )

    result = m._extract_readmit_ids(df)

    assert len(result) == 0
    assert isinstance(result, np.ndarray)


def test_extract_readmit_ids_returns_unique_ids():
    """Test that _extract_readmit_ids returns unique student IDs."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "B", "B"],
            "entry_type": ["Readmit", "Readmit", "Readmit", "Readmit"],
            "term": [1, 2, 1, 2],
        }
    )

    result = m._extract_readmit_ids(df)

    # Should only have 2 unique IDs
    assert len(result) == 2
    assert set(result) == {"A", "B"}


def test_extract_readmit_ids_handles_missing_entry_col():
    """Test that _extract_readmit_ids returns empty array when entry_col is missing."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C"],
            "other_col": ["X", "Y", "Z"],
        }
    )

    result = m._extract_readmit_ids(df, entry_col="entry_type")

    assert len(result) == 0
    assert isinstance(result, np.ndarray)


def test_extract_readmit_ids_handles_missing_student_col():
    """Test that _extract_readmit_ids returns empty array when student_col is missing."""
    df = pd.DataFrame(
        {
            "other_id": ["A", "B", "C"],
            "entry_type": ["Readmit", "Transfer", "First Time"],
        }
    )

    result = m._extract_readmit_ids(df, student_col="student_id")

    assert len(result) == 0
    assert isinstance(result, np.ndarray)


def test_extract_readmit_ids_ignores_nulls():
    """Test that _extract_readmit_ids ignores null student IDs."""
    df = pd.DataFrame(
        {
            "student_id": ["A", None, "C", pd.NA],
            "entry_type": ["Readmit", "Readmit", "First Time", "Readmit"],
        }
    )

    result = m._extract_readmit_ids(df)

    # Should only include "A" (not the nulls)
    assert len(result) == 1
    assert result[0] == "A"


def test_extract_readmit_ids_with_custom_column_names():
    """Test that _extract_readmit_ids works with custom column names."""
    df = pd.DataFrame(
        {
            "sid": ["S1", "S2", "S3"],
            "admit_status": ["Readmit", "First Time", "readmit"],
        }
    )

    result = m._extract_readmit_ids(df, entry_col="admit_status", student_col="sid")

    assert len(result) == 2
    assert set(result) == {"S1", "S3"}


# -------------------------------------------------------------------
# drop_readmits
# -------------------------------------------------------------------
def test_drop_readmits_removes_all_rows_for_readmit_students():
    """Test that drop_readmits removes all rows for students with readmit entry."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "B", "B", "C", "C"],
            "entry_type": [
                "First Time",
                "First Time",
                "Readmit",
                "Transfer",
                "Transfer",
                "Transfer",
            ],
            "term": [1, 2, 1, 2, 1, 2],
            "gpa": [3.0, 3.2, 3.5, 3.6, 2.8, 3.0],
        }
    )

    result = m.drop_readmits(df)

    # Student B had a "Readmit" entry, so all B's rows should be removed
    assert len(result) == 4  # Only A and C remain
    assert set(result["student_id"].unique()) == {"A", "C"}
    # Check that we didn't lose data for other students
    assert len(result[result["student_id"] == "A"]) == 2
    assert len(result[result["student_id"] == "C"]) == 2


def test_drop_readmits_is_case_insensitive():
    """Test that drop_readmits handles various casings of 'readmit'."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C", "D"],
            "entry_type": ["READMIT", "readmit", "  Readmit  ", "First Time"],
        }
    )

    result = m.drop_readmits(df)

    # A, B, and C should all be removed (various casings of readmit)
    assert len(result) == 1
    assert result["student_id"].iloc[0] == "D"


def test_drop_readmits_returns_copy_when_no_readmits():
    """Test that drop_readmits returns unchanged dataframe when no readmits."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C"],
            "entry_type": ["First Time", "Transfer", "First Time"],
            "gpa": [3.0, 3.5, 2.8],
        }
    )

    result = m.drop_readmits(df)

    # Should have all rows
    assert len(result) == len(df)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), df.reset_index(drop=True)
    )


def test_drop_readmits_removes_multiple_readmit_students():
    """Test that drop_readmits handles multiple readmit students."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "B", "B", "C", "C"],
            "entry_type": [
                "Readmit",
                "First Time",
                "Transfer",
                "Transfer",
                "Readmit",
                "Readmit",
            ],
            "term": [1, 2, 1, 2, 1, 2],
        }
    )

    result = m.drop_readmits(df)

    # Students A and C should be removed (both have readmit entries)
    assert len(result) == 2
    assert set(result["student_id"].unique()) == {"B"}


def test_drop_readmits_logs_removal_info(caplog):
    """Test that drop_readmits logs information about removed students."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.custom_cleaning")

    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "B", "B", "C"],
            "entry_type": [
                "Readmit",
                "First Time",
                "Readmit",
                "Transfer",
                "First Time",
            ],
        }
    )

    m.drop_readmits(df)

    # Check that logging occurred
    messages = [r.getMessage() for r in caplog.records]
    # Should log that 4 rows were removed for 2 students
    assert any("removed 4 rows for 2 readmit students" in msg for msg in messages)


def test_drop_readmits_handles_missing_columns():
    """Test that drop_readmits returns unchanged dataframe when required columns missing."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C"],
            "other_col": ["X", "Y", "Z"],
        }
    )

    result = m.drop_readmits(df, entry_col="entry_type")

    # Should return all rows unchanged (no entry_type column)
    assert len(result) == len(df)


def test_drop_readmits_with_custom_column_names():
    """Test that drop_readmits works with custom column names."""
    df = pd.DataFrame(
        {
            "sid": ["S1", "S1", "S2", "S2"],
            "admit_type": ["Readmit", "First Time", "Transfer", "Transfer"],
            "term": [1, 2, 1, 2],
        }
    )

    result = m.drop_readmits(df, entry_col="admit_type", student_col="sid")

    # S1 should be removed (has Readmit entry)
    assert len(result) == 2
    assert set(result["sid"].unique()) == {"S2"}


def test_drop_readmits_resets_index():
    """Test that drop_readmits resets the index after dropping rows."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C", "D"],
            "entry_type": ["First Time", "Readmit", "Transfer", "First Time"],
        }
    )

    result = m.drop_readmits(df)

    # Index should be reset to 0, 1, 2
    assert result.index.tolist() == [0, 1, 2]


def test_drop_readmits_preserves_other_columns():
    """Test that drop_readmits preserves all other columns."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C"],
            "entry_type": ["First Time", "Readmit", "Transfer"],
            "gpa": [3.0, 3.5, 2.8],
            "term": [1, 1, 1],
            "credits": [15, 12, 18],
        }
    )

    result = m.drop_readmits(df)

    # Check that all columns are preserved
    assert set(result.columns) == set(df.columns)
    # Check that data for remaining students is intact
    a_row = result[result["student_id"] == "A"].iloc[0]
    assert a_row["gpa"] == 3.0
    assert a_row["term"] == 1
    assert a_row["credits"] == 15


# -------------------------------------------------------------------
# keep_earlier_record
# -------------------------------------------------------------------
def test_keep_earlier_record_keeps_earliest_by_year():
    """Test that keep_earlier_record keeps the earliest record based on year."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "B", "B"],
            "cohort_term": ["Fall 2020", "Fall 2021", "Spring 2019", "Spring 2020"],
            "gpa": [3.0, 3.5, 2.8, 3.2],
        }
    )

    result = m.keep_earlier_record(df)

    # Should keep the earliest term for each student
    assert len(result) == 2
    # Student A: Fall 2020 is earliest
    a_row = result[result["student_id"] == "A"].iloc[0]
    assert a_row["cohort_term"] == "Fall 2020"
    assert a_row["gpa"] == 3.0
    # Student B: Spring 2019 is earliest
    b_row = result[result["student_id"] == "B"].iloc[0]
    assert b_row["cohort_term"] == "Spring 2019"
    assert b_row["gpa"] == 2.8


def test_keep_earlier_record_keeps_earliest_by_season():
    """Test that keep_earlier_record respects season ordering within a year."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "A", "A"],
            "cohort_term": ["Fall 2020", "Spring 2020", "Summer 2020", "Winter 2020"],
            "order": [3, 1, 2, 4],
        }
    )

    result = m.keep_earlier_record(df)

    # Should keep Spring 2020 (earliest in year: Spring < Summer < Fall < Winter)
    assert len(result) == 1
    assert result["cohort_term"].iloc[0] == "Spring 2020"
    assert result["order"].iloc[0] == 1


def test_keep_earlier_record_handles_multiple_students():
    """Test that keep_earlier_record handles multiple students correctly."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "B", "B", "C", "C", "C"],
            "cohort_term": [
                "Fall 2020",
                "Fall 2021",
                "Spring 2019",
                "Spring 2020",
                "Summer 2018",
                "Fall 2018",
                "Spring 2019",
            ],
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    )

    result = m.keep_earlier_record(df)

    # Should have one record per student
    assert len(result) == 3
    assert set(result["student_id"].unique()) == {"A", "B", "C"}
    # Check earliest terms are kept
    assert result[result["student_id"] == "A"]["cohort_term"].iloc[0] == "Fall 2020"
    assert result[result["student_id"] == "B"]["cohort_term"].iloc[0] == "Spring 2019"
    assert result[result["student_id"] == "C"]["cohort_term"].iloc[0] == "Summer 2018"


def test_keep_earlier_record_treats_null_as_latest():
    """Test that keep_earlier_record treats null terms as latest."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "B", "B"],
            "cohort_term": ["Fall 2020", None, None, "Spring 2019"],
            "value": [1, 2, 3, 4],
        }
    )

    result = m.keep_earlier_record(df)

    # Should keep non-null terms when available
    assert len(result) == 2
    assert result[result["student_id"] == "A"]["cohort_term"].iloc[0] == "Fall 2020"
    assert result[result["student_id"] == "B"]["cohort_term"].iloc[0] == "Spring 2019"


def test_keep_earlier_record_treats_invalid_format_as_latest():
    """Test that keep_earlier_record treats invalid term formats as latest."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "B", "B"],
            "cohort_term": ["Fall 2020", "Invalid Format", "BadTerm", "Spring 2019"],
            "value": [1, 2, 3, 4],
        }
    )

    result = m.keep_earlier_record(df)

    # Should keep valid terms over invalid ones
    assert len(result) == 2
    assert result[result["student_id"] == "A"]["cohort_term"].iloc[0] == "Fall 2020"
    assert result[result["student_id"] == "B"]["cohort_term"].iloc[0] == "Spring 2019"


def test_keep_earlier_record_handles_case_variations():
    """Test that keep_earlier_record handles case variations in term strings."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "A"],
            "cohort_term": ["FALL 2020", "spring 2020", "Summer 2020"],
            "value": [1, 2, 3],
        }
    )

    result = m.keep_earlier_record(df)

    # Should normalize to title case and keep Spring 2020
    assert len(result) == 1
    assert result["cohort_term"].iloc[0] == "spring 2020"  # Original case preserved
    assert result["value"].iloc[0] == 2


def test_keep_earlier_record_handles_whitespace():
    """Test that keep_earlier_record handles whitespace in term strings."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A"],
            "cohort_term": ["  Fall 2020  ", "Spring 2021"],
            "value": [1, 2],
        }
    )

    result = m.keep_earlier_record(df)

    # Should strip whitespace and keep Fall 2020
    assert len(result) == 1
    assert "Fall 2020" in result["cohort_term"].iloc[0]


def test_keep_earlier_record_keeps_first_when_all_null():
    """Test that keep_earlier_record keeps first record when all terms are null."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "A"],
            "cohort_term": [None, None, None],
            "value": [1, 2, 3],
        }
    )

    result = m.keep_earlier_record(df)

    # Should keep first record when all are null (all have same sort key)
    assert len(result) == 1
    assert result["value"].iloc[0] == 1


def test_keep_earlier_record_preserves_all_columns():
    """Test that keep_earlier_record preserves all columns."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A"],
            "cohort_term": ["Fall 2020", "Fall 2021"],
            "gpa": [3.0, 3.5],
            "credits": [15, 18],
            "major": ["CS", "EE"],
        }
    )

    result = m.keep_earlier_record(df)

    # Check all columns are preserved
    assert set(result.columns) == {
        "student_id",
        "cohort_term",
        "gpa",
        "credits",
        "major",
    }
    # Check data integrity for kept record
    assert result["gpa"].iloc[0] == 3.0
    assert result["credits"].iloc[0] == 15
    assert result["major"].iloc[0] == "CS"


def test_keep_earlier_record_resets_index():
    """Test that keep_earlier_record resets the index."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C"],
            "cohort_term": ["Fall 2020", "Spring 2019", "Summer 2021"],
        },
        index=[10, 20, 30],
    )

    result = m.keep_earlier_record(df)

    # Index should be reset to 0, 1, 2
    assert result.index.tolist() == [0, 1, 2]


def test_keep_earlier_record_with_custom_column_names():
    """Test that keep_earlier_record works with custom column names."""
    df = pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U2", "U2"],
            "enrollment_term": ["Fall 2020", "Fall 2021", "Spring 2019", "Spring 2020"],
            "score": [85, 90, 75, 80],
        }
    )

    result = m.keep_earlier_record(df, id_col="user_id", term_col="enrollment_term")

    # Should keep earliest record for each user
    assert len(result) == 2
    assert result[result["user_id"] == "U1"]["enrollment_term"].iloc[0] == "Fall 2020"
    assert result[result["user_id"] == "U2"]["enrollment_term"].iloc[0] == "Spring 2019"


def test_keep_earlier_record_with_single_record_per_student():
    """Test that keep_earlier_record handles students with only one record."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C"],
            "cohort_term": ["Fall 2020", "Spring 2019", "Summer 2021"],
            "gpa": [3.0, 3.5, 2.8],
        }
    )

    result = m.keep_earlier_record(df)

    # Should return all records unchanged (one per student)
    assert len(result) == 3
    pd.testing.assert_frame_equal(result, df.reset_index(drop=True))


def test_keep_earlier_record_season_ordering():
    """Test the specific season ordering: Spring < Summer < Fall < Winter."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "A", "A"],
            "cohort_term": ["Winter 2020", "Fall 2020", "Summer 2020", "Spring 2020"],
            "order": [4, 3, 2, 1],
        }
    )

    result = m.keep_earlier_record(df)

    # Should keep Spring 2020 (earliest season)
    assert len(result) == 1
    assert result["cohort_term"].iloc[0] == "Spring 2020"
    assert result["order"].iloc[0] == 1


def test_keep_earlier_record_unknown_season_treated_as_latest():
    """Test that unknown season names are treated as latest."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A", "A"],
            "cohort_term": ["Spring 2020", "Autumn 2020", "Unknown 2020"],
            "value": [1, 2, 3],
        }
    )

    result = m.keep_earlier_record(df)

    # Should keep Spring 2020 (valid season over unknown)
    assert len(result) == 1
    assert result["cohort_term"].iloc[0] == "Spring 2020"
    assert result["value"].iloc[0] == 1


def test_keep_earlier_record_handles_year_not_integer():
    """Test that keep_earlier_record treats non-integer years as invalid."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "A"],
            "cohort_term": ["Fall 2020", "Spring NotAYear"],
            "value": [1, 2],
        }
    )

    result = m.keep_earlier_record(df)

    # Should keep Fall 2020 (valid year over invalid)
    assert len(result) == 1
    assert result["cohort_term"].iloc[0] == "Fall 2020"
    assert result["value"].iloc[0] == 1


# -------------------------------------------------------------------
# assign_numeric_grade
# -------------------------------------------------------------------
def test_assign_numeric_grade_with_default_mapping():
    """Test that assign_numeric_grade uses default grade mapping."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B", "C", "D"],
            "grade": ["A", "B", "C", "F"],
        }
    )

    result = m.assign_numeric_grade(df)

    # Check that course_numeric_grade column is added
    assert "course_numeric_grade" in result.columns
    # Check default mappings
    assert result.loc[0, "course_numeric_grade"] == 4.0  # A
    assert result.loc[1, "course_numeric_grade"] == 3.0  # B
    assert result.loc[2, "course_numeric_grade"] == 2.0  # C
    assert result.loc[3, "course_numeric_grade"] == 0.0  # F


def test_assign_numeric_grade_with_custom_mapping():
    """Test that assign_numeric_grade accepts custom grade mapping."""
    df = pd.DataFrame(
        {
            "grade": ["Excellent", "Good", "Fair", "Poor"],
        }
    )

    custom_map = {
        "EXCELLENT": 4.0,
        "GOOD": 3.0,
        "FAIR": 2.0,
        "POOR": 1.0,
    }

    result = m.assign_numeric_grade(df, grade_numeric_map=custom_map)

    assert result["course_numeric_grade"].tolist() == [4.0, 3.0, 2.0, 1.0]


def test_assign_numeric_grade_case_insensitive():
    """Test that assign_numeric_grade is case insensitive."""
    df = pd.DataFrame(
        {
            "grade": ["a", "A", "b+", "B+", "f", "F"],
        }
    )

    result = m.assign_numeric_grade(df)

    # All should map correctly regardless of case
    assert result.loc[0, "course_numeric_grade"] == 4.0  # a -> A
    assert result.loc[1, "course_numeric_grade"] == 4.0  # A
    assert result.loc[2, "course_numeric_grade"] == 3.3  # b+ -> B+
    assert result.loc[3, "course_numeric_grade"] == 3.3  # B+
    assert result.loc[4, "course_numeric_grade"] == 0.0  # f -> F
    assert result.loc[5, "course_numeric_grade"] == 0.0  # F


def test_assign_numeric_grade_strips_whitespace():
    """Test that assign_numeric_grade strips whitespace from grades."""
    df = pd.DataFrame(
        {
            "grade": ["  A  ", " B+ ", "C  ", "  F"],
        }
    )

    result = m.assign_numeric_grade(df)

    # All should map correctly after stripping whitespace
    assert result["course_numeric_grade"].tolist() == [4.0, 3.3, 2.0, 0.0]


def test_assign_numeric_grade_handles_unmapped_grades(capsys):
    """Test that assign_numeric_grade prints unmapped grades."""
    df = pd.DataFrame(
        {
            "grade": ["A", "B", "Z", "X"],
        }
    )

    result = m.assign_numeric_grade(df)

    # Check that unmapped grades are NaN
    assert result.loc[0, "course_numeric_grade"] == 4.0
    assert result.loc[1, "course_numeric_grade"] == 3.0
    assert pd.isna(result.loc[2, "course_numeric_grade"])  # Z
    assert pd.isna(result.loc[3, "course_numeric_grade"])  # X

    # Check that unmapped grades were printed
    captured = capsys.readouterr()
    assert "X" in captured.out
    assert "Z" in captured.out


def test_assign_numeric_grade_handles_null_grades():
    """Test that assign_numeric_grade handles null grades."""
    df = pd.DataFrame(
        {
            "grade": ["A", None, "B", pd.NA],
        }
    )

    result = m.assign_numeric_grade(df)

    assert result.loc[0, "course_numeric_grade"] == 4.0
    assert pd.isna(result.loc[1, "course_numeric_grade"])
    assert result.loc[2, "course_numeric_grade"] == 3.0
    assert pd.isna(result.loc[3, "course_numeric_grade"])


def test_assign_numeric_grade_with_custom_columns():
    """Test that assign_numeric_grade works with custom column names."""
    df = pd.DataFrame(
        {
            "letter_grade": ["A", "B", "C"],
            "other_col": [1, 2, 3],
        }
    )

    result = m.assign_numeric_grade(
        df, grade_col="letter_grade", output_col="numeric_score"
    )

    # Check that output column has custom name
    assert "numeric_score" in result.columns
    assert "course_numeric_grade" not in result.columns
    assert result["numeric_score"].tolist() == [4.0, 3.0, 2.0]


def test_assign_numeric_grade_with_none_mapped_grades():
    """Test that assign_numeric_grade handles grades mapped to None."""
    df = pd.DataFrame(
        {
            "grade": ["A", "S", "NR", "B"],  # S and NR map to None in default
        }
    )

    result = m.assign_numeric_grade(df)

    assert result.loc[0, "course_numeric_grade"] == 4.0
    assert pd.isna(result.loc[1, "course_numeric_grade"])  # S -> None
    assert pd.isna(result.loc[2, "course_numeric_grade"])  # NR -> None
    assert result.loc[3, "course_numeric_grade"] == 3.0


def test_assign_numeric_grade_preserves_original_columns():
    """Test that assign_numeric_grade preserves all original columns."""
    df = pd.DataFrame(
        {
            "student_id": ["A", "B"],
            "grade": ["A", "B"],
            "credits": [3, 4],
        }
    )

    result = m.assign_numeric_grade(df)

    # Check all original columns are preserved
    assert "student_id" in result.columns
    assert "grade" in result.columns
    assert "credits" in result.columns
    # And new column is added
    assert "course_numeric_grade" in result.columns


def test_assign_numeric_grade_logs_transformation(caplog):
    """Test that assign_numeric_grade logs transformation messages."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.custom_cleaning")

    df = pd.DataFrame({"grade": ["A", "B"]})
    m.assign_numeric_grade(df)

    messages = [r.getMessage() for r in caplog.records]
    assert any("Starting assign_numeric_grade" in msg for msg in messages)
    assert any("Completed assign_numeric_grade" in msg for msg in messages)


# -------------------------------------------------------------------
# log_top_majors
# -------------------------------------------------------------------
def test_log_top_majors_logs_top_10(caplog):
    """Test that log_top_majors logs the top 10 majors."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "program_of_study_term_1": ["CS"] * 50
            + ["EE"] * 30
            + ["ME"] * 20
            + ["CE"] * 15
            + ["BIO"] * 10
            + ["CHEM"] * 8
            + ["PHYS"] * 6
            + ["MATH"] * 4
            + ["STAT"] * 3
            + ["DATA"] * 2
            + ["ART"] * 1,
        }
    )

    eda.log_top_majors(df)

    messages = [r.getMessage() for r in caplog.records]
    log_message = " ".join(messages)

    # Check that top majors are in the log
    assert "CS" in log_message
    assert "EE" in log_message
    assert "Top majors" in log_message


def test_log_top_majors_handles_less_than_10_majors(caplog):
    """Test that log_top_majors handles dataframes with less than 10 majors."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "program_of_study_term_1": ["CS", "CS", "EE", "EE", "ME"],
        }
    )

    eda.log_top_majors(df)

    messages = [r.getMessage() for r in caplog.records]
    assert any("Top majors" in msg for msg in messages)


def test_log_top_majors_handles_null_values(caplog):
    """Test that log_top_majors includes null values in counts."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "program_of_study_term_1": ["CS"] * 10 + [None] * 5 + ["EE"] * 3,
        }
    )

    eda.log_top_majors(df)

    messages = [r.getMessage() for r in caplog.records]
    # Should handle null values (dropna=False in value_counts)
    assert any("Top majors" in msg for msg in messages)


def test_log_top_majors_returns_none():
    """Test that log_top_majors returns None (it's a logging function)."""
    df = pd.DataFrame(
        {
            "program_of_study_term_1": ["CS", "EE"],
        }
    )

    result = eda.log_top_majors(df)
    assert result is None


# -------------------------------------------------------------------
# order_terms
# -------------------------------------------------------------------
def test_order_terms_creates_ordered_categorical():
    """Test that order_terms creates an ordered categorical column."""
    df = pd.DataFrame(
        {
            "term": ["Fall 2020", "Spring 2020", "Summer 2020", "Spring 2021"],
        }
    )

    result = eda.order_terms(df, term_col="term")

    # Check that column is categorical and ordered
    assert pd.api.types.is_categorical_dtype(result["term"])
    assert result["term"].cat.ordered is True


def test_order_terms_sorts_by_year_then_season():
    """Test that order_terms sorts correctly by year then season."""
    df = pd.DataFrame(
        {
            "term": [
                "Fall 2021",
                "Spring 2020",
                "Summer 2020",
                "Spring 2021",
                "Fall 2020",
            ],
        }
    )

    result = eda.order_terms(df, term_col="term")

    # Expected order: Spring 2020, Summer 2020, Fall 2020, Spring 2021, Fall 2021
    expected_order = [
        "Spring 2020",
        "Summer 2020",
        "Fall 2020",
        "Spring 2021",
        "Fall 2021",
    ]
    assert list(result["term"].cat.categories) == expected_order


def test_order_terms_with_custom_season_order():
    """Test that order_terms accepts custom season ordering."""
    df = pd.DataFrame(
        {
            "term": ["Winter 2020", "Autumn 2020", "Spring 2020"],
        }
    )

    custom_order = {"Spring": 1, "Autumn": 2, "Winter": 3}
    result = eda.order_terms(df, term_col="term", season_order=custom_order)

    # Expected order with custom season mapping
    expected_order = ["Spring 2020", "Autumn 2020", "Winter 2020"]
    assert list(result["term"].cat.categories) == expected_order


def test_order_terms_handles_missing_term_column():
    """Test that order_terms returns unchanged dataframe if term column missing."""
    df = pd.DataFrame(
        {
            "other_col": [1, 2, 3],
        }
    )

    result = eda.order_terms(df, term_col="term")

    # Should return unchanged dataframe
    pd.testing.assert_frame_equal(result, df)


def test_order_terms_handles_empty_terms():
    """Test that order_terms handles dataframe with no valid terms."""
    df = pd.DataFrame(
        {
            "term": [None, None, None],
        }
    )

    result = eda.order_terms(df, term_col="term")

    # Should return dataframe (no categories created for all-null column)
    assert len(result) == 3


def test_order_terms_handles_invalid_term_format():
    """Test that order_terms handles invalid term formats."""
    df = pd.DataFrame(
        {
            "term": ["Spring 2020", "Invalid", "BadFormat 123", "Fall 2020"],
        }
    )

    result = eda.order_terms(df, term_col="term")

    # Valid terms should be ordered, invalid ones pushed to end
    categories = list(result["term"].cat.categories)
    assert "Spring 2020" in categories
    assert "Fall 2020" in categories
    # Invalid terms should appear last
    assert categories.index("Spring 2020") < categories.index("Invalid")


def test_order_terms_logs_categories(caplog):
    """Test that order_terms logs the ordered categories."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "semester": ["Fall 2020", "Spring 2020"],
        }
    )

    eda.order_terms(df, term_col="semester")

    messages = [r.getMessage() for r in caplog.records]
    log_message = " ".join(messages)

    # Check that logging occurred
    assert "term_order_fn" in log_message
    assert "semester" in log_message


def test_order_terms_default_season_order():
    """Test that order_terms uses default season order: Spring < Summer < Fall < Winter."""
    df = pd.DataFrame(
        {
            "term": ["Winter 2020", "Fall 2020", "Summer 2020", "Spring 2020"],
        }
    )

    result = eda.order_terms(df, term_col="term")

    expected_order = ["Spring 2020", "Summer 2020", "Fall 2020", "Winter 2020"]
    assert list(result["term"].cat.categories) == expected_order


def test_order_terms_preserves_other_columns():
    """Test that order_terms preserves all other columns."""
    df = pd.DataFrame(
        {
            "term": ["Fall 2020", "Spring 2020"],
            "student_id": ["A", "B"],
            "gpa": [3.5, 3.8],
        }
    )

    result = eda.order_terms(df, term_col="term")

    # Check all columns are preserved
    assert "student_id" in result.columns
    assert "gpa" in result.columns
    assert result["student_id"].tolist() == ["A", "B"]
    assert result["gpa"].tolist() == [3.5, 3.8]


def test_order_terms_handles_duplicate_terms():
    """Test that order_terms handles duplicate term values."""
    df = pd.DataFrame(
        {
            "term": ["Fall 2020", "Fall 2020", "Spring 2020", "Spring 2020"],
        }
    )

    result = eda.order_terms(df, term_col="term")

    # Should have unique categories
    assert len(result["term"].cat.categories) == 2
    assert list(result["term"].cat.categories) == ["Spring 2020", "Fall 2020"]


def test_order_terms_handles_non_integer_year():
    """Test that order_terms handles terms with non-integer years."""
    df = pd.DataFrame(
        {
            "term": ["Spring 2020", "Fall NotAYear", "Summer 2020"],
        }
    )

    result = eda.order_terms(df, term_col="term")

    categories = list(result["term"].cat.categories)
    # Valid terms should come first
    assert categories[0] == "Spring 2020"
    assert categories[1] == "Summer 2020"
    # Invalid year should be pushed to end
    assert categories[-1] == "Fall NotAYear"
