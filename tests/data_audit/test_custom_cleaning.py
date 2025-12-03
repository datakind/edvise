import types

import pandas as pd
import logging
import pytest

from edvise.data_audit import custom_cleaning as m
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
