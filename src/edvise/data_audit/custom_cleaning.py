"""
Summary:
  • Normalize headers & clean DataFrames
  • Geneate robust training-time dtypes with thresholds (avoid train–infer skew)
  • Freeze per-dataset schemas and assemble a multi-dataset **preprocess schema**
  • Enforce schemas at inference
  • Save/load the schema

Notes:
  - Uses pandas nullable dtypes: Int64, Float64, boolean, string.
        - This is so we can handle nulls applicable to any schema or any number of datasets.
        - This differs from PDP, where we use sklearn dtypes (non-nullable), which is due to
          the fact that the PDP schema & pipeline are narrow and well-defined in scope.
        - Unfortunately, this differs greatly from custom schools.
  - Uses schema enforcement at inference.
        - This is critical for data reliability and to ensure visibility with training-inference skew.
        - The enforcement is meant to be: strict with dtypes, primary keys and extra datasets, and
          lenient with dataset shape extra columns (drop with a warning), or missing columns (filled with NA).
  - Date parsing uses coercion + type-confidence thresholds and minimum non-null counts
  - No categorical vocabulary capture or version fields
"""

import typing as t
import logging
import warnings
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from edvise.utils.data_cleaning import convert_to_snake_case

LOGGER = logging.getLogger(__name__)


# ---------------------------
# Create datasets object for processing
# ---------------------------
def create_datasets(
    df: pd.DataFrame,
    bronze: t.Mapping[str, t.Any] | t.Any,
    *,
    include_empty: bool = False,
) -> dict:
    """
    Normalize a df + bronze config into a single entry.
    - Maps: drop_cols -> "drop columns", non_null_cols -> "non-null columns", primary_keys -> "unique keys"
    - Includes a field only if present and non-empty (unless include_empty=True).
    - Works whether bronze is a dict or an object with attributes.
    """
    out = {"data": df}
    mapping = {
        "drop_cols": "drop columns",
        "non_null_cols": "non-null columns",
        "primary_keys": "unique keys",
    }

    def _get(obj, attr):
        return obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)

    for attr, out_key in mapping.items():
        val = _get(bronze, attr)
        if include_empty or val not in (None, [], {}, ()):
            out[out_key] = val
    return out


# ---------------------------
# Column normalization
# ---------------------------
def normalize_columns(cols: t.Iterable[str]) -> tuple[pd.Index, dict[str, list[str]]]:
    """
    Normalize column names to snake_case using `convert_to_snake_case`,
    and return:
      - normalized pd.Index
      - mapping {normalized_name: [original_names...]} to detect collisions.
    """
    # Ensure we are working with strings
    orig = [str(c) for c in cols]

    # Reuse single-string snake-case logic for every column
    norm_list = [convert_to_snake_case(c) for c in orig]
    norm = pd.Index(norm_list)

    # Build mapping: normalized -> list of original names
    mapping: dict[str, list[str]] = {}
    for o, n in zip(orig, norm):
        mapping.setdefault(n, []).append(o)

    return norm, mapping


# ---------------------------
# Robust training-time dtype generation (nullable dtypes only)
# ---------------------------
@dataclass
class DtypeGenerationOptions:
    date_formats: tuple[str, ...] = ("%m/%d/%Y",)
    # Accept a type at training-time if at least this fraction of values
    # can be successfully coerced to that type.
    dtype_confidence_threshold: float = 0.75
    # Also require at least this many non-null values to trust the generated dtype.
    min_non_null: int = 10
    boolean_map: dict[str, bool] = field(
        default_factory=lambda: {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
        }
    )


def generate_column_training_dtype(
    series: pd.Series, opts: DtypeGenerationOptions | None = None
) -> pd.Series:
    """
    Generate a training-time dtype for a column using coercion & confidence thresholds.
    Returns a converted Series using pandas nullable dtypes.

    NOTE: This is intended for training-time schema learning only.
    At inference-time, use `enforce_schema` with a frozen schema instead of
    calling this again (to avoid train–inference skew). The dtypes could also
    be inferred differently if columns are missing more (or less) values at
    inference vs. training time.
    """
    if not isinstance(series, pd.Series):
        return series

    if opts is None:
        opts = DtypeGenerationOptions()

    s = series.copy()

    # --- Short-circuit for already-typed columns --------------------------

    # Already datetime-like → keep as datetime (no thresholds needed)
    if ptypes.is_datetime64_any_dtype(s) or ptypes.is_datetime64tz_dtype(s):
        return s

    # Already boolean → normalize to pandas nullable boolean
    if ptypes.is_bool_dtype(s):
        return s.astype("boolean")

    # Already numeric → normalize to nullable numeric
    if ptypes.is_integer_dtype(s) or ptypes.is_float_dtype(s):
        non_na = s.dropna()
        if len(non_na) and np.all(np.isclose(non_na % 1, 0)):
            return s.astype("Int64")
        return s.astype("Float64")

    # --- From here on, we assume "string-ish" / object data ---------------

    def _enough_non_null(mask: pd.Series | np.ndarray) -> bool:
        # mask is boolean or notna() result
        count = int(mask.sum())
        frac = float(count) / len(s) if len(s) else 0.0
        return count >= opts.min_non_null and frac >= opts.dtype_confidence_threshold

    # Try declared date formats with coercion
    for fmt in opts.date_formats:
        dt = pd.to_datetime(s, format=fmt, errors="coerce")
        mask = dt.notna()
        if _enough_non_null(mask):
            return dt

    # Try inferred datetime (quietly suppress the "could not infer format" warning)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually",
            category=UserWarning,
        )
        dt_inf = pd.to_datetime(s, errors="coerce")
    if _enough_non_null(dt_inf.notna()):
        return dt_inf

    # Try numeric with coercion (nullable pandas dtypes)
    num = pd.to_numeric(s, errors="coerce")
    if _enough_non_null(num.notna()):
        non_na = num.dropna()
        if len(non_na) and np.all(np.isclose(non_na % 1, 0)):
            return num.astype("Int64")
        return num.astype("Float64")

    # Try boolean via map (nullable boolean)
    lower = s.astype("string").str.strip().str.lower()
    uniq = set(lower.dropna().unique())
    if uniq and uniq.issubset(set(opts.boolean_map.keys())):
        return lower.map(opts.boolean_map).astype("boolean")

    # Fallback: nullable string
    return s.astype("string")


def generate_training_dtypes(
    df: pd.DataFrame, opts: DtypeGenerationOptions | None = None
) -> pd.DataFrame:
    """
    Generate training-time dtypes for an entire DataFrame. This is needed since
    many of our custom schools give us CSV files, which do not save dtypes (everything
    is usually nullable string type).

    NOTE: Use only on training data. At inference time, rely on `enforce_schema`
    using a schema frozen from the trained data.
    """
    opts = opts or DtypeGenerationOptions()
    out = df.copy()
    for col in out.columns:
        try:
            out[col] = generate_column_training_dtype(out[col], opts=opts)
        except Exception as e:
            LOGGER.warning("Failed to infer dtype for column %s: %s", col, e)
    return out


# ---------------------------
# Cleaning
# ---------------------------
@dataclass
class CleanSpec:
    drop_columns: list[str] | None = None
    non_null_columns: list[str] | None = None
    unique_keys: list[str] | None = None
    # Optional custom hooks
    student_id_alias: str | None = None
    dedupe_fn: t.Callable[[pd.DataFrame], pd.DataFrame] | None = None
    term_order_fn: t.Callable[[pd.DataFrame, str], pd.DataFrame] | None = None
    term_column: str = "term"

    # For schema mapping
    _orig_cols_: list[str] | None = None


def clean_dataset(
    df: pd.DataFrame,
    spec: dict | CleanSpec,
    dataset_name: str = "",
    inference_opts: DtypeGenerationOptions | None = None,
    enforce_uniqueness: bool = True,
    generate_dtypes: bool = True,
) -> pd.DataFrame:
    """
    End-to-end cleaner with robust training-time dtype generation and consistent policies.

    Typical pattern:
      - TRAINING:
          clean_dataset(..., generate_dtypes=True)
          -> build_preprocess_schema(...)
      - INFERENCE:
          clean_dataset(..., generate_dtypes=False)  # if you still want cleaning hooks
          then enforce_schema(...) using the frozen schema.

    `generate_dtypes=False` is useful at inference to avoid re-generating dtypes
    and introducing train–inference skew; instead, `enforce_schema` should dictate
    the final dtypes.
    """
    inference_opts = inference_opts or DtypeGenerationOptions()
    if isinstance(spec, dict):
        spec = CleanSpec(
            drop_columns=(spec.get("drop columns") or spec.get("drop_columns")),
            non_null_columns=(
                spec.get("non-null columns") or spec.get("non_null_columns")
            ),
            unique_keys=spec.get("unique keys") or spec.get("unique_keys"),
            dedupe_fn=spec.get("dedupe_fn"),
            term_order_fn=spec.get("term_order_fn"),
            term_column=spec.get("term_column", "term"),
            _orig_cols_=spec.get("_orig_cols_", list(df.columns)),
        )

    g = df.copy()

    # 1) normalize column names
    spec._orig_cols_ = list(g.columns)
    norm, mapping = normalize_columns(g.columns)
    collisions = {n: srcs for n, srcs in mapping.items() if len(srcs) > 1}
    if collisions:
        raise ValueError(
            f"{dataset_name} - Column-name collisions after normalization: {collisions}"
        )
    g.columns = norm

    # 2) canonical student_id rename
    alias = spec.student_id_alias or "student_id_randomized_datakind"

    if alias in g.columns:
        # If alias exists and student_id does NOT already exist
        if alias != "student_id" and "student_id" not in g.columns:
            LOGGER.info(
                "%s - Renaming student ID alias '%s' -> 'student_id'",
                dataset_name,
                alias,
            )
            g = g.rename(columns={alias: "student_id"})

            # Keep primary-key spec in sync
            if spec.unique_keys:
                spec.unique_keys = [
                    "student_id" if k == alias else k for k in spec.unique_keys
                ]

        # If both alias and student_id exist → ambiguous
        elif alias != "student_id" and "student_id" in g.columns:
            LOGGER.warning(
                "%s - Found both 'student_id' and alias '%s'; leaving both unchanged.",
                dataset_name,
                alias,
            )

    # 3) normalize null tokens & whitespace
    g = g.replace("(Blank)", np.nan)
    obj_cols = g.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols):
        g[obj_cols] = g[obj_cols].apply(
            lambda s: s.replace(r"^\s*$", pd.NA, regex=True)
        )

    # 4) drop requested columns (never drop student_id)
    to_drop = set(spec.drop_columns or [])
    to_drop.discard("student_id")
    if to_drop:
        g = g.drop(columns=list(to_drop), errors="ignore")
        LOGGER.info(
            "%s - Dropping columns: %s | shape=%s",
            dataset_name,
            sorted(to_drop),
            g.shape,
        )

    # 5) drop rows requiring non-nulls
    if spec.non_null_columns:
        g = g.dropna(subset=list(spec.non_null_columns), how="any")
        LOGGER.info(
            "%s - Dropped rows missing %s | shape=%s",
            dataset_name,
            spec.non_null_columns,
            g.shape,
        )

    # 6) generate dtypes at training-time
    if generate_dtypes:
        g = generate_training_dtypes(g, opts=inference_opts)
    if "student_id" in g.columns:
        g["student_id"] = g["student_id"].astype("string")

    # 7) optional dataset-specific dedupe hook (pre-key)
    if spec.dedupe_fn and callable(spec.dedupe_fn):
        g = spec.dedupe_fn(g)

    # 8) drop full row duplicates
    before = len(g)
    g = g.drop_duplicates().reset_index(drop=True)
    LOGGER.info(
        "%s - Removed full row duplicates: %d removed | shape=%s",
        dataset_name,
        before - len(g),
        g.shape,
    )

    if enforce_uniqueness:
        # 9) deduplicate rows based on primary keys
        before = len(g)
        if spec.unique_keys:
            g = g.drop_duplicates(subset=spec.unique_keys, keep="first").reset_index(
                drop=True
            )
        LOGGER.info(
            "%s - Deduplicated rows based on primary keys %s | %d removed | shape=%s",
            dataset_name,
            spec.unique_keys,
            before - len(g),
            g.shape,
        )

        # 10) enforce uniqueness by primary keys
        if spec.unique_keys:
            dups = g.duplicated(subset=spec.unique_keys)
            if dups.any():
                raise ValueError(
                    f"{dataset_name} - Duplicate rows detected on primary keys {spec.unique_keys}. Count={int(dups.sum())}"
                )

    # 11) optional term order
    if (
        spec.term_order_fn
        and callable(spec.term_order_fn)
        and spec.term_column in g.columns
    ):
        LOGGER.info(
            "%s - Applying term order function to column '%s'",
            dataset_name,
            spec.term_column,
        )
        g = spec.term_order_fn(g, spec.term_column)

    return g


def clean_all_datasets_map(
    datasets: t.Dict[str, t.Dict],
    enforce_uniqueness: bool = True,
    generate_dtypes: bool = True,
) -> t.Dict[str, pd.DataFrame]:
    """
    Clean each dataset bundle and return {name: cleaned_df}.
    Each bundle must be shaped like {"data": df, "...spec keys..."}.

    `generate_dtypes` is passed through to `clean_dataset`, so you can also
    reuse this at inference-time by setting it to False and then applying
    `enforce_preprocess_schema` / `enforce_schema`.
    """
    cleaned: dict[str, pd.DataFrame] = {}
    for key, bundle in datasets.items():
        if "data" not in bundle:
            raise KeyError(f"Dataset '{key}' is missing required 'data' key")
        LOGGER.info(
            "%s - Starting cleaning; shape=%s",
            key,
            getattr(bundle["data"], "shape", None),
        )
        spec = {k: v for k, v in bundle.items() if k != "data"}
        cleaned[key] = clean_dataset(
            df=bundle["data"],
            dataset_name=key,
            spec=spec,
            enforce_uniqueness=enforce_uniqueness,
            generate_dtypes=generate_dtypes,
        )
        LOGGER.info("%s - Finished cleaning; final shape=%s", key, cleaned[key].shape)
    return cleaned


# ---------------------------
# Schema freezing & enforcing
# ---------------------------
@dataclass
class SchemaFreezeOptions:
    include_column_order_hash: bool = True


def _hash_list(values: list[str]) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()


def freeze_schema(
    df: pd.DataFrame, spec: dict[str, t.Any], opts: SchemaFreezeOptions | None = None
) -> dict[str, t.Any]:
    """Freeze names, dtypes, non-null policy, unique keys. No vocab, no version."""
    opts = opts or SchemaFreezeOptions()
    schema = {
        "normalized_columns": {
            o: n for o, n in zip(spec.get("_orig_cols_", list(df.columns)), df.columns)
        },
        "dtypes": {
            c: str(df[c].dtype) for c in df.columns
        },  # expect Int64, Float64, boolean, string, datetime64[ns]
        "non_null_columns": list(spec.get("non-null columns", []) or []),
        "unique_keys": list(spec.get("unique keys", []) or []),
        "null_tokens": ["(Blank)"],
        "boolean_map": {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
        },
    }
    if opts.include_column_order_hash:
        schema["column_order_hash"] = _hash_list(list(df.columns))
    return schema


def enforce_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    g = df.copy()

    # 1) normalize columns identically
    g.columns = (
        pd.Index(g.columns)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/\-]+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    # 2) align to expected columns
    expected = list(schema["dtypes"].keys())
    for c in expected:
        if c not in g.columns:
            g[c] = pd.NA
    extra = [c for c in g.columns if c not in expected]
    if extra:
        LOGGER.warning("Unexpected columns at inference: %s", extra)
    g = g[expected]

    # 3) cast to frozen (nullable) dtypes
    bmap = schema.get(
        "boolean_map",
        {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False},
    )
    for c, dt in schema["dtypes"].items():
        try:
            if dt.startswith("datetime64"):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Could not infer format, so each element will be parsed individually",
                        category=UserWarning,
                    )
                    g[c] = pd.to_datetime(g[c], errors="coerce")
            elif dt == "Int64":
                g[c] = pd.to_numeric(g[c], errors="coerce").astype("Int64")
            elif dt == "Float64":
                g[c] = pd.to_numeric(g[c], errors="coerce").astype("Float64")
            elif dt == "boolean":
                g[c] = (
                    g[c]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                    .map(bmap)
                    .astype("boolean")
                )
            elif dt == "string":
                g[c] = g[c].astype("string")
            else:
                # fallback: attempt exact cast (kept for forward compatibility)
                g[c] = g[c].astype(dt)
        except Exception as e:
            raise ValueError(f"Failed to cast column {c} to {dt}: {e}")

    # 4) enforce non-nulls
    nn = schema.get("non_null_columns", [])
    if nn:
        before = len(g)
        g = g.dropna(subset=nn, how="any")
        LOGGER.info("Enforce non-nulls %s: dropped %d rows", nn, before - len(g))

    # 5) enforce key uniqueness
    keys = schema.get("unique_keys")
    if keys:
        dups = g.duplicated(subset=keys)
        if dups.any():
            raise ValueError(
                f"Duplicate rows on unique keys {keys} at inference. Count={int(dups.sum())}"
            )

    return g


# ---------------------------
# Multi-dataset preprocess schema
# ---------------------------
@dataclass
class SchemaContractMeta:
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    null_tokens: list[str] = field(default_factory=lambda: ["(Blank)"])


def build_preprocess_schema(
    cleaned_map: dict[str, pd.DataFrame],
    specs: dict[str, dict[str, t.Any] | CleanSpec],
    *,
    meta: SchemaContractMeta | None = None,
    freeze_opts: SchemaFreezeOptions | None = None,
) -> dict:
    meta = meta or SchemaContractMeta()
    freeze_opts = freeze_opts or SchemaFreezeOptions()
    datasets: dict[str, t.Any] = {}

    for name, df in cleaned_map.items():
        raw_spec = specs[name]

        # Always convert raw_spec into a dict[str, Any] for freeze_schema
        if isinstance(raw_spec, CleanSpec):
            spec_dict: dict[str, t.Any] = {
                # Only the keys freeze_schema actually cares about
                "non-null columns": raw_spec.non_null_columns,
                "unique keys": raw_spec.unique_keys,
                "_orig_cols_": raw_spec._orig_cols_ or list(df.columns),
            }
        else:
            # raw_spec is already a dict[str, Any]
            spec_dict = dict(raw_spec)
            spec_dict.setdefault("_orig_cols_", list(df.columns))

        datasets[name] = freeze_schema(df, spec_dict, opts=freeze_opts)

    return {
        "created_at": meta.created_at,
        "null_tokens": meta.null_tokens,
        "datasets": datasets,
    }


def enforce_preprocess_schema(
    raw_map: dict[str, pd.DataFrame], preprocess_schema: dict
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, df in raw_map.items():
        if name not in preprocess_schema["datasets"]:
            raise KeyError(f"Dataset '{name}' is not present in preprocess_schema")
        out[name] = enforce_schema(df, preprocess_schema["datasets"][name])
    return out


# ---------------------------
# Serialization helpers
# ---------------------------
def save_preprocess_schema(preprocess_schema: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preprocess_schema, f, indent=2, ensure_ascii=False)


def load_preprocess_schema(path: str) -> dict[str, t.Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return t.cast(dict[str, t.Any], data)


__all__ = [
    "DtypeGenerationOptions",
    "generate_column_training_dtype",
    "generate_training_dtypes",
    "CleanSpec",
    "clean_dataset",
    "SchemaFreezeOptions",
    "freeze_schema",
    "enforce_schema",
    "SchemaContractMeta",
    "build_preprocess_schema",
    "enforce_preprocess_schema",
    "save_preprocess_schema",
    "load_preprocess_schema",
    "normalize_columns",
    "create_datasets",
    "clean_all_datasets_map",
]
