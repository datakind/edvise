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
from edvise.feature_generation.term import add_term_order
from edvise.configs.custom import CustomProjectConfig, CleaningConfig

LOGGER = logging.getLogger(__name__)


# Type aliases for clarity
TermOrderFn = t.Callable[[pd.DataFrame, str], pd.DataFrame]
DedupeFn = t.Callable[[pd.DataFrame], pd.DataFrame]


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


def build_datasets_from_bronze(
    cfg: CustomProjectConfig,
    df_map: t.Dict[str, t.Tuple[str, pd.DataFrame]],
) -> dict[str, dict]:
    return {
        name: create_datasets(
            df,
            cfg.datasets.bronze[raw_key],
        )
        for name, (raw_key, df) in df_map.items()
    }


def attach_cleaning_hooks(
    datasets: dict[str, dict],
    cleaning_cfg: t.Optional[CleaningConfig] = None,
    *,
    # global defaults
    term_order_fn: t.Optional[TermOrderFn] = add_term_order,
    term_col: str = "term",
    dedupe_fn: t.Optional[DedupeFn] = None,
    # per-dataset overrides
    term_order_by_dataset: t.Optional[dict[str, t.Tuple[TermOrderFn, str]]] = None,
    dedupe_fn_by_dataset: t.Optional[dict[str, DedupeFn]] = None,
) -> None:
    """
    Mutate `datasets` in-place to attach cleaning hooks for each bundle.

    Precedence:
      - term_order_by_dataset[name] → (fn, col)
      - term_order_fn / term_col

      - dedupe_fn_by_dataset[name]
      - dedupe_fn

    Also propagates CleaningConfig.student_id_alias down into each bundle.
    """
    for name, bundle in datasets.items():
        # --- term_order_fn + term_column (per-dataset first) ---
        if term_order_by_dataset and name in term_order_by_dataset:
            fn, col = term_order_by_dataset[name]
            bundle.setdefault("term_order_fn", fn)
            bundle.setdefault("term_column", col)
        elif term_order_fn is not None:
            bundle.setdefault("term_order_fn", term_order_fn)
            bundle.setdefault("term_column", term_col)

        # --- student_id alias from global CleaningConfig ---
        if cleaning_cfg and cleaning_cfg.student_id_alias:
            bundle.setdefault("student_id_alias", cleaning_cfg.student_id_alias)

        # --- dedupe_fn (per-dataset first) ---
        if dedupe_fn_by_dataset and name in dedupe_fn_by_dataset:
            bundle.setdefault("dedupe_fn", dedupe_fn_by_dataset[name])
        elif dedupe_fn is not None:
            bundle.setdefault("dedupe_fn", dedupe_fn)


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
    # forced dtype overrides by normalized column name.
    # e.g. {"student_id": "string", "term_order": "Int64"}
    forced_dtypes: dict[str, str] = field(default_factory=dict)
    # if False, a failed forced cast will raise instead of falling back
    allow_forced_cast_fallback: bool = True


def dtype_opts_from_cleaning_config(
    cfg: "CleaningConfig | None",
) -> DtypeGenerationOptions:
    if cfg is None:
        return DtypeGenerationOptions()
    return DtypeGenerationOptions(
        date_formats=cfg.date_formats,
        dtype_confidence_threshold=cfg.dtype_confidence_threshold,
        min_non_null=cfg.min_non_null,
        boolean_map=cfg.boolean_map,
        forced_dtypes=getattr(cfg, "forced_dtypes", {}) or {},
        allow_forced_cast_fallback=getattr(cfg, "allow_forced_cast_fallback", True),
    )


def _cast_series_to_nullable_dtype(
    s: pd.Series,
    dtype_str: str,
    boolean_map: dict[str, bool],
) -> pd.Series:
    """
    Cast a Series to one of our supported nullable dtypes.

    Shared between training-time forced dtypes and inference-time schema enforcement.
    """
    try:
        if dtype_str.startswith("datetime64"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format, so each element will be parsed individually",
                    category=UserWarning,
                )
                return pd.to_datetime(s, errors="coerce")

        if dtype_str == "Int64":
            return pd.to_numeric(s, errors="coerce").astype("Int64")

        if dtype_str == "Float64":
            return pd.to_numeric(s, errors="coerce").astype("Float64")

        if dtype_str == "boolean":
            return (
                s.astype("string")
                .str.strip()
                .str.lower()
                .map(boolean_map)
                .astype("boolean")
            )

        if dtype_str == "string":
            return s.astype("string")

        # fallback: let pandas handle it (for forward compatibility)
        return s.astype(dtype_str)

    except Exception as e:
        raise ValueError(f"Failed to cast Series to {dtype_str}: {e}")


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
    if ptypes.is_datetime64_any_dtype(s):
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

    Honors any `forced_dtypes` configured in `DtypeGenerationOptions` before applying
    heuristic inference.

    NOTE: Use only on training data. At inference time, rely on `enforce_schema`
    using a schema frozen from the trained data.
    """
    opts = opts or DtypeGenerationOptions()
    out = df.copy()

    for col in out.columns:
        # 1) Forced dtype override, if configured for this normalized column
        forced = opts.forced_dtypes.get(col)
        if forced is not None:
            try:
                out[col] = _cast_series_to_nullable_dtype(
                    out[col],
                    forced,
                    boolean_map=opts.boolean_map,
                )
                LOGGER.info(
                    "Applied forced dtype override: column=%s dtype=%s", col, forced
                )
                # Skip heuristic inference for this column
                continue
            except Exception as e:
                msg = f"Failed to apply forced dtype '{forced}' for column '{col}': {e}"
                if opts.allow_forced_cast_fallback:
                    LOGGER.warning("%s; falling back to inferred dtype", msg)
                    # Fall through to normal inference
                else:
                    # Strict mode: fail fast
                    raise

        # 2) Normal heuristic inference for all other columns
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
    cleaning_cfg: "CleaningConfig | None" = None,
) -> pd.DataFrame:
    """
    End-to-end cleaner with robust training-time dtype generation and consistent policies.

    Typical pattern:
      - TRAINING:
          clean_dataset(..., generate_dtypes=True)
          -> build_schema_contract(...)
      - INFERENCE:
          clean_dataset(..., generate_dtypes=False)  # if you still want cleaning hooks
          then enforce_schema(...) using the frozen schema.

    `generate_dtypes=False` is useful at inference to avoid re-generating dtypes
    and introducing train–inference skew; instead, `enforce_schema` should dictate
    the final dtypes.
    """
    if cleaning_cfg is not None:
        inference_opts = dtype_opts_from_cleaning_config(cleaning_cfg)
    else:
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
            # pick up alias from cleaning_cfg if not present
            student_id_alias=spec.get("student_id_alias")
            if "student_id_alias" in spec
            else (cleaning_cfg.student_id_alias if cleaning_cfg else None),
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
    null_tokens = cleaning_cfg.null_tokens if cleaning_cfg else ["(Blank)"]
    g = g.replace(null_tokens, np.nan)

    obj_cols = g.select_dtypes(include=["object", "string"]).columns
    if cleaning_cfg is None or cleaning_cfg.treat_empty_strings_as_null:
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
    cleaning_cfg: "CleaningConfig | None" = None,
) -> t.Dict[str, pd.DataFrame]:
    """
    Clean each dataset bundle and return {name: cleaned_df}.
    Each bundle must be shaped like {"data": df, "...spec keys..."}.

    `generate_dtypes` is passed through to `clean_dataset`, so you can also
    reuse this at inference-time by setting it to False and then applying
    `enforce_schema_contract` / `enforce_schema`.
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
            cleaning_cfg=cleaning_cfg,
        )
        LOGGER.info("%s - Finished cleaning; final shape=%s", key, cleaned[key].shape)
    return cleaned


def clean_bronze_datasets(
    cfg: CustomProjectConfig,
    df_map: t.Dict[str, t.Tuple[str, pd.DataFrame]],
    run_type: str,
    *,
    cleaning_cfg: CleaningConfig | None = None,
    term_order_fn: TermOrderFn | None = add_term_order,
    term_col: str = "term",
    dedupe_fn: DedupeFn | None = None,
    # per-dataset overrides
    term_order_by_dataset: t.Optional[dict[str, t.Tuple[TermOrderFn, str]]] = None,
    dedupe_fn_by_dataset: t.Optional[dict[str, DedupeFn]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper for:
      - build_datasets_from_bronze(...)
      - attach_cleaning_hooks(...)
      - clean_all_datasets_map(...)
    """
    datasets = build_datasets_from_bronze(cfg, df_map)

    attach_cleaning_hooks(
        datasets,
        cleaning_cfg=cleaning_cfg,
        term_order_fn=term_order_fn,
        term_col=term_col,
        dedupe_fn=dedupe_fn,
        term_order_by_dataset=term_order_by_dataset,
        dedupe_fn_by_dataset=dedupe_fn_by_dataset,
    )

    generate_dtypes = run_type == "train"
    return clean_all_datasets_map(
        datasets,
        enforce_uniqueness=True,
        generate_dtypes=generate_dtypes,
        cleaning_cfg=cleaning_cfg,
    )


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
            g[c] = _cast_series_to_nullable_dtype(g[c], dt, bmap)
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


def build_schema_contract(
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


def enforce_schema_contract(
    raw_map: dict[str, pd.DataFrame], schema_contract: dict
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, df in raw_map.items():
        if name not in schema_contract["datasets"]:
            raise KeyError(f"Dataset '{name}' is not present in schema_contract")
        out[name] = enforce_schema(df, schema_contract["datasets"][name])
    return out


def load_or_build_schema_contract(
    cfg: CustomProjectConfig,
    run_type: str,
    cleaned: dict[str, pd.DataFrame],
    specs: dict[str, dict[str, t.Any] | CleanSpec],
) -> dict[str, t.Any]:
    """
    If run_type=="train": build + save the schema_contract to
    cfg.preprocessing.cleaning.schema_contract_path.

    Otherwise: load it from that path.
    """
    cleaning_cfg: CleaningConfig | None = getattr(
        getattr(cfg, "preprocessing", None), "cleaning", None
    )
    schema_path = cleaning_cfg.schema_contract_path if cleaning_cfg else None
    if not schema_path:
        raise ValueError(
            "preprocessing.cleaning.schema_contract_path must be set "
            "on CustomProjectConfig for schema contract I/O."
        )

    if run_type == "train":
        schema_contract = build_schema_contract(cleaned, specs=specs)
        save_schema_contract(schema_contract, schema_path)
    else:
        schema_contract = load_schema_contract(schema_path)
    return schema_contract


def align_and_rank_dataframes(
    dfs: t.List[pd.DataFrame],
    term_column: str = "term_order",
    core_term_col: t.Optional[str] = "is_core_term",
) -> t.List[pd.DataFrame]:
    """
    Align multiple dataframes to a common term range and assign ranks.
    - Always assigns `term_rank` based on chronological order.
    - If `core_term_col` is provided AND exists in all dataframes,
      adds `core_term_rank` based on rows where that column is True.
    - If not provided or missing in any dataframe, `core_term_rank` is omitted.
    """
    if len(dfs) <= 1:
        raise ValueError("Must provide at least two dataframes to align")
    if not all(term_column in df.columns for df in dfs):
        raise ValueError(f"All dataframes must have column '{term_column}'.")
    if any(df.empty for df in dfs):
        raise ValueError("There is an empty dataframe in the list of dataframes.")

    # Determine common overlapping term range
    try:
        min_term = max(df[term_column].dropna().min() for df in dfs)
        max_term = min(df[term_column].dropna().max() for df in dfs)
    except ValueError:
        raise ValueError(
            "Cannot determine term range; one or more dataframes have no valid term values."
        )
    if pd.isna(min_term) or pd.isna(max_term) or min_term > max_term:
        raise ValueError(f"No overlapping {term_column} range across dataframes.")
    LOGGER.info("Common term range across dataframes: %s → %s", min_term, max_term)
    # Check if we have a usable core-term column in all dataframes
    has_core_flag = core_term_col is not None and all(
        core_term_col in df.columns for df in dfs
    )
    if has_core_flag:
        LOGGER.info(
            "Detected '%s' in all dataframes; core-term ranking will be computed.",
            core_term_col,
        )
    else:
        LOGGER.info("No valid core-term column detected; skipping core-term ranking.")

    # Collect all terms in range for rank mapping
    term_union = pd.concat(
        [
            df.loc[df[term_column].between(min_term, max_term), [term_column]]
            for df in dfs
        ],
        ignore_index=True,
    ).drop_duplicates()
    term_order_sorted = sorted(term_union[term_column].unique())
    term_rank_map = {term: i for i, term in enumerate(term_order_sorted)}

    # Build core-term rank map if applicable
    core_term_rank_map = None
    if has_core_flag:
        core_union = pd.concat(
            [
                df.loc[
                    df[term_column].between(min_term, max_term)
                    & df[core_term_col].astype(bool),
                    [term_column],
                ]
                for df in dfs
            ],
            ignore_index=True,
        ).drop_duplicates()
        if not core_union.empty:
            core_term_order_sorted = sorted(core_union[term_column].unique())
            core_term_rank_map = {
                term: i for i, term in enumerate(core_term_order_sorted)
            }
            LOGGER.info(
                "Computed core-term ranks for %d distinct terms (range: %s → %s).",
                len(core_term_order_sorted),
                core_term_order_sorted[0],
                core_term_order_sorted[-1],
            )
    # Apply range filtering and rank assignment
    result: t.List[pd.DataFrame] = []
    for i, df in enumerate(dfs, start=1):
        mask = df[term_column].between(min_term, max_term)
        df_filtered = df.loc[mask].copy()
        df_filtered["term_rank"] = (
            df_filtered[term_column].map(term_rank_map).astype("Int64")
        )
        if has_core_flag and core_term_rank_map:
            df_filtered["core_term_rank"] = (
                df_filtered[term_column].map(core_term_rank_map).astype("Int64")
            )

        result.append(df_filtered.reset_index(drop=True))
        LOGGER.info(
            "DataFrame %d aligned: shape=%s, term range=[%s → %s]",
            i,
            df_filtered.shape,
            df_filtered[term_column].min(),
            df_filtered[term_column].max(),
        )
    LOGGER.info(
        "Alignment complete: %d dataframes processed (core_term_col=%s, core_term_rank=%s).",
        len(result),
        core_term_col if has_core_flag else "None",
        "included" if has_core_flag and core_term_rank_map else "omitted",
    )

    return result


def _extract_readmit_ids(
    df: pd.DataFrame,
    entry_col: str = "entry_type",
    student_col: str = "student_id",
) -> np.ndarray:
    """
    Return the array of student_ids that have entry_type == 'readmit' in this df.
    If required columns are missing, returns an empty array.
    """
    if entry_col not in df.columns or student_col not in df.columns:
        return np.array([], dtype=object)

    entry = df[entry_col].astype("string").str.lower().str.strip()
    readmit_ids = df.loc[entry == "readmit", student_col].dropna().unique()
    return np.asarray(readmit_ids, dtype=object)


def drop_readmits(
    cohort_df: pd.DataFrame,
    entry_col: str = "entry_type",
    student_col: str = "student_id",
) -> pd.DataFrame:
    """
    Remove ALL rows for any student who has an entry_type of 'readmit'
    (based only on this dataframe).
    """
    out = cohort_df.copy()
    readmit_ids = _extract_readmit_ids(
        out, entry_col=entry_col, student_col=student_col
    )

    if len(readmit_ids) == 0:
        return out

    before = len(out)
    out = out[~out[student_col].isin(readmit_ids)].reset_index(drop=True)
    LOGGER.info(
        "drop_readmits: removed %d rows for %d readmit students",
        before - len(out),
        len(readmit_ids),
    )
    return out


def keep_earlier_record(
    df: pd.DataFrame,
    id_col: str = "student_id",
    term_col: str = "entry_term",
) -> pd.DataFrame:
    """
    Keeps the earliest record per id_col based on term_col, where term_col
    looks like 'Spring 2020', 'Fall 2020', etc.
    """

    def term_to_sort_key(term):
        if pd.isna(term):
            return float("inf")  # treat missing as latest
        term = str(term).strip().title()
        parts = term.split()
        if len(parts) != 2:
            return float("inf")  # unknown format
        season, year_str = parts
        try:
            year = int(year_str)
        except ValueError:
            return float("inf")
        season_order = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
        return year * 10 + season_order.get(season, 5)

    out = df.copy()
    out["_term_sort_key"] = out[term_col].apply(term_to_sort_key)

    out = (
        out.sort_values(by=[id_col, "_term_sort_key"])
        .drop_duplicates(subset=id_col, keep="first")
        .drop(columns=["_term_sort_key"])
        .reset_index(drop=True)
    )

    return out


def assign_numeric_grade(
    df: pd.DataFrame,
    *,
    grade_numeric_map: t.Optional[dict[str, t.Optional[float]]] = None,
    grade_col: str = "grade",
    output_col: str = "course_numeric_grade",
) -> pd.DataFrame:
    """
    CUSTOM SCHOOL FUNCTION

    Assign a numeric value to each grade based on a provided mapping.
    Grades not found in the mapping are skipped (NaN) and printed.
    """

    LOGGER.info("Starting assign_numeric_grade transformation.")

    if grade_numeric_map is None:
        grade_numeric_map = {
            "A": 4.0,
            "A-": 3.7,
            "B+": 3.3,
            "B": 3.0,
            "B-": 2.7,
            "C+": 2.3,
            "C": 2.0,
            "C-": 1.7,
            "D+": 1.3,
            "D": 1.0,
            "D-": 0.7,
            "P": 4.0,
            "P*": 4.0,
            "CH": 4.0,
            "F": 0.0,
            "F*": 0.0,
            "E": 0.0,
            "REF": 0.0,
            "W": 0.0,
            "W*": 0.0,
            "WI": 0.0,
            "WE": 0.0,
            "WC": 0.0,
            "WA": 0.0,
            "WB+": 0.0,
            "WB": 0.0,
            "WB-": 0.0,
            "WD": 0.0,
            "WD-": 0.0,
            "WC+": 0.0,
            "WC-": 0.0,
            "WA-": 0.0,
            "I": 0.0,
            # no numeric GPA equivalent
            "^C": None,
            "^C-": None,
            "^D-": None,
            "^D": None,
            "^D+": None,
            "^E": None,
            "ZD-": None,
            "ZD": None,
            "ZE": None,
            "NR": None,
            "S": None,
            "REP": None,
        }

    grades = df[grade_col].astype("string").str.strip().str.upper()
    df[output_col] = grades.map(grade_numeric_map)

    # -------------------------------------------------
    # Print missing grade keys (present in data, absent from map)
    # -------------------------------------------------
    missing_keys = sorted(set(grades.dropna().unique()) - set(grade_numeric_map.keys()))

    if missing_keys:
        print(f"Grades not found in mapping (skipped): {missing_keys}")

    LOGGER.info("Completed assign_numeric_grade transformation.")
    return df


# ---------------------------
# Serialization helpers
# ---------------------------
def save_schema_contract(schema_contract: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema_contract, f, indent=2, ensure_ascii=False)


def load_schema_contract(path: str) -> dict[str, t.Any]:
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
    "build_schema_contract",
    "enforce_schema_contract",
    "save_schema_contract",
    "load_schema_contract",
    "normalize_columns",
    "create_datasets",
    "clean_all_datasets_map",
]
