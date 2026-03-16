# Schema Validation Comparison: API vs Custom Cleaning Module

## Overview

This document compares the schema validation approaches between:
- **API** (`edvise-api/src/webapp/validation.py`): File upload validation for the web API
- **Custom Cleaning Module** (`edvise/src/edvise/data_audit/custom_cleaning.py`): Training-time and inference-time data cleaning for custom schools

## Key Differences

### 1. **Purpose & Context**

**API Validation:**
- Validates files **at upload time** before they enter the system
- Focuses on **structural validation** (columns, types, constraints)
- Returns validation status to users via HTTP responses
- Uses Pandera schemas for row-level validation

**Custom Cleaning Module:**
- Used during **training and inference** for custom school pipelines
- Focuses on **data transformation** and **schema enforcement**
- Generates training-time dtypes and freezes schemas for inference
- Uses pandas nullable dtypes (Int64, Float64, boolean, string)

### 2. **Column Normalization**

**API (`normalize_col` in `validation.py`):**
```python
def normalize_col(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)  # Replace non-alphanumeric with underscore
    name = re.sub(r"_+", "_", name)  # Collapse multiple underscores
    return name.strip("_")
```
- Simple regex-based approach
- Replaces all non-alphanumeric characters with underscores
- Collapses multiple underscores

**Custom Cleaning (`normalize_columns` in `custom_cleaning.py`):**
```python
def normalize_columns(cols: t.Iterable[str]) -> tuple[pd.Index, dict[str, list[str]]]:
    norm_list = [convert_to_snake_case(c) for c in orig]
    # ...
```
- Uses `convert_to_snake_case` from `edvise.utils.data_cleaning`
- More sophisticated word splitting (handles camelCase, PascalCase, numbers)
- Preserves word boundaries better
- Example: "Student-ID#" → "student_id_#" (API) vs "student_id" (custom cleaning)

**⚠️ REPEATING ELEMENT:** Both implement column normalization, but with different algorithms that may produce different results.

### 3. **Schema Definition & Validation**

**API:**
- **PDP/Edvise**: Uses Pandera `DataFrameModel` schemas from edvise repo:
  - `RawPDPCohortDataSchema` (from `edvise.data_audit.schemas.raw_cohort`)
  - `RawPDPCourseDataSchema` (from `edvise.data_audit.schemas.raw_course`)
- **Custom Schools**: Uses JSON-based Pandera schemas (`DataFrameSchema` built from JSON specs)
- Validates with Pandera's `schema.validate(df, lazy=True)`
- Handles missing required columns, extra columns, and row-level checks

**Custom Cleaning:**
- Uses a **custom schema contract system**:
  - `freeze_schema()`: Captures column names, dtypes, non-null policies, unique keys
  - `enforce_schema()`: Applies frozen schema at inference time
- Schema stored as JSON with:
  - `normalized_columns`: Mapping from original to normalized names
  - `dtypes`: Frozen pandas dtypes (Int64, Float64, boolean, string, datetime64[ns])
  - `non_null_columns`: Columns that must not be null
  - `unique_keys`: Primary key columns
- **No Pandera dependency** - uses pandas directly

**⚠️ REPEATING ELEMENT:** Both validate:
- Column presence (required vs optional)
- Data types
- Nullability constraints
- Unique key constraints

### 4. **Dtype Handling**

**API:**
- Uses Pandera's dtype system (string, Int64, Float64, datetime64, categorical, boolean)
- Dtypes defined in Pandera schemas or JSON specs
- Coercion handled by Pandera

**Custom Cleaning:**
- Uses **pandas nullable dtypes** exclusively:
  - `Int64` (nullable integer)
  - `Float64` (nullable float)
  - `boolean` (nullable boolean)
  - `string` (nullable string)
- **Training-time dtype generation** with confidence thresholds:
  - `dtype_confidence_threshold`: Minimum fraction of values that must coerce successfully (default 0.75)
  - `min_non_null`: Minimum non-null count required (default 10)
  - Tries: datetime (with format strings) → numeric → boolean → string
- **Inference-time**: Uses frozen dtypes from training schema

**⚠️ REPEATING ELEMENT:** Both handle:
- Datetime parsing with multiple format attempts
- Numeric coercion
- Boolean mapping
- String normalization

### 5. **Error Handling**

**API:**
- Raises `HardValidationError` with:
  - `missing_required`: List of missing required columns
  - `extra_columns`: List of unexpected columns
  - `schema_errors`: Pandera schema error details
  - `failure_cases`: Row-level validation failures
  - `raw_to_canon` / `canon_to_raw`: Column name mappings
- Errors formatted for HTTP responses

**Custom Cleaning:**
- Raises `ValueError` for:
  - Column name collisions after normalization
  - Duplicate rows on primary keys
  - Missing required columns (in `enforce_schema`)
- Logs warnings for:
  - Extra columns (drops with warning)
  - Missing optional columns (fills with NA)
- Less structured error format (designed for pipeline logs, not API responses)

**⚠️ REPEATING ELEMENT:** Both detect and report:
- Missing required columns
- Extra/unexpected columns
- Data type mismatches
- Constraint violations

### 6. **Data Cleaning Operations**

**API:**
- **Minimal cleaning** - focuses on validation:
  - Column normalization
  - Type coercion (via Pandera)
  - Encoding detection
  - Header-only pass for performance

**Custom Cleaning:**
- **Comprehensive cleaning pipeline**:
  1. Column normalization
  2. Student ID alias renaming
  3. Null token replacement
  4. Empty string → null conversion
  5. Column dropping
  6. Row dropping (non-null constraints)
  7. Dtype generation (training) or enforcement (inference)
  8. Deduplication (full row + primary key)
  9. Term order assignment
  10. Uniqueness enforcement

**⚠️ REPEATING ELEMENT:** Both perform:
- Column name normalization
- Null handling
- Type coercion

## Repeating Elements Summary

### ⚠️ **PDP-SPECIFIC DUPLICATION** (Critical Issue)

For **PDP data specifically**, there is significant duplication:

1. **Column Normalization Happens TWICE**
   - **API validation** (`validation.py`):
     - First: Header pass uses `normalize_col()` (regex-based) to map raw headers → canonical
     - Then: Calls `read_raw_pdp_cohort_data()` / `read_raw_pdp_course_data()` from edvise
   - **edvise read functions** (`dataio/read.py`):
     - Normalizes columns AGAIN with `convert_to_snake_case()` (word-splitting based)
   - **Problem**: Two different normalization algorithms applied sequentially!
   - **Location**: `edvise/src/edvise/dataio/read.py:173` calls `convert_to_snake_case` after API already normalized

2. **Schema Validation Happens in Multiple Places**
   - **API**: Validates with Pandera schemas (`RawPDPCohortDataSchema`, `RawPDPCourseDataSchema`)
   - **edvise read functions**: Also validate with the same Pandera schemas
   - **Result**: Same validation logic executed twice (though this may be intentional for safety)

3. **Column Name Mapping Duplication**
   - API builds `raw_to_canon` and `canon_to_raw` mappings in header pass
   - But then `read_raw_pdp_*_data` re-normalizes columns, potentially creating mismatches

### High Overlap (Similar Functionality, Different Implementation)

1. **Column Normalization** (General)
   - API: `normalize_col()` - regex-based
   - Custom: `normalize_columns()` - word-splitting based (`convert_to_snake_case`)
   - **PDP-specific**: Both are used sequentially (duplication!)
   - **Recommendation**: For PDP, API should skip normalization and let edvise read functions handle it

2. **Dtype Coercion**
   - API: Pandera handles coercion
   - Custom: Custom `_cast_series_to_nullable_dtype()` function
   - Both handle: datetime, numeric, boolean, string
   - **Recommendation**: Could share datetime parsing logic

3. **Null Handling**
   - API: Pandera's `nullable` field
   - Custom: Explicit null token replacement + empty string handling
   - **Recommendation**: API could benefit from null token handling for custom schools

4. **Missing/Extra Column Detection**
   - Both detect missing required columns
   - Both detect extra columns
   - **Recommendation**: Logic is similar but could be shared

### Low Overlap (Different Purposes)

1. **Schema Definition**: Pandera vs Custom JSON contract
2. **Training-time Dtype Generation**: Only in custom cleaning
3. **Data Cleaning Pipeline**: Only in custom cleaning (API is validation-only)
4. **Error Formatting**: API formats for HTTP, custom cleaning logs to console

## Recommendations

### Priority 1: Fix PDP Duplication

1. **Eliminate Double Column Normalization for PDP**
   - **Problem**: API normalizes columns, then edvise read functions normalize again (different algorithms!)
   - **Solution**: For PDP uploads, API should:
     - Skip the header normalization pass, OR
     - Use `convert_to_snake_case` instead of `normalize_col` to match edvise, OR
     - Pass through raw headers and let `read_raw_pdp_*_data` handle all normalization
   - **Impact**: Currently, column names may not match between API validation and pipeline processing

2. **Clarify PDP Validation Flow**
   - Document that API validation for PDP calls edvise read functions
   - Consider if double validation (API + edvise) is necessary or redundant
   - If redundant, skip API's Pandera validation and rely on edvise read functions

### Priority 2: General Improvements

3. **Standardize Column Normalization**
   - Choose one algorithm (`convert_to_snake_case` is more sophisticated)
   - Use consistently across API and edvise
   - Document when to use which (if both must exist)

4. **Share Datetime Parsing Logic**
   - Both try multiple datetime formats
   - Could extract to shared utility

5. **Consider Unified Schema System**
   - API uses Pandera for validation
   - Custom cleaning uses custom schema contract
   - Could explore if custom cleaning could use Pandera for consistency

6. **Document the Split**
   - API = upload-time validation (structural)
   - Custom cleaning = pipeline-time transformation (comprehensive)
   - Make the distinction clear in code comments

7. **Null Token Handling**
   - Custom cleaning has sophisticated null token replacement
   - API could benefit from this for custom school uploads

## Code Locations

- **API Validation**: `edvise-api/src/webapp/validation.py`
- **API PDP/Edvise Schemas**: `edvise-api/src/webapp/validation_pdp_edvise.py`
- **Custom Cleaning**: `edvise/src/edvise/data_audit/custom_cleaning.py`
- **Column Normalization (edvise)**: `edvise/src/edvise/utils/data_cleaning.py::convert_to_snake_case`
- **Pandera Schemas (edvise)**: 
  - `edvise/src/edvise/data_audit/schemas/raw_cohort.py`
  - `edvise/src/edvise/data_audit/schemas/raw_course.py`
- **PDP Read Functions (where duplication occurs)**: 
  - `edvise/src/edvise/dataio/read.py::read_raw_pdp_cohort_data`
  - `edvise/src/edvise/dataio/read.py::read_raw_pdp_course_data`
  - `edvise/src/edvise/dataio/read.py::_read_and_prepare_pdp_data` (line 173: `convert_to_snake_case`)

## PDP-Specific Flow Analysis

### Current PDP Validation Flow (with duplication):

```
1. API receives file upload
2. API validation.py::validate_dataset()
   ├─ Header pass with normalize_col() → raw_to_canon mapping
   ├─ Detects PDP institution_id
   └─ Calls _validate_pdp_with_edvise_read()
      └─ Calls read_raw_pdp_cohort_data() or read_raw_pdp_course_data()
         └─ Calls _read_and_prepare_pdp_data()
            ├─ Reads CSV/table
            ├─ ⚠️ NORMALIZES COLUMNS AGAIN with convert_to_snake_case() (line 173)
            ├─ Applies transformations
            └─ Validates with Pandera schema (RawPDPCohortDataSchema/RawPDPCourseDataSchema)
```

**Issues:**
- Column normalization happens twice with different algorithms
- Column name mappings from API header pass may not match final normalized names
- Potential for mismatches between API validation and pipeline processing
