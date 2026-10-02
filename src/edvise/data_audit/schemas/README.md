## Overview

SST tools for validating data schemas using Pandera.

* **`validation.py`**: Validates incoming datasets against a base schema (and any institution-specific extensions).

### Directory Structure

```
├── base_schema.json         # The core, organization-wide schema definition
├── validation.py            # Validation engine (hard- and soft-errors) using Pandera
└── README.md                # This file
```

#### Base Schema (`base_schema.json`)

* Contains the base schema definitions for each data model, from https://docs.google.com/spreadsheets/d/1zOLv2VOIhDpy6f_2KdOJqLOgA9GNhxW8ZUwneMPF-8A/edit?gid=1337889658#gid=1337889658.
* Defines required and optional columns, data types, and validation checks.

#### Validation (`validation.py`)

This script performs full dataset validation:

1. **Load & Normalize**: Reads a CSV or DataFrame, normalizes column names (only lowercase and underscores).
2. **Schema Merge**: Loads `base_schema.json` and any institution-specific extension from `extensions/`.
3. **Column Discovery**:

   * Flags **extra** columns not in the merged schema (hard error).
   * Flags **missing required** columns (hard error).
   * Flags **missing optional** columns (soft error).
4. **Pandera Validation**: Builds a `DataFrameSchema` from merged specs and runs element-wise checks.

   * **Hard errors** (schema mismatch or check failures) raise a `HardValidationError`.
   * **Soft errors** (missing optional) are reported in the returned status.

**Usage**:

```bash
from validation import validate_dataset
ingestion_data = pd.read_csv('data.csv')
validate_dataset(ingestion_data, [student | semester | course], 'institution_id')
```

**example**

```bash
from validation import validate_dataset
ingestion_data = pd.read_csv('institution_student.csv')
validate_dataset(
      df=ingestion_data, 
      models=[student], 
      institution='institution')
```

The script returns a JSON-like dict:

```python
{
  "validation_status": "passed"  # or "passed_with_soft_errors"
  "missing_optional": [ ... ]
}
```

#### Exceptions

* `HardValidationError`: Raised if there are missing required columns, unexpected columns, or Pandera schema errors.
