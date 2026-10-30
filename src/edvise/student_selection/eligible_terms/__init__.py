"""Schema-specific projectors for eligible inference term listing.

Each module turns a raw upload into student and course frames. PDP and ES
uploads already have those columns, so they do not come through here.
"""

from edvise.student_selection.eligible_terms.genai import (
    BatchInputFrame,
    GenaiTermProjectionError,
    project_genai_eligible_terms,
)

__all__ = [
    "BatchInputFrame",
    "GenaiTermProjectionError",
    "project_genai_eligible_terms",
]
