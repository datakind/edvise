#!/usr/bin/env python3
"""
Generate synthetic course enrollment rows from a validated Edvise student file.

Each learner gets a latent performance profile (ability, full-time vs part-time) that
drives grade distributions, completion speed, and dropout risk. Course sequences mix
major-specific ladders with gen-ed; levels progress from 100- to 400-level as credits
accumulate. Output validates against RawEdviseCourseDataSchema.

Typical use (after generating students):

    PYTHONPATH=src python src/edvise/scripts/generate_synthetic_edvise_course_data.py \\
        --students data/synthetic/edvise_students_full.csv \\
        --output data/synthetic/edvise_courses.csv
"""

from __future__ import annotations

import argparse
import logging
import math
import pathlib
import random
import re
import sys
from typing import Any, Iterator

import pandas as pd

from edvise.data_audit.schemas import RawEdviseCourseDataSchema

LOGGER = logging.getLogger(__name__)

TERM_ORDER = ("FALL", "WINTER", "SPRING", "SUMMER")

# --- Curriculum: gen-ed + major ladders (prefix, number, title, credits, gen_ed_flag, level 1-4) ---

GenEdRow = tuple[str, str, str, float, bool, int]

GEN_ED_POOL: list[GenEdRow] = [
    ("ENGL", "101", "English Composition I", 3.0, True, 1),
    ("ENGL", "102", "English Composition II", 3.0, True, 1),
    ("MATH", "111", "College Algebra", 3.0, True, 1),
    ("MATH", "121", "Precalculus", 3.0, True, 1),
    ("HIST", "101", "World History I", 3.0, True, 1),
    ("PSYC", "101", "Introduction to Psychology", 3.0, True, 1),
    ("SOC", "101", "Introduction to Sociology", 3.0, True, 1),
    ("COMM", "101", "Public Speaking", 3.0, True, 1),
    ("ART", "101", "Introduction to Visual Art", 1.0, True, 1),
    ("PE", "100", "Wellness Activity", 1.0, True, 1),
]

MAJOR_COURSES: dict[str, list[GenEdRow]] = {
    "Biology": [
        ("BIOL", "101", "Principles of Biology I", 3.0, False, 1),
        ("BIOL", "102", "Principles of Biology II", 3.0, False, 1),
        ("BIOL", "201", "Genetics", 3.0, False, 2),
        ("BIOL", "202", "Cell Biology", 3.0, False, 2),
        ("BIOL", "301", "Ecology", 3.0, False, 3),
        ("BIOL", "302", "Microbiology", 4.0, False, 3),
        ("BIOL", "401", "Molecular Biology", 3.0, False, 4),
        ("BIOL", "402", "Senior Seminar", 3.0, False, 4),
        ("CHEM", "121", "General Chemistry I", 4.0, False, 1),
        ("CHEM", "122", "General Chemistry II", 4.0, False, 2),
    ],
    "Chemistry": [
        ("CHEM", "121", "General Chemistry I", 4.0, False, 1),
        ("CHEM", "122", "General Chemistry II", 4.0, False, 1),
        ("CHEM", "221", "Organic Chemistry I", 4.0, False, 2),
        ("CHEM", "222", "Organic Chemistry II", 4.0, False, 2),
        ("CHEM", "331", "Physical Chemistry I", 3.0, False, 3),
        ("CHEM", "332", "Physical Chemistry II", 3.0, False, 3),
        ("CHEM", "401", "Inorganic Chemistry", 3.0, False, 4),
        ("CHEM", "402", "Senior Laboratory", 3.0, False, 4),
        ("MATH", "221", "Calculus I", 4.0, False, 1),
        ("PHYS", "201", "Physics I", 4.0, False, 2),
    ],
    "Computer Science": [
        ("CS", "101", "Introduction to Programming", 3.0, False, 1),
        ("CS", "102", "Data Structures", 3.0, False, 1),
        ("CS", "201", "Algorithms", 3.0, False, 2),
        ("CS", "210", "Computer Systems", 3.0, False, 2),
        ("CS", "301", "Databases", 3.0, False, 3),
        ("CS", "302", "Software Engineering", 3.0, False, 3),
        ("CS", "401", "Operating Systems", 3.0, False, 4),
        ("CS", "402", "Capstone Project", 3.0, False, 4),
        ("MATH", "221", "Calculus I", 4.0, False, 1),
        ("MATH", "222", "Calculus II", 4.0, False, 2),
    ],
    "Mathematics": [
        ("MATH", "221", "Calculus I", 4.0, False, 1),
        ("MATH", "222", "Calculus II", 4.0, False, 1),
        ("MATH", "301", "Linear Algebra", 3.0, False, 2),
        ("MATH", "302", "Differential Equations", 3.0, False, 2),
        ("MATH", "311", "Abstract Algebra", 3.0, False, 3),
        ("MATH", "321", "Real Analysis I", 3.0, False, 3),
        ("MATH", "401", "Real Analysis II", 3.0, False, 4),
        ("MATH", "402", "Topology", 3.0, False, 4),
        ("STAT", "301", "Probability", 3.0, False, 2),
        ("CS", "101", "Introduction to Programming", 3.0, False, 1),
    ],
    "Psychology": [
        ("PSYC", "201", "Research Methods", 3.0, False, 2),
        ("PSYC", "220", "Developmental Psychology", 3.0, False, 2),
        ("PSYC", "301", "Cognitive Psychology", 3.0, False, 3),
        ("PSYC", "302", "Abnormal Psychology", 3.0, False, 3),
        ("PSYC", "401", "Clinical Practicum", 3.0, False, 4),
        ("PSYC", "402", "Senior Thesis", 3.0, False, 4),
        ("STAT", "201", "Statistics for Social Science", 3.0, False, 2),
        ("BIOL", "101", "Principles of Biology I", 3.0, False, 1),
    ],
    "Business": [
        ("BUS", "101", "Introduction to Business", 3.0, False, 1),
        ("ACCT", "201", "Financial Accounting", 3.0, False, 2),
        ("ACCT", "202", "Managerial Accounting", 3.0, False, 2),
        ("FIN", "301", "Corporate Finance", 3.0, False, 3),
        ("MKTG", "301", "Principles of Marketing", 3.0, False, 3),
        ("MGMT", "401", "Strategic Management", 3.0, False, 4),
        ("BUS", "402", "Business Capstone", 3.0, False, 4),
        ("ECON", "201", "Microeconomics", 3.0, False, 2),
        ("ECON", "202", "Macroeconomics", 3.0, False, 2),
    ],
    "Nursing": [
        ("NURS", "101", "Introduction to Nursing", 3.0, False, 1),
        ("NURS", "201", "Health Assessment", 4.0, False, 2),
        ("NURS", "301", "Medical-Surgical Nursing I", 4.0, False, 3),
        ("NURS", "302", "Medical-Surgical Nursing II", 4.0, False, 3),
        ("NURS", "401", "Community Health", 3.0, False, 4),
        ("BIOL", "201", "Genetics", 3.0, False, 2),
        ("CHEM", "121", "General Chemistry I", 4.0, False, 1),
    ],
    "History": [
        ("HIST", "201", "U.S. History to 1865", 3.0, False, 2),
        ("HIST", "202", "U.S. History Since 1865", 3.0, False, 2),
        ("HIST", "301", "Historiography", 3.0, False, 3),
        ("HIST", "401", "Senior Seminar", 3.0, False, 4),
        ("POLS", "101", "American Government", 3.0, False, 1),
    ],
    "Spanish": [
        ("SPAN", "101", "Elementary Spanish I", 3.0, False, 1),
        ("SPAN", "102", "Elementary Spanish II", 3.0, False, 1),
        ("SPAN", "201", "Intermediate Spanish", 3.0, False, 2),
        ("SPAN", "301", "Advanced Composition", 3.0, False, 3),
        ("SPAN", "401", "Spanish Literature", 3.0, False, 4),
    ],
    "Liberal Arts": [
        ("LA", "101", "Liberal Arts Seminar I", 3.0, False, 1),
        ("LA", "201", "Liberal Arts Seminar II", 3.0, False, 2),
        ("LA", "301", "Interdisciplinary Project", 3.0, False, 3),
        ("LA", "401", "Capstone", 3.0, False, 4),
    ],
}

DEFAULT_MAJOR_KEY = "Liberal Arts"

# Extra electives so bachelor totals (~120 cr) are reachable without duplicate prefixes.
EXTRA_ELECTIVES: list[GenEdRow] = [
    (
        "ELEC",
        f"{200 + i}",
        f"General Elective {i}",
        3.0,
        False,
        min(4, 1 + (i - 1) // 8),
    )
    for i in range(1, 25)
]

# Modality distribution: face-to-face, hybrid, online
MODALITY_VALUES = ["Face-to-face", "Hybrid", "Online"]
MODALITY_WEIGHTS = [60, 10, 30]

# Letter grades for GPA-bearing courses (schema-valid)
_GRADE_HIGH = (
    ["A", "A-", "B+", "B", "B-"],
    [0.35, 0.25, 0.20, 0.12, 0.08],
)
_GRADE_MID = (
    ["A-", "B+", "B", "B-", "C+", "C"],
    [0.10, 0.15, 0.25, 0.20, 0.15, 0.15],
)
_GRADE_LOW = (
    ["B-", "C+", "C", "C-", "D+", "D", "F", "W"],
    [0.12, 0.22, 0.26, 0.18, 0.12, 0.07, 0.02, 0.01],
)


def _parse_entry_year(label: str) -> int:
    """First calendar year of academic label 'YYYY-YY'."""
    s = str(label).strip()
    m = re.match(r"^(\d{4})-\d{2}$", s)
    if not m:
        raise ValueError(f"Invalid entry_year / academic_year: {label!r}")
    return int(m.group(1))


def _term_index(term_val: Any) -> int:
    """Map student/course term string or category to 0..3."""
    if pd.isna(term_val):
        raise ValueError("entry_term is null")
    u = str(term_val).strip().upper()
    if u in TERM_ORDER:
        return TERM_ORDER.index(u)
    aliases = {"FA": 0, "WI": 1, "SP": 2, "SU": 3, "SM": 3}
    if u in aliases:
        return aliases[u]
    # e.g. "FALL 2023"
    for i, name in enumerate(TERM_ORDER):
        if name in u.replace(" ", ""):
            return i
    raise ValueError(f"Unrecognized term: {term_val!r}")


def _academic_year_for(fall_start_year: int) -> str:
    y = fall_start_year
    return f"{y}-{str(y + 1)[2:]}"


def _advance_term(fall_start_year: int, term_idx: int) -> tuple[int, int]:
    term_idx += 1
    if term_idx >= len(TERM_ORDER):
        term_idx = 0
        fall_start_year += 1
    return fall_start_year, term_idx


def iter_term_labels(
    entry_year: str, entry_term: Any
) -> Iterator[tuple[str, str]]:
    """Yield (academic_year YYYY-YY, term name FALL|...) in chronological order."""
    fall_y = _parse_entry_year(entry_year)
    ti = _term_index(entry_term)
    # If entry_term is Spring of AY 2023-24, fall_y should still be 2023
    fy = fall_y
    while True:
        yield _academic_year_for(fy), TERM_ORDER[ti]
        fy, ti = _advance_term(fy, ti)


# Bachelor must allow >32 term slots so part-time students can finish (on-time window is 32).
BACHELOR_MAX_TERMS_CAP = 40


def program_target_credits(intended_program_type: Any) -> tuple[int, int]:
    """
    Return (target_degree_credits, max_terms_cap) from intended_program_type string.
    """
    if intended_program_type is None or (
        isinstance(intended_program_type, float) and math.isnan(intended_program_type)
    ):
        return 120, BACHELOR_MAX_TERMS_CAP
    s = str(intended_program_type).strip().lower()
    if not s or s == "nan":
        return 120, BACHELOR_MAX_TERMS_CAP
    if "bachelor" in s:
        return 120, BACHELOR_MAX_TERMS_CAP
    if "associate" in s:
        return 60, 18
    if "certificate" in s or "diploma" in s:
        return 30, 12
    return 120, BACHELOR_MAX_TERMS_CAP


def _normalize_major(declared_major_at_entry: Any) -> str:
    if declared_major_at_entry is None or (
        isinstance(declared_major_at_entry, float) and math.isnan(declared_major_at_entry)
    ):
        return DEFAULT_MAJOR_KEY
    name = str(declared_major_at_entry).strip()
    if not name or name.lower() == "nan":
        return DEFAULT_MAJOR_KEY
    if name in MAJOR_COURSES:
        return name
    if name == "Undeclared":
        return DEFAULT_MAJOR_KEY
    return DEFAULT_MAJOR_KEY


def _draw_credit_hours(rng: random.Random) -> float:
    """~82% 3cr, 5% 1cr, 13% 4cr."""
    r = rng.random()
    if r < 0.05:
        return 1.0
    if r < 0.18:
        return 4.0
    return 3.0


def _credit_hours_for_course(base_credits: float, rng: random.Random) -> float:
    """Use catalog hours for fixed lab/sequence courses; otherwise sample 3/1/4 mix."""
    if base_credits not in (3.0,):
        return float(base_credits)
    return _draw_credit_hours(rng)


def _sample_grade(ability: float, rng: random.Random) -> str:
    if rng.random() < 0.001 + 0.006 * (1.0 - ability):
        return "W"
    if ability >= 0.50:
        g, w = _GRADE_HIGH
    elif ability >= 0.26:
        g, w = _GRADE_MID
    else:
        g, w = _GRADE_LOW
    return rng.choices(g, weights=w, k=1)[0]


def _credits_earned_for_grade(grade: str, attempted: float) -> float:
    g = grade.strip().upper()
    if g in ("F", "W", "WD", "U", "UNSAT", "I", "IP", "AU", "NG", "NR"):
        return 0.0
    if g in ("P", "PASS", "S", "SAT", "M", "O"):
        return attempted
    # A through D-
    if g.startswith("A") or g.startswith("B") or g.startswith("C") or g.startswith("D"):
        return attempted
    return 0.0


def _term_credit_target(full_time: bool, ability: float, rng: random.Random) -> float:
    """Full-time ~16–19; part-time ~10–15 (targets ~60% on-time within FT/PT windows)."""
    if full_time:
        lo, hi = 16, 19
        if ability >= 0.55:
            lo, hi = 17, 19
        elif ability < 0.30:
            lo, hi = 14, 18
    else:
        lo, hi = 10, 15
        if ability < 0.30:
            lo, hi = 10, 13
    return float(rng.randint(lo, hi))


# Winter / summer: at most 2 courses; most terms have 0, then 1, few have 2
WINTER_SUMMER_COURSE_COUNTS = [0, 1, 2]
WINTER_SUMMER_COURSE_WEIGHTS = [46, 40, 14]


def _winter_summer_max_courses(rng: random.Random) -> int:
    return rng.choices(WINTER_SUMMER_COURSE_COUNTS, weights=WINTER_SUMMER_COURSE_WEIGHTS, k=1)[
        0
    ]


def _max_allowed_level(progress: float) -> int:
    """progress in [0,1] -> highest course level (1-4) student may enroll in."""
    if progress < 0.07:
        return 1
    if progress < 0.22:
        return 2
    if progress < 0.48:
        return 3
    return 4


def _catalog_for_major(major_key: str) -> list[GenEdRow]:
    major_rows = list(MAJOR_COURSES.get(major_key, MAJOR_COURSES[DEFAULT_MAJOR_KEY]))
    pool: list[GenEdRow] = []
    pool.extend(GEN_ED_POOL)
    pool.extend(major_rows)
    pool.extend(EXTRA_ELECTIVES)
    return pool


# Calendar "on-time" windows for bachelor's (120 cr): 150% of nominal pace, four slots / year
ON_TIME_TERM_SLOTS_FULL_TIME_BACHELOR = 24  # 6 years * Fall/Winter/Spring/Summer
ON_TIME_TERM_SLOTS_PART_TIME_BACHELOR = 32  # 8 years * four slots


def _run_learner_enrollment(
    *,
    learner_id: str,
    entry_year: str,
    entry_term: Any,
    intended_program_type: Any,
    declared_major_at_entry: Any,
    pell_recipient_year1: Any,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int, float, float, bool, bool]:
    """
    Simulate one learner; return rows, terms_used, earned, target, full_time, graduated.
    ``terms_used`` counts every Fall/Winter/Spring/Summer step (including terms with 0 courses).
    """
    target, max_terms = program_target_credits(intended_program_type)
    major_key = _normalize_major(declared_major_at_entry)
    catalog = _catalog_for_major(major_key)

    ability = rng.betavariate(4.2, 2.0)
    full_time = rng.random() < 0.78
    transfer_boost = rng.uniform(0, 24) if rng.random() < 0.18 else 0.0
    earned = min(float(transfer_boost), target * 0.45)

    pell_s = None if pd.isna(pell_recipient_year1) else str(pell_recipient_year1).strip().upper()
    if pell_s in ("Y", "YES", "N", "NO"):
        pell_norm = "Y" if pell_s.startswith("Y") else "N"
    else:
        pell_norm = None

    completed: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    term_iter = iter_term_labels(entry_year, entry_term)
    terms_used = 0

    while earned < target and terms_used < max_terms:
        academic_year, term_name = next(term_iter)
        terms_used += 1

        progress = min(1.0, earned / max(target, 1.0))
        # Very rare dropout — most students finish; ~40% exceed on-time window
        if terms_used > 20 and earned < target * 0.08 and ability < 0.16:
            if rng.random() < 0.012:
                break
        if terms_used > 34 and earned < target * 0.36:
            if rng.random() < 0.015 + 0.03 * (1.0 - ability):
                break

        remaining = target - earned
        term_u = str(term_name).strip().upper()
        if term_u in ("FALL", "SPRING"):
            budget = _term_credit_target(full_time, ability, rng)
            budget = min(budget, max(remaining + 6, 6))
            max_courses_this_term = 10**6
        else:
            # WINTER / SUMMER: 0–2 courses (most often 0, then 1, rarely 2)
            max_courses_this_term = _winter_summer_max_courses(rng)
            budget = (
                min(max_courses_this_term * 4.5 + 1.0, max(remaining + 4.0, 4.0))
                if max_courses_this_term > 0
                else 0.0
            )

        term_credits_scheduled = 0.0
        courses_added_this_term = 0
        max_level = _max_allowed_level(progress)
        section_i = 0

        candidates = [
            c
            for c in catalog
            if (c[0], c[1]) not in completed and c[5] <= max_level
        ]
        rng.shuffle(candidates)
        if not candidates and earned < target:
            candidates = [c for c in catalog if (c[0], c[1]) not in completed]
            rng.shuffle(candidates)

        while (
            term_credits_scheduled < budget
            and candidates
            and courses_added_this_term < max_courses_this_term
        ):
            fits: list[tuple[GenEdRow, float]] = []
            for c in list(candidates):
                cr_try = _credit_hours_for_course(c[3], rng)
                if term_credits_scheduled + cr_try <= budget + 1.5:
                    fits.append((c, cr_try))
            if not fits:
                break
            pick, cr = fits[rng.randrange(len(fits))]
            candidates.remove(pick)
            prefix, num, title, _base_cr, is_gen, lvl = pick

            grade = _sample_grade(ability, rng)
            attempted = cr
            earned_cr = _credits_earned_for_grade(grade, attempted)

            modality = rng.choices(MODALITY_VALUES, weights=MODALITY_WEIGHTS, k=1)[0]
            section_i += 1
            section_id = f"{terms_used:02d}{section_i:02d}"

            term_pell: str | None
            if terms_used <= 3 and pell_norm is not None:
                term_pell = pell_norm
            elif rng.random() < 0.5:
                term_pell = rng.choice(["Y", "N"])
            else:
                term_pell = None

            rows.append(
                {
                    "learner_id": str(learner_id),
                    "academic_year": academic_year,
                    "academic_term": term_name,
                    "course_prefix": prefix,
                    "course_number": str(num),
                    "course_title": title,
                    "course_section_id": section_id,
                    "grade": grade,
                    "course_credits_attempted": attempted,
                    "course_credits_earned": earned_cr,
                    "department": prefix,
                    "instructional_format": "Lecture",
                    "academic_level": f"Level {lvl}",
                    "course_begin_date": pd.NaT,
                    "course_end_date": pd.NaT,
                    "instructional_modality": modality,
                    "gen_ed_flag": "Y" if is_gen else "N",
                    "prerequisite_flag": "N",
                    "instructor_appointment_status": rng.choice(
                        ["Full-time", "Part-time", "Adjunct"]
                    ),
                    "gateway_or_developmental_flag": "N",
                    "course_section_size": float(rng.randint(12, 120)),
                    "term_degree": str(intended_program_type)
                    if intended_program_type is not None
                    and str(intended_program_type).strip()
                    else None,
                    "term_declared_major": declared_major_at_entry
                    if declared_major_at_entry is not None
                    and str(declared_major_at_entry).strip()
                    else None,
                    "intent_to_transfer_flag": (
                        rng.choices(["Y", "N"], weights=[15, 85], k=1)[0]
                        if rng.random() < 0.88
                        else None
                    ),
                    "term_pell_recipient": term_pell,
                }
            )

            completed.add((prefix, str(num)))
            term_credits_scheduled += cr
            courses_added_this_term += 1
            earned += earned_cr

            if earned >= target:
                break

    graduated = earned >= target
    return rows, terms_used, earned, target, full_time, graduated


def build_course_rows_for_learner(
    *,
    learner_id: str,
    entry_year: str,
    entry_term: Any,
    intended_program_type: Any,
    declared_major_at_entry: Any,
    pell_recipient_year1: Any,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows, _, _, _, _, _ = _run_learner_enrollment(
        learner_id=learner_id,
        entry_year=entry_year,
        entry_term=entry_term,
        intended_program_type=intended_program_type,
        declared_major_at_entry=declared_major_at_entry,
        pell_recipient_year1=pell_recipient_year1,
        rng=rng,
    )
    return rows


def summarize_bachelor_on_time_rates(
    n_trials: int = 20_000,
    seed: int = 0,
) -> dict[str, float]:
    """
    Monte Carlo for bachelor's seekers only.

    On-time: earned >= 120 within ON_TIME_TERM_SLOTS (24 if full-time = 6y, 32 if part-time = 8y).
    Simulation is tuned so roughly ~60% are on-time and ~40% not (late graduate or no degree).

    Returns shares in [0, 1] for overall / full-time / part-time cohorts.
    """
    majors = list(MAJOR_COURSES.keys())
    rng = random.Random(seed)
    ft_on_time = ft_total = 0
    pt_on_time = pt_total = 0
    all_on_time = 0

    for i in range(n_trials):
        r = random.Random(rng.randint(0, 2**31 - 1))
        _rows, terms_used, earned, _target, full_time, graduated = _run_learner_enrollment(
            learner_id=str(i),
            entry_year="2020-21",
            entry_term="Fall",
            intended_program_type="Bachelor's Degree",
            declared_major_at_entry=majors[i % len(majors)],
            pell_recipient_year1=None,
            rng=r,
        )
        window = (
            ON_TIME_TERM_SLOTS_FULL_TIME_BACHELOR
            if full_time
            else ON_TIME_TERM_SLOTS_PART_TIME_BACHELOR
        )
        on_time = bool(graduated and terms_used <= window)
        if on_time:
            all_on_time += 1
        if full_time:
            ft_total += 1
            if on_time:
                ft_on_time += 1
        else:
            pt_total += 1
            if on_time:
                pt_on_time += 1

    def _share(num: int, den: int) -> float:
        return 0.0 if den == 0 else 1.0 - (num / den)

    return {
        "share_not_on_time_overall": _share(all_on_time, n_trials),
        "share_on_time_overall": all_on_time / n_trials,
        "share_not_on_time_full_time": _share(ft_on_time, ft_total),
        "share_on_time_full_time": ft_on_time / ft_total if ft_total else 0.0,
        "share_not_on_time_part_time": _share(pt_on_time, pt_total),
        "share_on_time_part_time": pt_on_time / pt_total if pt_total else 0.0,
        "n_trials": float(n_trials),
        "n_full_time": float(ft_total),
        "n_part_time": float(pt_total),
    }


def generate_course_dataframe(
    students_df: pd.DataFrame,
    *,
    seed: int | None = None,
    learner_id_col: str = "learner_id",
    validate_schema: bool = True,
) -> pd.DataFrame:
    """
    Build long-format course rows for every student in ``students_df``.

    Expects columns: learner_id, entry_year, entry_term, intended_program_type,
    declared_major_at_entry, pell_recipient_year1 (optional).
    """
    required = [learner_id_col, "entry_year", "entry_term"]
    missing = [c for c in required if c not in students_df.columns]
    if missing:
        raise ValueError(f"students_df missing columns: {missing}")

    rng_master = random.Random(seed)
    all_rows: list[dict[str, Any]] = []
    for _, srow in students_df.iterrows():
        sub_seed = rng_master.randint(0, 2**31 - 1)
        r = random.Random(sub_seed)
        rows = build_course_rows_for_learner(
            learner_id=str(srow[learner_id_col]),
            entry_year=str(srow["entry_year"]),
            entry_term=srow["entry_term"],
            intended_program_type=srow.get("intended_program_type"),
            declared_major_at_entry=srow.get("declared_major_at_entry"),
            pell_recipient_year1=srow.get("pell_recipient_year1"),
            rng=r,
        )
        all_rows.extend(rows)

    schema = RawEdviseCourseDataSchema.to_schema()
    columns = list(schema.columns.keys())
    df = pd.DataFrame(all_rows).reindex(columns=columns)
    for col in ("course_credits_attempted", "course_credits_earned", "course_section_size"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if validate_schema:
        return RawEdviseCourseDataSchema.validate(df, lazy=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic Edvise course rows from a student CSV/DataFrame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--students",
        type=pathlib.Path,
        required=True,
        help="Path to student CSV (e.g. edvise_students_full.csv).",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output CSV or Parquet path.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip RawEdviseCourseDataSchema.validate (not recommended).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    df_stu = pd.read_csv(args.students)
    LOGGER.info("Loaded %s students from %s", len(df_stu), args.students)

    df_courses = generate_course_dataframe(
        df_stu,
        seed=args.seed,
        validate_schema=not args.no_validate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        df_courses.to_parquet(args.output, index=False)
    else:
        df_courses.to_csv(args.output, index=False)
    LOGGER.info("Wrote %s course rows to %s", len(df_courses), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
