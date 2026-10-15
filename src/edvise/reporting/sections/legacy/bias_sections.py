import os
import logging
import re

import pandas as pd

from .. import bias_sections as base_bias_sections
from ...utils import utils

LOGGER = logging.getLogger(__name__)

_BIAS_MITIGATION_LOG_KEYS = (
    "bias_mitigation_custom_threshold",
    "bias_mitigation_group_fnr",
    "bias_mitigation_reference_fnr",
    "bias_mitigation_fnr_abs_gap",
)
_FNR_COMPARISON_KEYS = (
    "bias_mitigation_group_fnr",
    "bias_mitigation_reference_fnr",
    "bias_mitigation_fnr_abs_gap",
)
_DEFAULT_THRESHOLD = 0.5
_DEFAULT_GROUP_COL = "student_group"
_DEFAULT_GROUP_COL_ALIAS = "Student Group"
_BIAS_MITIGATION_ARTIFACT_DIR = "bias_mitigation"
_THRESHOLD_ARTIFACT_PREFIX = "bias_mitigation_thresholds_"
_ARTIFACT_COLUMNS = {
    "threshold": "bias_mitigation_custom_threshold",
    "group_fnr": "bias_mitigation_group_fnr",
    "reference_fnr": "bias_mitigation_reference_fnr",
    "fnr_abs_gap": "bias_mitigation_fnr_abs_gap",
}
_PRE_ADJUSTMENT_NOTE = (
    "These results use one threshold ({default_threshold}) for all students."
)
_ADJUSTMENT_RESULTS_UNAVAILABLE = (
    "Results after the fairness adjustment are not available for this model."
)
_POST_FLAGS_DIR = "bias_flags/post_mitigation"
_POST_PLOTS_DIR = "fnr_plots/post_mitigation"
_PRE_DISPLAY_PLOTS_DIR = "fnr_plots/pre_mitigation_display"
_POST_FNR_RATES = f"{_POST_PLOTS_DIR}/fnr_rates.csv"
_AGE_BIN = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_REFERENCE_OTHER = "other"
_POST_GROUP_NOTE = (
    "{target} students are identified at a threshold of {group_threshold}. "
    "{reference} are identified at a threshold of {default_threshold}."
)
_ALL_OTHER_STUDENTS = "all other students"
_SUPPORT_LABEL_NOTE = "Advisors see this dual-threshold decision via Support Label when the pipeline writes one."


def resolve_student_group_label(card, group_key):
    try:
        alias_dict = getattr(card.cfg, "student_group_aliases", {})
        if not isinstance(alias_dict, dict):
            LOGGER.warning(
                f"[resolve_student_group_label] 'student_group_aliases' is not a dict: {type(alias_dict).__name__}"
            )
            return card.format.friendly_case(group_key)
        return alias_dict.get(group_key, card.format.friendly_case(group_key))
    except Exception as e:
        LOGGER.warning(
            f"[resolve_student_group_label] Error resolving alias for '{group_key}': {e}"
        )
        return card.format.friendly_case(group_key)


def _coerce_float(value):
    if isinstance(value, bool) or isinstance(value, (list, dict)):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_threshold(value):
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _format_rate(value):
    return f"{float(value) * 100:.1f}%"


def _as_mapping(value):
    if isinstance(value, dict):
        return value
    return {}


def configured_bias_mitigation(card):
    """Return bias mitigation config when a legacy school has set the rule."""
    from edvise.configs.legacy import bias_mitigation_is_configured

    if not bias_mitigation_is_configured(getattr(card, "cfg", None)):
        return None
    return card.cfg.modeling.bias_mitigation


def mitigation_group_col(mitigation) -> str | None:
    col = getattr(mitigation, "student_group_col", None)
    if isinstance(col, str) and col.strip():
        return col.strip()
    return None


def reference_age_display_label(target_group: str, age_bins) -> str | None:
    """Name the complement of an age span, e.g. 18-24 with bins through 65-99 → 25-99.

    The one-student fine age bin ``Other`` is not this complement and is ignored.
    """
    match = _AGE_BIN.fullmatch(str(target_group).strip())
    if match is None:
        return None
    target_end = int(match.group(2))
    latest = None
    for raw in age_bins:
        text = str(raw).strip()
        if text.lower() == _REFERENCE_OTHER:
            continue
        bin_match = _AGE_BIN.fullmatch(text)
        if bin_match is None:
            continue
        hi = int(bin_match.group(2))
        if hi <= target_end:
            continue
        latest = hi if latest is None else max(latest, hi)
    if latest is None:
        return None
    return f"{target_end + 1}-{latest}"


def relabel_comparison(subgroups: str, display_label: str | None) -> str:
    if not display_label:
        return str(subgroups)
    parts = [part.strip() for part in str(subgroups).split("vs")]
    return " vs ".join(
        display_label if part == _REFERENCE_OTHER else part for part in parts
    )


def apply_reference_age_label(df, group_col: str, target_group: str, age_bins):
    """Display-only rename of the mitigation column's ``other`` bucket."""
    label = reference_age_display_label(target_group, age_bins)
    if label is None or group_col not in df.columns:
        return df
    out = df.copy()
    values = out[group_col].astype("string").str.strip()
    out[group_col] = values.mask(values.eq(_REFERENCE_OTHER), label)
    return out


def default_positive_threshold(card):
    """Threshold applied to students outside the mitigated group."""
    inference = getattr(card.cfg, "inference", None)
    if inference is None:
        return _DEFAULT_THRESHOLD
    value = _coerce_float(getattr(inference, "min_prob_pos_label", None))
    if value is None:
        return _DEFAULT_THRESHOLD
    return value


def group_column_label(card, mitigation):
    """Reader-facing name of the group column: the configured alias, else a friendly name."""
    col = getattr(mitigation, "student_group_col", None)
    if not isinstance(col, str) or not col.strip():
        col = _DEFAULT_GROUP_COL
    else:
        col = col.strip()
    return _group_column_alias(card, mitigation, col) or card.format.friendly_case(col)


def reference_students_phrase(display_label):
    """Name the students outside the adjusted subgroup, using the age span when known."""
    if display_label:
        return f"{display_label} students"
    return _ALL_OTHER_STUDENTS


def _sentence_case(text):
    return text[:1].upper() + text[1:]


def _group_column_alias(card, mitigation, col):
    aliases = getattr(card.cfg, "student_group_aliases", None)
    if isinstance(aliases, dict):
        alias = aliases.get(col)
        if isinstance(alias, str) and alias.strip():
            return alias.strip()
    col_alias = getattr(mitigation, "student_group_col_alias", None)
    if isinstance(col_alias, str) and col_alias.strip():
        cleaned = col_alias.strip()
        if cleaned == _DEFAULT_GROUP_COL_ALIAS and col != _DEFAULT_GROUP_COL:
            return None
        return cleaned
    return None


def logged_bias_mitigation_values(card):
    """Read post-mitigation params and metrics from the training run, if present."""
    client = getattr(card, "client", None)
    run_id = getattr(card, "run_id", None)
    if client is None or not run_id:
        return {}
    try:
        run = client.get_run(run_id)
    except Exception as exc:
        LOGGER.warning(
            "Could not read bias mitigation metrics from MLflow run %s: [%s] %s",
            run_id,
            type(exc).__name__,
            exc,
        )
        return {}

    data = getattr(run, "data", None)
    params = _as_mapping(getattr(data, "params", None))
    metrics = _as_mapping(getattr(data, "metrics", None))
    found = {}
    for key in _BIAS_MITIGATION_LOG_KEYS:
        if key in metrics:
            raw = metrics[key]
        elif key in params:
            raw = params[key]
        else:
            continue
        number = _coerce_float(raw)
        if number is None:
            LOGGER.warning("Ignoring non-numeric MLflow value for %s", key)
            continue
        found[key] = number
    return found


def _pre_adjustment_note(card):
    return _PRE_ADJUSTMENT_NOTE.format(
        default_threshold=_format_threshold(default_positive_threshold(card))
    )


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _select_threshold_row(frame, target_threshold):
    """Row for the configured threshold, otherwise the search's chosen row."""
    required = {"threshold", "group_fnr", "reference_fnr", "fnr_abs_gap"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    work = frame.copy()
    work["_threshold"] = pd.to_numeric(work["threshold"], errors="coerce")
    matched = work[(work["_threshold"] - float(target_threshold)).abs() <= 1e-4]
    if not matched.empty:
        return matched.iloc[0]
    if "within_tolerance" in work.columns:
        eligible = work[work["within_tolerance"].map(_as_bool)]
        if not eligible.empty:
            balanced = (
                pd.to_numeric(eligible["balanced_accuracy"], errors="coerce")
                if "balanced_accuracy" in eligible.columns
                else pd.Series(pd.NA, index=eligible.index)
            )
            eligible = eligible.assign(
                _balanced=balanced,
                _gap=pd.to_numeric(eligible["fnr_abs_gap"], errors="coerce"),
            )
            return eligible.sort_values(
                ["_balanced", "_gap"], ascending=[False, True], na_position="last"
            ).iloc[0]
    work = work.assign(_gap=pd.to_numeric(work["fnr_abs_gap"], errors="coerce"))
    return work.sort_values("_gap", ascending=True, na_position="last").iloc[0]


def artifact_bias_mitigation_values(card, mitigation):
    """Read the chosen threshold row from the bias_mitigation search CSV."""
    run_id = getattr(card, "run_id", None)
    if not run_id:
        return {}
    paths = utils.list_paths_in_directory(run_id, _BIAS_MITIGATION_ARTIFACT_DIR)
    csv_paths = sorted(
        path
        for path in paths
        if os.path.basename(path).startswith(_THRESHOLD_ARTIFACT_PREFIX)
        and path.endswith(".csv")
    )
    if not csv_paths:
        return {}
    local_path = utils.download_artifact(
        run_id=run_id,
        local_folder=getattr(card, "assets_folder", None),
        artifact_path=csv_paths[-1],
    )
    if not isinstance(local_path, str) or not os.path.isfile(local_path):
        LOGGER.warning(
            "Bias mitigation threshold artifact was listed but could not be read: %s",
            csv_paths[-1],
        )
        return {}
    try:
        frame = pd.read_csv(local_path)
    except Exception as exc:
        LOGGER.warning(
            "Could not read bias mitigation threshold artifact %s: [%s] %s",
            csv_paths[-1],
            type(exc).__name__,
            exc,
        )
        return {}
    row = _select_threshold_row(frame, float(mitigation.custom_threshold))
    if row is None:
        return {}
    found = {}
    for column, key in _ARTIFACT_COLUMNS.items():
        number = _coerce_float(row[column])
        if number is None:
            LOGGER.warning(
                "Ignoring non-numeric bias mitigation artifact value for %s", column
            )
            continue
        found[key] = number
    return found


def render_post_mitigation_section(card, mitigation, display_label=None):
    """Explain the group-specific threshold and any logged FNR comparison."""
    target_group = str(mitigation.student_group).strip()
    target_threshold = float(mitigation.custom_threshold)
    default_threshold = default_positive_threshold(card)
    reference = reference_students_phrase(display_label)
    logged = logged_bias_mitigation_values(card)
    if not all(key in logged for key in _FNR_COMPARISON_KEYS):
        for key, value in artifact_bias_mitigation_values(card, mitigation).items():
            logged.setdefault(key, value)

    bold = card.format.bold
    group_label = group_column_label(card, mitigation)
    lines = [
        f"{card.format.header_level(4)}Fairness Adjustment\n",
        (
            "This model uses a group-specific threshold so that underprediction is "
            f"more even across the subgroups of {group_label}. "
            "Every student gets the same Support Score. What differs is the score at "
            "which a student is identified as likely to need support: "
            f"{bold(_format_threshold(target_threshold))} for "
            f"{bold(target_group)} students and {bold(_format_threshold(default_threshold))} "
            f"for {reference}.\n"
        ),
        f"- Student group: {bold(group_label)}",
        f"- Subgroup with a fairness-adjusted threshold: {bold(target_group)}",
        f"- Threshold for this subgroup: {bold(_format_threshold(target_threshold))}",
        (f"- Threshold for {reference}: {bold(_format_threshold(default_threshold))}"),
        "",
    ]

    if all(key in logged for key in _FNR_COMPARISON_KEYS):
        gap_points = f"{logged['bias_mitigation_fnr_abs_gap'] * 100:.1f}"
        lines.append(
            f"{bold('Effect on results.')} After the adjustment, the model "
            f"underpredicts for {bold(_format_rate(logged['bias_mitigation_group_fnr']))} "
            f"of {target_group} students who need support, compared with "
            f"{bold(_format_rate(logged['bias_mitigation_reference_fnr']))} of "
            f"{reference}, a gap of {bold(gap_points + ' percentage points')}."
        )
    else:
        lines.append(card.format.italic(_ADJUSTMENT_RESULTS_UNAVAILABLE))

    if target_threshold < default_threshold:
        lines.extend(
            [
                "",
                f"{bold('Tradeoff.')} A lower threshold reduces underprediction for "
                "this group but can increase overprediction: some "
                f"{target_group} students will be identified as likely to need "
                "support who ultimately do not.",
            ]
        )

    lines.extend(["", _SUPPORT_LABEL_NOTE])
    return "\n".join(lines)


def _usable_plot(value):
    if not isinstance(value, str) or not value.strip():
        return None
    if value.lstrip().startswith("<"):
        return value
    if os.path.isfile(value):
        return value
    return None


def _describe_disparity(card, subgroups, diff, stat_summary):
    try:
        subgroup_1, subgroup_2 = [s.strip() for s in str(subgroups).split("vs")]
    except ValueError:
        return f"{card.format.bold('Could not parse subgroup comparison')}"
    sg1 = card.format.bold(card.format.italic(subgroup_1))
    sg2 = card.format.bold(card.format.italic(subgroup_2))
    percent_higher = card.format.bold(f"{int(round(float(diff) * 100))}% difference")
    return (
        f"- {sg1} students have a {percent_higher} in False Negative Rate (FNR) than {sg2} students. "
        f"Statistical analysis indicates {stat_summary}."
    )


def _post_flag_artifact_paths(flags_dir, level):
    """Canonical flag name, then the prefixed name from earlier postprocessing runs."""
    prefix = os.path.basename(flags_dir.rstrip("/"))
    names = [f"{level}_bias_flags.csv"]
    if prefix:
        names.append(f"{prefix}_{level}_bias_flags.csv")
    return [f"{flags_dir}/{name}" for name in names]


def _first_existing_artifact(card, artifact_paths):
    for artifact_path in artifact_paths:
        local_path = utils.download_artifact(
            run_id=card.run_id,
            local_folder=card.assets_folder,
            artifact_path=artifact_path,
        )
        if isinstance(local_path, str) and os.path.isfile(local_path):
            return local_path
    return None


def _age_bins_from_rates(card) -> list[str]:
    local_path = _first_existing_artifact(card, [_POST_FNR_RATES])
    if local_path is None:
        return []
    try:
        frame = pd.read_csv(local_path)
    except Exception as exc:
        LOGGER.warning(
            "Could not read post-mitigation FNR rates: [%s] %s",
            type(exc).__name__,
            exc,
        )
        return []
    if frame.empty or "subgroup" not in frame.columns:
        return []
    return [str(value) for value in frame["subgroup"].dropna().tolist()]


def _load_flag_descriptions(card, flags_dir, mitigation_col=None, display_label=None):
    descriptions_by_group = {}
    for level in ("high", "moderate", "low"):
        local_path = _first_existing_artifact(
            card, _post_flag_artifact_paths(flags_dir, level)
        )
        if local_path is None:
            continue
        try:
            frame = pd.read_csv(local_path)
        except Exception as exc:
            LOGGER.warning(
                "Could not read %s bias flags: [%s] %s",
                level,
                type(exc).__name__,
                exc,
            )
            continue
        if frame.empty or "split_name" not in frame.columns:
            continue
        frame = frame[frame["split_name"] == "test"]
        for _, row in frame.iterrows():
            group = row["group"]
            subgroups = row["subgroups"]
            if display_label and group == mitigation_col:
                subgroups = relabel_comparison(subgroups, display_label)
            descriptions_by_group.setdefault(group, []).append(
                _describe_disparity(
                    card,
                    subgroups,
                    row["fnr_percentage_difference"],
                    row["type"],
                )
            )
    return descriptions_by_group


def _post_group_order(card, mitigation, descriptions_by_group):
    ordered = []
    mitigation_col = getattr(mitigation, "student_group_col", None)
    configured = []
    if isinstance(mitigation_col, str) and mitigation_col.strip():
        configured.append(mitigation_col.strip())
    configured.extend(getattr(card.cfg, "student_group_cols", None) or [])
    for group in [*configured, *descriptions_by_group]:
        if isinstance(group, str) and group and group not in ordered:
            ordered.append(group)
    return ordered


def render_post_mitigation_disparities(card, mitigation):
    """Plots for every group after the fairness adjustment, with a disparity sentence only when flagged."""
    mitigation_col = mitigation_group_col(mitigation)
    display_label = reference_age_display_label(
        str(mitigation.student_group), _age_bins_from_rates(card)
    )
    descriptions_by_group = _load_flag_descriptions(
        card,
        _POST_FLAGS_DIR,
        mitigation_col=mitigation_col,
        display_label=display_label,
    )
    blocks = []
    for group in _post_group_order(card, mitigation, descriptions_by_group):
        label = resolve_student_group_label(card, group)
        normalized_name = group.lower().replace(" ", "_")
        plot_md = _usable_plot(
            utils.download_artifact(
                run_id=card.run_id,
                local_folder=card.assets_folder,
                artifact_path=f"{_POST_PLOTS_DIR}/test_{normalized_name}_fnr.png",
                description=f"False Negative Parity Rate for {label} after the fairness adjustment",
                caption=f"FNR Parity for {label} after the fairness adjustment",
            )
        )
        descriptions = descriptions_by_group.get(group, [])
        if not descriptions and plot_md is None:
            continue
        header = f"{card.format.header_level(5)}{label}\n\n"
        text = "\n\n".join(descriptions)
        plot_text = f"\n\n{plot_md}" if plot_md else ""
        if text:
            blocks.append(header + text + plot_text)
        else:
            blocks.append(header + (plot_md or ""))
    if not blocks:
        return ""
    target = str(mitigation.student_group).strip()
    note = _POST_GROUP_NOTE.format(
        target=target,
        group_threshold=_format_threshold(mitigation.custom_threshold),
        reference=_sentence_case(reference_students_phrase(display_label)),
        default_threshold=_format_threshold(default_positive_threshold(card)),
    )
    return (
        f"\n{card.format.header_level(4)}Disparities After the Fairness Adjustment\n\n"
        f"{note}\n\n" + "\n\n".join(blocks)
    )


def register_bias_sections(card, registry):
    def group_label_fn(col: str) -> str:
        return str(card.format.friendly_case(resolve_student_group_label(card, col)))

    # Register base sections (bias_groups_section uses aliases via group_label_fn)
    base_bias_sections.register_bias_sections(
        card, registry, group_label_fn=group_label_fn
    )
    for index, (key, fn) in enumerate(registry._sections):
        if key != "bias_groups_section":
            continue

        def _with_pre_mitigation_note(fn=fn):
            mitigation = configured_bias_mitigation(card)
            if mitigation is None:
                return fn()
            intro = (
                f"{card.format.indent_level(1)}- Our assessment for FNR Parity was "
                "conducted across the following student groups.\n"
            )
            note = f"{card.format.indent_level(1)}- {_pre_adjustment_note(card)}\n"
            col = mitigation_group_col(mitigation)
            if col is None:
                return intro + note
            label = group_label_fn(col)
            return intro + note + f"{card.format.indent_level(2)}- {label}\n"

        registry._sections[index] = (key, _with_pre_mitigation_note)
        break

    mitigation = configured_bias_mitigation(card)
    mitigation_col = mitigation_group_col(mitigation) if mitigation else None
    display_label = (
        reference_age_display_label(
            str(mitigation.student_group), _age_bins_from_rates(card)
        )
        if mitigation is not None
        else None
    )
    bias_levels = ["high", "moderate", "low"]
    group_disparities = {}

    def generate_description(group, subgroups, diff, stat_summary):
        try:
            subgroup_1, subgroup_2 = [s.strip() for s in subgroups.split("vs")]
        except ValueError:
            LOGGER.warning(f"Could not parse subgroups for {group}: {subgroups}")
            return f"{card.format.bold('Could not parse subgroup comparison')}"

        sg1 = card.format.bold(card.format.italic(subgroup_1))
        sg2 = card.format.bold(card.format.italic(subgroup_2))
        percent_higher = card.format.bold(
            f"{int(round(float(diff) * 100))}% difference"
        )

        return (
            f"- {sg1} students have a {percent_higher} in False Negative Rate (FNR) than {sg2} students. "
            f"Statistical analysis indicates {stat_summary}."
        )

    # Load bias flag CSVs and filter for test split
    for level in bias_levels:
        try:
            bias_path = f"bias_flags/{level}_bias_flags.csv"
            local_path = utils.download_artifact(
                run_id=card.run_id,
                local_folder=card.assets_folder,
                artifact_path=bias_path,
            )

            if not os.path.exists(local_path):
                LOGGER.warning(
                    f"{level} bias flags file does not exist. Bias evaluation likely has not run."
                )
                continue

            df = pd.read_csv(local_path)

            if df.empty or "split_name" not in df.columns:
                LOGGER.warning(
                    f"{level} bias flags file exists but has no data or missing 'split_name' column."
                )
                continue

            df = df[df["split_name"] == "test"]
            if df.empty:
                LOGGER.info(f"{level} bias flags file has no rows for test split.")
                continue

            for _, row in df.iterrows():
                group = row["group"]
                if mitigation_col is not None and group != mitigation_col:
                    continue
                subgroups = row["subgroups"]
                if display_label and group == mitigation_col:
                    subgroups = relabel_comparison(subgroups, display_label)
                desc = generate_description(
                    group,
                    subgroups,
                    row["fnr_percentage_difference"],
                    row["type"],
                )
                group_disparities.setdefault(group, []).append(desc)

        except Exception as e:
            LOGGER.warning(
                f"Could not load {level} bias flags: [{type(e).__name__}] {str(e)}"
            )

    # Build bias summary content using student group aliases
    all_blocks = []

    for group_name, descriptions in group_disparities.items():
        label = resolve_student_group_label(card, group_name)
        normalized_name = group_name.lower().replace(" ", "_")

        plot_md = None
        if mitigation_col is not None and group_name == mitigation_col:
            plot_md = _usable_plot(
                utils.download_artifact(
                    run_id=card.run_id,
                    local_folder=card.assets_folder,
                    artifact_path=(
                        f"{_PRE_DISPLAY_PLOTS_DIR}/test_{normalized_name}_fnr.png"
                    ),
                    description=f"False Negative Parity Rate for {label} on Test Data",
                    caption=f"FNR Parity for {label} on Test Data",
                )
            )
        if plot_md is None:
            plot_artifact_path = f"fnr_plots/test_{normalized_name}_fnr.png"
            try:
                plot_md = utils.download_artifact(
                    run_id=card.run_id,
                    local_folder=card.assets_folder,
                    artifact_path=plot_artifact_path,
                    description=f"False Negative Parity Rate for {label} on Test Data",
                    caption=f"FNR Parity for {label} on Test Data",
                )
            except Exception as e:
                LOGGER.warning(f"Could not load plot for {group_name}: {str(e)}")
                plot_md = (
                    f"{card.format.bold(f'Unable to retrieve plot for {label}')}\n"
                )

        header = f"{card.format.header_level(5)}{label}\n\n"
        text_block = "\n\n".join(descriptions)
        all_blocks.append(header + text_block + "\n\n" + plot_md)

    @registry.register("bias_summary_section")
    def bias_summary_section():
        """
        Returns a markdown string containing the bias summary section of the model card.
        """
        no_disparities = card.format.italic(
            "No statistically significant disparities were found on test dataset across groups."
        )
        mitigation = configured_bias_mitigation(card)
        if not all_blocks and mitigation is None:
            LOGGER.warning(
                "No disparities found or bias evaluation artifacts missing. Skipping bias summary section."
            )
            return no_disparities
        if not all_blocks:
            LOGGER.warning(
                "No disparities found or bias evaluation artifacts missing. "
                "Documenting the configured dual-threshold rule without a pre-mitigation disparity list."
            )
            pre_mitigation = (
                f"\n{card.format.header_level(4)}Disparities Before the Fairness Adjustment\n\n"
                f"{_pre_adjustment_note(card)}\n\n"
                f"{no_disparities}"
            )
            return (
                pre_mitigation
                + "\n\n"
                + render_post_mitigation_disparities(card, mitigation)
                + "\n\n"
                + render_post_mitigation_section(card, mitigation, display_label)
            )

        if mitigation is None:
            section_header = (
                f"\n{card.format.header_level(4)}Disparities by Student Group\n\n"
            )
            return section_header + "\n\n".join(all_blocks)

        section_header = (
            f"\n{card.format.header_level(4)}Disparities Before the Fairness Adjustment\n\n"
            f"{_pre_adjustment_note(card)}\n\n"
        )
        return (
            section_header
            + "\n\n".join(all_blocks)
            + "\n\n"
            + render_post_mitigation_disparities(card, mitigation)
            + "\n\n"
            + render_post_mitigation_section(card, mitigation, display_label)
        )
