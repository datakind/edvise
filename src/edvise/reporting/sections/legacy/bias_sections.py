import os
import logging
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
_PRE_MITIGATION_NOTE = (
    "This is the pre-mitigation training audit at the default threshold, "
    "using one threshold for every student."
)
_METRICS_NOT_LOGGED = "Post-mitigation metrics were not logged."
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


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    modeling = getattr(card.cfg, "modeling", None)
    if modeling is None:
        return None
    mitigation = getattr(modeling, "bias_mitigation", None)
    if mitigation is None:
        return None
    group = getattr(mitigation, "student_group", None)
    threshold = getattr(mitigation, "custom_threshold", None)
    if isinstance(group, str) and group.strip() and _is_number(threshold):
        return mitigation
    return None


def default_positive_threshold(card):
    """Threshold applied to students outside the mitigated group."""
    inference = getattr(card.cfg, "inference", None)
    if inference is None:
        return _DEFAULT_THRESHOLD
    value = _coerce_float(getattr(inference, "min_prob_pos_label", None))
    if value is None:
        return _DEFAULT_THRESHOLD
    return value


def group_column_phrase(card, mitigation):
    """Group column plus a display alias when one is configured."""
    col = getattr(mitigation, "student_group_col", None)
    if not isinstance(col, str) or not col.strip():
        col = _DEFAULT_GROUP_COL
    else:
        col = col.strip()
    alias = _group_column_alias(card, mitigation, col)
    if alias:
        return f"{card.format.bold(alias)} (`{col}`)"
    return f"`{col}`"


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


def render_post_mitigation_section(card, mitigation):
    """Document the configured dual-threshold rule and any logged FNR comparison."""
    target_group = str(mitigation.student_group).strip()
    target_threshold = float(mitigation.custom_threshold)
    default_threshold = default_positive_threshold(card)
    logged = logged_bias_mitigation_values(card)

    lines = [
        f"{card.format.header_level(4)}Post-Mitigation Dual Threshold\n",
        (
            "Advisor output applies a dual threshold after training. "
            "Support scores stay the same.\n"
        ),
        f"- Group column: {group_column_phrase(card, mitigation)}",
        f"- Target group: {card.format.bold(target_group)}",
        f"- Target threshold: {card.format.bold(_format_threshold(target_threshold))}",
        (
            "- Default threshold: "
            f"{card.format.bold(_format_threshold(default_threshold))}"
        ),
        "",
    ]

    if all(key in logged for key in _FNR_COMPARISON_KEYS):
        lines.append(
            "Post-mitigation FNR comparison for the target group versus everyone else:\n"
        )
        if "bias_mitigation_custom_threshold" in logged:
            lines.append(
                "- Logged target threshold: "
                f"{card.format.bold(_format_threshold(logged['bias_mitigation_custom_threshold']))}"
            )
        lines.append(
            "- Target group FNR: "
            f"{card.format.bold(_format_rate(logged['bias_mitigation_group_fnr']))}"
        )
        lines.append(
            "- Everyone else FNR: "
            f"{card.format.bold(_format_rate(logged['bias_mitigation_reference_fnr']))}"
        )
        lines.append(
            "- Absolute FNR gap: "
            f"{card.format.bold(_format_rate(logged['bias_mitigation_fnr_abs_gap']))}"
        )
    else:
        lines.append(_METRICS_NOT_LOGGED)

    lines.extend(["", _SUPPORT_LABEL_NOTE])
    return "\n".join(lines)


def register_bias_sections(card, registry):
    # Register base sections
    base_bias_sections.register_bias_sections(card, registry)

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
                desc = generate_description(
                    group,
                    row["subgroups"],
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
            plot_md = f"{card.format.bold(f'Unable to retrieve plot for {label}')}\n"

        header = f"{card.format.header_level(5)}{label}\n\n"
        text_block = "\n\n".join(descriptions)
        all_blocks.append(header + text_block + "\n\n" + plot_md)

    @registry.register("bias_groups_section")
    def bias_groups_section():
        """
        Returns bias groups for legacy schools using aliases.
        """
        intro = f"{card.format.indent_level(1)}- Our assessment for FNR Parity was conducted across the following student groups.\n"
        if configured_bias_mitigation(card) is not None:
            intro += f"{card.format.indent_level(1)}- {_PRE_MITIGATION_NOTE}\n"

        try:
            alias_dict = card.cfg.student_group_aliases
            assert isinstance(alias_dict, dict), (
                "student_group_aliases must be a dictionary"
            )

            group_labels = list(alias_dict.values())
            nested = [
                f"{card.format.indent_level(2)}- {card.format.friendly_case(label)}\n"
                for label in group_labels
            ]
            return intro + "".join(nested)

        except (AttributeError, AssertionError, TypeError) as e:
            LOGGER.warning(
                f"[bias_groups_section] Failed to extract student groups: {e}"
            )
            fallback = (
                f"{card.format.indent_level(2)}- Unable to extract student groups\n"
            )
            return intro + fallback

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
                f"\n{card.format.header_level(4)}Pre-Mitigation Disparities by Student Group\n\n"
                f"{_PRE_MITIGATION_NOTE}\n\n"
                f"{no_disparities}"
            )
            return (
                pre_mitigation
                + "\n\n"
                + render_post_mitigation_section(card, mitigation)
            )

        if mitigation is None:
            section_header = (
                f"\n{card.format.header_level(4)}Disparities by Student Group\n\n"
            )
            return section_header + "\n\n".join(all_blocks)

        section_header = (
            f"\n{card.format.header_level(4)}Pre-Mitigation Disparities by Student Group\n\n"
            f"{_PRE_MITIGATION_NOTE}\n\n"
        )
        return (
            section_header
            + "\n\n".join(all_blocks)
            + "\n\n"
            + render_post_mitigation_section(card, mitigation)
        )
