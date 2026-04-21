"""
Streamlit UI: review GenAI HITL JSON files on Unity Catalog volumes.

Loads ``identity_*_hitl.json`` or ``sma_hitl*.json``, shows questions and options,
writes reviewer ``choice`` (1-based index) back to the same path.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from helpers import (
    catalog_from_env,
    detect_envelope_kind,
    hitl_volume_path,
    read_volume_json,
    suggested_relative_path,
    write_volume_json,
)

st.set_page_config(
    page_title="GenAI HITL review",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _option_labels_for_item(item: dict) -> list[str]:
    opts = item.get("options") or []
    out: list[str] = []
    for j, o in enumerate(opts):
        lab = o.get("label", "")
        desc = o.get("description", "")
        out.append(f"{j + 1}. {lab} — {desc}")
    return out


def _selected_is_sma_direct_edit(item: dict, choice_one_based: int) -> bool:
    opts = item.get("options") or []
    if choice_one_based < 1 or choice_one_based > len(opts):
        return False
    o = opts[choice_one_based - 1]
    return o.get("option_id") == "direct_edit" or o.get("reentry") == "direct_edit"


def main() -> None:
    st.title("GenAI HITL review")
    st.caption(
        "Open a HITL JSON file on a UC volume, pick an option per item, save. "
        "Uses the Databricks Files API (same auth as other workspace tools)."
    )

    default_cat = catalog_from_env()
    with st.sidebar:
        st.subheader("Location")
        catalog = st.text_input("UC catalog", value=default_cat)
        institution_id = st.text_input("Institution ID")
        hitl_file = st.selectbox(
            "HITL file",
            options=["identity_grain", "identity_term", "sma"],
            format_func=lambda x: {
                "identity_grain": "Identity — grain (identity_grain_hitl.json)",
                "identity_term": "Identity — term (identity_term_hitl.json)",
                "sma": "SMA (sma_hitl.json)",
            }[x],
        )
        _rel_key = f"bronze_rel_{hitl_file}"
        if _rel_key not in st.session_state:
            st.session_state[_rel_key] = suggested_relative_path(hitl_file)
        st.text_input(
            "Path under bronze volume",
            key=_rel_key,
            help=(
                "Relative to …/<institution>_bronze/bronze_volume/. "
                "Edit to match where the JSON file lives (e.g. add a folder under genai_pipeline/)."
            ),
        )
        full_path_override = st.text_input(
            "Or full /Volumes/... path",
            help="If set, ignores catalog, institution, and path-under-bronze above.",
        )

        load = st.button("Load", type="primary")

    if "hitl_data" not in st.session_state:
        st.session_state.hitl_data = None
        st.session_state.hitl_path = None
        st.session_state.hitl_kind = None
        st.session_state.hitl_widget_gen = 0

    if load:
        try:
            if full_path_override.strip():
                path = full_path_override.strip()
            else:
                if not institution_id.strip():
                    st.error("Institution ID is required (or use a full /Volumes/... path).")
                    st.stop()
                rel = str(st.session_state.get(_rel_key, "")).strip()
                if not rel:
                    st.error("Set **Path under bronze volume** (or use a full /Volumes/... path).")
                    st.stop()
                path = hitl_volume_path(
                    catalog=catalog,
                    institution_id=institution_id,
                    relative_path_under_bronze=rel,
                )
            data = read_volume_json(path)
            kind = detect_envelope_kind(data)
            st.session_state.hitl_data = data
            st.session_state.hitl_path = path
            st.session_state.hitl_kind = kind
            st.session_state.hitl_widget_gen = (
                int(st.session_state.get("hitl_widget_gen", 0)) + 1
            )
        except Exception as exc:
            st.error(f"Could not load file: {exc}")
            st.stop()

    if st.session_state.hitl_data is None:
        st.info("Configure the sidebar and click **Load**.")
        return

    data = st.session_state.hitl_data
    path = st.session_state.hitl_path
    kind = st.session_state.hitl_kind

    st.success(f"Loaded `{path}` (envelope: **{kind}**).")

    items = data.get("items") or []
    if not items:
        st.warning("No HITL items in this file — nothing to review.")
        return

    wg = int(st.session_state.get("hitl_widget_gen", 0))

    with st.form("hitl_review"):
        new_notes: list[str | None] = []
        new_choices: list[int | None] = []
        new_de_maps: list[dict | None] = []

        for idx, item in enumerate(items):
            iid = item.get("item_id", f"item_{idx}")
            st.divider()
            st.markdown(f"#### `{iid}`")
            st.markdown(item.get("hitl_question", ""))
            ctx = item.get("hitl_context")
            if ctx is not None:
                with st.expander("Context"):
                    if isinstance(ctx, (dict, list)):
                        st.json(ctx)
                    else:
                        st.text(str(ctx))

            opts = item.get("options") or []
            if not opts:
                st.error("Item has no options — fix the JSON.")
                new_choices.append(None)
                new_notes.append(None)
                new_de_maps.append(None)
                continue

            labels = _option_labels_for_item(item)
            cur = item.get("choice")
            pre = (cur - 1) if isinstance(cur, int) else None

            rk = f"choice_{wg}_{idx}"
            radio_kw: dict[str, Any] = {
                "label": "Resolution",
                "options": list(range(len(opts))),
                "format_func": lambda i, lab=labels: lab[i],
                "key": rk,
                "label_visibility": "visible",
            }
            if pre is not None:
                radio_kw["index"] = pre
            choice_i = st.radio(**radio_kw)
            new_choices.append(choice_i + 1)

            nk = f"note_{wg}_{idx}"
            note = st.text_input(
                "Reviewer note (optional)",
                value=item.get("reviewer_note") or "",
                key=nk,
            )
            new_notes.append(note if note.strip() else None)

            if kind == "sma":
                existing = item.get("direct_edit_field_mapping")
                if isinstance(existing, dict):
                    de_val = json.dumps(existing, indent=2, ensure_ascii=False)
                elif isinstance(existing, str):
                    de_val = existing
                else:
                    de_val = ""
                st.caption(
                    "If you select **direct edit**, paste a valid `FieldMappingRecord` JSON object."
                )
                dk = f"de_{wg}_{idx}"
                de_text = st.text_area(
                    "direct_edit_field_mapping (JSON object, required for direct edit)",
                    value=de_val,
                    height=160,
                    key=dk,
                )
                try:
                    parsed = json.loads(de_text) if de_text.strip() else None
                    if parsed is not None and not isinstance(parsed, dict):
                        st.warning("direct_edit_field_mapping must be a JSON object.")
                        new_de_maps.append(None)
                    else:
                        new_de_maps.append(parsed)
                except json.JSONDecodeError:
                    st.warning("Invalid JSON in direct_edit_field_mapping.")
                    new_de_maps.append(None)
            else:
                new_de_maps.append(None)

        submitted = st.form_submit_button("Save to volume")

    if submitted:
        try:
            for idx, item in enumerate(items):
                ch = new_choices[idx]
                if ch is None:
                    continue
                item["choice"] = ch
                item["reviewer_note"] = new_notes[idx]
                if kind == "sma":
                    if _selected_is_sma_direct_edit(item, ch):
                        dm = new_de_maps[idx]
                        if not dm:
                            st.error(
                                f"Item `{item.get('item_id')}`: direct_edit requires "
                                "a JSON object in direct_edit_field_mapping."
                            )
                            st.stop()
                        item["direct_edit_field_mapping"] = dm
                    else:
                        item["direct_edit_field_mapping"] = None

            write_volume_json(path, data)
            st.session_state.hitl_data = data
            st.success("Saved.")
        except Exception as exc:
            st.error(f"Save failed: {exc}")


main()
