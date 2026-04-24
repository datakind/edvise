"""
GenAI mapping — Unity Catalog ``hitl_reviews`` reviewer UI.

HITL **choice** values live in JSON on the **silver** volume under
``{silver_genai_mapping_root}/runs/...`` (see ``edvise_genai_ia`` / ``edvise_genai_sma``
``resolve_run_paths``). Paths come from ``hitl_reviews.artifact_path``. **IA grain** and **SMA**
manifests use **Save JSON & approve UC** (shared batch write + optional UC approve). Other phases
keep **Approve UC** / **Reject UC** under each run group. Unity Catalog: ``{catalog}.genai_mapping.hitl_reviews``.

**Local run** (repo root: ``uv pip install -e .`` or regenerate ``requirements.txt`` with
``python generate_requirements.py`` so ``edvise`` is installable; then from this directory,
with Databricks auth configured, e.g. ``databricks auth login``). Shared code lives in
``hitl_reviewer/`` next to ``app.py``::

    export DATABRICKS_WAREHOUSE_ID=<sql-warehouse-id>
    export GENAI_HITL_CATALOG=dev_sst_02   # optional; default in sidebar
    streamlit run app.py

**Databricks Apps (dev):** run ``databricks bundle deploy`` / ``databricks bundle run`` from
this directory (see ``databricks.yml``) or trigger CI
``.github/workflows/deploy-genai-hitl-app.yml`` (Actions → “Deploy GenAI HITL app (dev)”).
Use the same ``--var sql_warehouse_id=…`` and ``--var datakind_group_to_manage_workflow=…``
as the metadata dashboard app for dev.

The Databricks app identity (service principal) needs **read/write** on the silver
``/Volumes/.../..._silver/.../genai_mapping/...`` HITL paths, in addition to SQL access.
"""

from __future__ import annotations

import html
import json
import os
import re
import pandas as pd
import streamlit as st

from edvise.utils.institution_naming import format_institution_display_name
from hitl_reviewer.databricks_uc_sql import (
    approve_or_reject,
    get_warehouse_id,
    hitl_reviews_fqn,
    pipeline_runs_fqn,
    run_query,
    sql_str,
)
from hitl_reviewer.hitl_json_batch_commit import (
    persist_hitl_choice_radios_from_session,
    try_approve_uc_after_json_write,
)
from hitl_reviewer.ia.grain_review_ui import is_ia_grain_phase, render_ia_grain_hitl_cards
from hitl_reviewer.silver_hitl_paths import (
    artifact_path_contains_onboard_run_id,
)
from hitl_reviewer.sma.enriched_schema_contract import (
    enriched_schema_contract_path_from_manifest,
    extract_column_panel_fields,
    load_json_object_from_text,
    silver_relative_path,
)
from hitl_reviewer.unity_volume_files import read_unity_file_text, write_unity_file_text


def load_hitl_rows(
    catalog: str,
    *,
    onboard_run_id: str | None,
    phase: str | None,
    status: str | None,
    limit: int,
) -> pd.DataFrame:
    t_h = hitl_reviews_fqn(catalog)
    t_p = pipeline_runs_fqn(catalog)
    where: list[str] = []
    c_sql = sql_str(str(catalog).strip())
    if (onboard_run_id or "").strip():
        where.append(f"h.onboard_run_id = {sql_str(onboard_run_id.strip())}")
    if (phase or "").strip():
        where.append(f"h.phase = {sql_str(phase.strip())}")
    if (status or "").strip():
        where.append(f"h.status = {sql_str(status.strip())}")
    w = f"WHERE {' AND '.join(where)}" if where else ""
    lim = max(1, min(int(limit), 5000))
    q = f"""
    SELECT
      h.onboard_run_id,
      h.phase,
      h.artifact_type,
      h.artifact_path,
      h.status,
      h.reviewer,
      h.reviewed_at,
      p.institution_id
    FROM {t_h} h
    LEFT JOIN {t_p} p
      ON h.onboard_run_id = p.onboard_run_id
     AND p.`catalog` = {c_sql}
    {w}
    ORDER BY h.reviewed_at DESC NULLS FIRST, h.onboard_run_id, h.phase, h.artifact_type, h.artifact_path
    LIMIT {lim}
    """
    return run_query(q)


def _safe_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(s))[:80]


def _is_sma_hitl_context(phase: str, artifact_type: str) -> bool:
    return str(phase).strip().lower() == "sma_gate_1" and str(artifact_type).strip().lower() in (
        "cohort_manifest",
        "course_manifest",
    )


def _invalidate_sma_run_pending_cache(onboard_run_id: str) -> None:
    rid = str(onboard_run_id).strip()
    for suffix in ("total", "order"):
        k = f"sma-run-pending-{suffix}-{rid}"
        if k in st.session_state:
            del st.session_state[k]


def _sma_run_pending_ordered_pairs(
    pending_df: pd.DataFrame, onboard_run_id: str
) -> list[tuple[str, int]]:
    """
    Pending SMA HITL items (``choice`` unset, has ``options``) across manifests for the run.

    Order: ``cohort_manifest`` artifact rows first, then ``course_manifest``, each file's
    ``items`` in list order. Used for run-wide ``X of Y pending``.
    """
    rid = str(onboard_run_id).strip()
    cache_key = f"sma-run-pending-order-{rid}"
    if cache_key in st.session_state:
        return list(st.session_state[cache_key])
    sub = pending_df[
        (pending_df["onboard_run_id"].astype(str) == rid)
        & (pending_df["phase"].astype(str).str.lower() == "sma_gate_1")
        & (
            pending_df["artifact_type"]
            .astype(str)
            .str.lower()
            .isin(["cohort_manifest", "course_manifest"])
        )
    ]
    ordered: list[tuple[str, int]] = []
    for at in ("cohort_manifest", "course_manifest"):
        rows = sub[sub["artifact_type"].astype(str).str.lower() == at]
        if rows.empty:
            continue
        path = str(rows.iloc[0]["artifact_path"]).strip()
        if not path:
            continue
        try:
            raw = read_unity_file_text(path)
            data = json.loads(raw)
        except Exception:
            continue
        items = data.get("items")
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            opts = item.get("options")
            if not opts or not isinstance(opts, list) or len(opts) < 1:
                continue
            if item.get("choice") is None:
                ordered.append((path, idx))
    st.session_state[cache_key] = ordered
    st.session_state[f"sma-run-pending-total-{rid}"] = len(ordered)
    return ordered


def _inject_sma_hitl_css_once() -> None:
    if st.session_state.get("_hitl_sma_css"):
        return
    st.markdown(
        """
<style>
.hitl-sma-inst { font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin: 0 0 0.35rem 0;
  letter-spacing: -0.02em; }
.hitl-sma-q { font-size: 1.22rem; line-height: 1.55; margin: 1.35rem 0 1rem 0; font-weight: 500; }
.hitl-sma-meta { font-size: 0.9rem; color: rgba(49, 51, 63, 0.75); margin-bottom: 0.25rem; }
.hitl-chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.25rem; }
.hitl-chip { display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px; font-size: 0.78rem;
  background: rgba(111, 66, 193, 0.12); border: 1px solid rgba(111, 66, 193, 0.28); }
.hitl-sma-ctx-block { font-size: 0.95rem; line-height: 1.45; color: rgba(49, 51, 63, 0.92); }
</style>
""",
        unsafe_allow_html=True,
    )
    st.session_state["_hitl_sma_css"] = True


def _render_sma_review_context(*, item: dict) -> None:
    """``hitl_context``, ``validation_errors``, and ``current_field_mapping`` rationale/notes."""
    ctx = (item.get("hitl_context") or "").strip()
    raw_errs = item.get("validation_errors")
    err_lines: list[str] = []
    if isinstance(raw_errs, list):
        err_lines = [str(e).strip() for e in raw_errs if str(e).strip()]

    cfm = item.get("current_field_mapping")
    rationale = ""
    val_notes = ""
    if isinstance(cfm, dict):
        rationale = (str(cfm.get("rationale") or "")).strip()
        val_notes = (str(cfm.get("validation_notes") or "")).strip()

    if not ctx and not err_lines and not rationale and not val_notes:
        return
    with st.expander("Review context", expanded=True):
        if ctx:
            st.markdown('<p class="hitl-sma-ctx-block"><strong>Evidence</strong></p>', unsafe_allow_html=True)
            st.text(ctx)
        if err_lines:
            st.markdown('<p class="hitl-sma-ctx-block"><strong>Validation issues</strong></p>', unsafe_allow_html=True)
            st.markdown(
                '<ul class="hitl-sma-ctx-block">'
                + "".join(f"<li>{html.escape(e)}</li>" for e in err_lines)
                + "</ul>",
                unsafe_allow_html=True,
            )
        if rationale:
            st.markdown(
                '<p class="hitl-sma-ctx-block"><strong>Model rationale</strong> (current mapping)</p>',
                unsafe_allow_html=True,
            )
            st.text(rationale)
        if val_notes:
            st.markdown(
                '<p class="hitl-sma-ctx-block"><strong>Predicted validation notes</strong></p>',
                unsafe_allow_html=True,
            )
            st.text(val_notes)


def _render_sma_option_descriptions(*, options: list) -> None:
    """Per-option ``description`` from the HITL JSON (read before choosing)."""
    entries: list[tuple[int, str, str]] = []
    for j, opt in enumerate(options):
        if not isinstance(opt, dict):
            continue
        lab = (str(opt.get("label") or f"Option {j + 1}")).strip()
        desc = (str(opt.get("description") or "")).strip()
        entries.append((j + 1, lab, desc))
    if not entries:
        return
    has_any_desc = any(d for _, _, d in entries)
    if not has_any_desc:
        return
    with st.expander("What each option means", expanded=True):
        for num, lab, desc in entries:
            st.markdown(
                f'<p class="hitl-sma-ctx-block"><strong>{num}. {html.escape(lab)}</strong></p>',
                unsafe_allow_html=True,
            )
            if desc:
                st.text(desc)


def _hitl_option_label(options: list, j: int) -> str:
    o = options[j]
    if isinstance(o, dict):
        lab = o.get("label")
        if lab is not None:
            return f"{j + 1}. {lab}"
    return f"{j + 1}. {o!r}"


def _render_sma_source_column_panel(
    *,
    item: dict,
    enriched_contract_path: str | None,
) -> None:
    if not enriched_contract_path:
        st.caption(":gray[Column details unavailable]")
        return
    fm = item.get("current_field_mapping")
    if not isinstance(fm, dict):
        st.caption(":gray[Column details unavailable]")
        return
    source_column = fm.get("source_column")
    source_table = fm.get("source_table")
    if not source_table or not isinstance(source_table, str) or not str(source_table).strip():
        st.caption(":gray[Column details unavailable]")
        return
    cache_key = f"esc-json-{enriched_contract_path}"
    if cache_key not in st.session_state:
        try:
            raw_c = read_unity_file_text(enriched_contract_path)
            st.session_state[cache_key] = {"ok": True, "obj": load_json_object_from_text(raw_c)}
        except Exception as ex:  # noqa: BLE001
            st.session_state[cache_key] = {"ok": False, "err": str(ex)}
    cached = st.session_state[cache_key]
    if not cached.get("ok"):
        st.caption(":gray[Column details unavailable]")
        return
    panel = extract_column_panel_fields(
        cached["obj"],
        dataset_name=str(source_table).strip(),
        source_column=str(source_column).strip() if source_column else None,
    )
    if panel is None:
        st.caption(":gray[Column details unavailable]")
        return
    exp = st.expander("Source column details", expanded=True)
    with exp:
        null_pct = panel.get("null_percentage")
        n_null = panel.get("null_count")
        try:
            pct_s = f"{float(null_pct):.2f}" if null_pct is not None else "—"
        except (TypeError, ValueError):
            pct_s = str(null_pct) if null_pct is not None else "—"
        nrows = n_null if n_null is not None else "—"
        uniq = panel.get("unique_count")
        uniq_s = str(uniq) if uniq is not None else "—"
        inst_tok = panel.get("inst_null_tokens") or []
        ds_tok = panel.get("dataset_null_tokens") or []
        tok_note = ""
        merged = [t for t in (inst_tok + ds_tok) if t]
        if merged:
            shown = ", ".join(repr(t) for t in merged[:12])
            tok_note = f" Institution null tokens: {shown}."

        rows = [
            {"Field": "Original name", "Value": str(panel.get("original_name", "—"))},
            {"Field": "Type", "Value": str(panel.get("dtype", "—"))},
            {
                "Field": "Null rate",
                "Value": f"{pct_s}% ({nrows} rows){tok_note}",
            },
            {"Field": "Unique values", "Value": uniq_s},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        label = "Unique values" if panel.get("chip_mode") == "unique" else "Sample values"
        st.caption(label)
        chips = panel.get("chip_values") or []
        mode = panel.get("chip_mode", "sample")
        if chips:
            esc = "".join(
                f'<span class="hitl-chip">{html.escape(str(v))[:200]}</span>'
                for v in chips[:80]
            )
            st.markdown(
                f'<div class="hitl-chip-row" title="{html.escape(str(mode))}">{esc}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("—")


def render_silver_hitl_editor(
    *,
    catalog: str,
    default_artifact_path: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
    pending_df: pd.DataFrame | None = None,
    uc_group_pending: bool = False,
) -> None:
    is_sma = _is_sma_hitl_context(phase, artifact_type)
    is_ia_grain = is_ia_grain_phase(phase, artifact_type)
    if is_sma:
        _inject_sma_hitl_css_once()

    compact_chrome = is_ia_grain or is_sma
    if not compact_chrome:
        st.caption(
            f"**artifact_type** in UC: ``{artifact_type}`` — **onboard_run_id** (this review block): "
            f"``{onboard_run_id}`` (in standard onboard layout it appears in the path under "
            f"``…/genai_mapping/runs/onboard/{{onboard_run_id}}/``)."
        )
        st.markdown(
            "**HITL JSON (silver volume)** — paths match ``edvise_genai_ia`` / ``edvise_genai_sma`` "
            "``resolve_run_paths``: ``{silver}/genai_mapping/runs/…/identity_agent/`` (IA) or "
            "``…/schema_mapping_agent/`` (SMA) with the HITL filenames. "
            "The ``onboard_run_id`` is the run folder in ``runs/onboard/…/``. "
            "Default file path = ``hitl_reviews.artifact_path`` (full string already includes that segment)."
        )
    sk = f"{_safe_key(onboard_run_id)}-{_safe_key(phase)}-{_safe_key(artifact_type)}"
    pkey = f"path-{sk}"
    path_label = (
        "Silver JSON path (override only if needed)"
        if compact_chrome
        else "UC file path to read/write (absolute ``/Volumes/{catalog}/…_silver/…``)"
    )
    path_in = st.text_input(
        path_label,
        value=default_artifact_path,
        key=pkey,
    )
    silver_path = (path_in or "").strip()
    if compact_chrome and silver_path:
        rel = silver_relative_path(silver_path)
        if rel:
            st.caption(f"Volume-relative: ``{rel}``")

    if not silver_path:
        st.caption("Set the file path to load HITL JSON (typically the registered `artifact_path`).")
        return
    if not silver_path.startswith("/Volumes/"):
        st.warning("Path should be an absolute Unity Catalog volume path starting with ``/Volumes/``.")
    if not artifact_path_contains_onboard_run_id(silver_path, str(onboard_run_id)):
        st.warning(
            "This path string does not contain the **onboard_run_id** for this review row. "
            "Onboard HITL files usually include it under "
            f"``…/genai_mapping/runs/onboard/{onboard_run_id}/…``. "
            "Confirm the path, or keep going only if you intend another file (e.g. an execute run path). "
        )
    if not is_sma and not is_ia_grain:
        st.code(silver_path, language="text")
    try:
        raw = read_unity_file_text(silver_path)
    except Exception as e:  # noqa: BLE001 — show in UI
        st.error(f"Could not read file: {e}")
        return
    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return
    items = data.get("items")
    if not isinstance(items, list) or not items:
        st.info("No `items` in this HITL JSON, or the list is empty — nothing to select.")
        return

    if is_ia_grain:
        render_ia_grain_hitl_cards(
            data=data,
            items=items,
            silver_path=silver_path,
            sk=sk,
            onboard_run_id=str(onboard_run_id),
            pending_df=pending_df,
            uc_group_pending=uc_group_pending,
            approve_uc_if_complete=lambda: approve_or_reject(
                catalog,
                str(onboard_run_id),
                str(phase),
                str(artifact_type),
                st.session_state["reviewer"],
                "approved",
            ),
        )
        return

    if is_sma:
        school_raw = (data.get("school_id") or data.get("institution_id") or "").strip()
        inst_title = format_institution_display_name(school_raw)
        st.markdown(
            f'<p class="hitl-sma-inst">{html.escape(inst_title)}</p>',
            unsafe_allow_html=True,
        )

    def _default_choice_index(item: dict, n_opts: int) -> int:
        c_raw = item.get("choice")
        if c_raw is None:
            return 0
        try:
            c_int = int(c_raw)
        except (TypeError, ValueError):
            c_int = 1
        return max(0, min(n_opts - 1, c_int - 1))

    option_item_indices: list[int] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        opts = item.get("options")
        if not opts or not isinstance(opts, list) or len(opts) < 1:
            continue
        option_item_indices.append(i)
        rk = f"sv{sk}item{i}{item.get('item_id', i)}"
        if rk not in st.session_state:
            st.session_state[rk] = _default_choice_index(item, len(opts))

    pending_ixs = [
        i
        for i in option_item_indices
        if isinstance(items[i], dict) and items[i].get("choice") is None
    ]
    nav_ixs = pending_ixs if pending_ixs else option_item_indices

    if is_sma and nav_ixs:
        cur_key = f"sma-nav-{sk}"
        if cur_key not in st.session_state:
            st.session_state[cur_key] = 0
        cur_nav = max(0, min(int(st.session_state[cur_key]), len(nav_ixs) - 1))
        st.session_state[cur_key] = cur_nav
        i = nav_ixs[cur_nav]

        y_pending = len(pending_ixs)
        run_line = ""
        if pending_df is not None and not pending_df.empty:
            ordered = _sma_run_pending_ordered_pairs(pending_df, str(onboard_run_id))
            y_run = len(ordered)
            pr = (silver_path.strip(), int(i))
            if y_run > 0 and pr in ordered:
                x_run = ordered.index(pr) + 1
                run_line = (
                    f'Onboard run <code>{html.escape(str(onboard_run_id))}</code> — '
                    f"<strong>{x_run} of {y_run} pending</strong>"
                )
            elif y_run > 0:
                run_line = (
                    f'Onboard run <code>{html.escape(str(onboard_run_id))}</code> — '
                    f"{y_run} pending on this run; this item is already resolved in JSON "
                    "(browse or change a choice and Save)."
                )
            else:
                run_line = (
                    f'Onboard run <code>{html.escape(str(onboard_run_id))}</code> — '
                    "<strong>0 of 0 pending</strong>"
                )
        elif y_pending > 0:
            x_pf = cur_nav + 1 if pending_ixs and nav_ixs == pending_ixs else 0
            run_line = f"<strong>{x_pf} of {y_pending} pending</strong> (this manifest)"
        else:
            run_line = (
                "No unresolved items in this file — browsing "
                f"{len(nav_ixs)} item(s) with options; change a radio and Save to update."
            )
        st.markdown(f'<p class="hitl-sma-meta">{run_line}</p>', unsafe_allow_html=True)
        nc1, nc2, nc3 = st.columns([1, 1, 6])
        with nc1:
            if st.button("◀ Prev", key=f"prev-{sk}"):
                st.session_state[cur_key] = max(0, int(st.session_state[cur_key]) - 1)
                st.rerun()
        with nc2:
            if st.button("Next ▶", key=f"nxt-{sk}"):
                st.session_state[cur_key] = min(
                    len(nav_ixs) - 1, int(st.session_state[cur_key]) + 1
                )
                st.rerun()
        with nc3:
            st.caption(
                f"Item {cur_nav + 1} of {len(nav_ixs)} "
                f"with options in this JSON ({len(option_item_indices)} total)."
            )

        item = items[i]
        q = (item.get("hitl_question") or "").strip() or f"Item {i + 1}"
        st.markdown(f'<div class="hitl-sma-q">{html.escape(q)}</div>', unsafe_allow_html=True)
        options = item.get("options")
        if not options or not isinstance(options, list):
            options = []
        _render_sma_review_context(item=item)
        _render_sma_option_descriptions(options=options)
        enriched = enriched_schema_contract_path_from_manifest(silver_path, str(onboard_run_id))
        _render_sma_source_column_panel(
            item=item,
            enriched_contract_path=enriched,
        )
        n = len(options)  # type: ignore[arg-type]
        rk = f"sv{sk}item{i}{item.get('item_id', i)}"
        st.radio(
            "Decision",
            list(range(n)),
            format_func=lambda j, opts=options: _hitl_option_label(opts, j),  # type: ignore[arg-type]
            key=rk,
        )
    elif is_sma and not nav_ixs:
        st.info("No HITL items with `options` in this JSON.")
    else:
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            q = (item.get("hitl_question") or "").strip() or f"Item {i + 1}"
            options = item.get("options")
            if not options or not isinstance(options, list):
                st.caption(f"**{q}** — no `options` (direct manifest edit may be required).")
                continue
            n = len(options)
            if n < 1:
                continue
            default_ix = _default_choice_index(item, n)
            rk = f"sv{sk}item{i}{item.get('item_id', i)}"
            st.radio(
                q,
                list(range(n)),
                index=default_ix,
                format_func=lambda j, opts=options: _hitl_option_label(opts, j),
                key=rk,
            )
    def _approve_uc() -> None:
        approve_or_reject(
            catalog,
            str(onboard_run_id),
            str(phase),
            str(artifact_type),
            st.session_state["reviewer"],
            "approved",
        )

    if st.button(
        "Save JSON & approve UC",
        key=f"ssave{sk}",
        type="primary",
        help="Writes every ``choice`` from the radios into this manifest, then approves the UC row when pending.",
    ):
        ok, err = persist_hitl_choice_radios_from_session(
            silver_path=silver_path,
            sk=sk,
            option_item_indices=option_item_indices,
            default_choice_index=_default_choice_index,
        )
        if not ok:
            st.error(err)
        else:
            if is_sma:
                _invalidate_sma_run_pending_cache(str(onboard_run_id))
                esc_path = enriched_schema_contract_path_from_manifest(
                    silver_path, str(onboard_run_id)
                )
                if esc_path:
                    ck = f"esc-json-{esc_path}"
                    if ck in st.session_state:
                        del st.session_state[ck]
            ap_ok, ap_err = try_approve_uc_after_json_write(
                uc_group_pending=uc_group_pending,
                approve_uc_if_complete=_approve_uc,
            )
            if not ap_ok:
                st.warning(f"JSON saved, but UC approve failed: {ap_err}")
            elif uc_group_pending:
                st.success("Saved manifest JSON and approved the UC row.")
                st.toast("JSON + UC complete.", icon="✅")
            else:
                st.success("Saved manifest JSON. UC was not pending, so UC approve was skipped.")
            st.rerun()


def _default_catalog() -> str:
    for key in ("GENAI_HITL_CATALOG", "DB_workspace"):
        v = (os.getenv(key) or "").strip()
        if v:
            return v
    return "dev_sst_02"


_CATALOG_SAFE = re.compile(r"^[a-zA-Z0-9_]+$")


def _validate_catalog(catalog: str) -> str:
    c = str(catalog).strip()
    if not c or not _CATALOG_SAFE.match(c):
        raise ValueError("Catalog must be a simple identifier (letters, digits, underscore).")
    return c


st.set_page_config(page_title="GenAI HITL reviews", layout="wide")
st.title("GenAI mapping — UC HITL reviews")
st.caption(
    "``hitl_reviews`` tracks UC status by ``(onboard_run_id, phase, artifact_type)``. "
    "**IA grain** and **SMA manifests** use **Save JSON & approve UC** (one file write + UC approve "
    "when pending). Use **Reject UC** below when you need to block without approving."
)

if "reviewer" not in st.session_state:
    st.session_state["reviewer"] = os.getenv("GENAI_HITL_REVIEWER", "").strip() or (
        os.getenv("USER", "") or os.getenv("USERNAME", "") or "reviewer"
    ).strip()

with st.sidebar:
    st.subheader("Connection")
    try:
        get_warehouse_id()
        st.success("DATABRICKS_WAREHOUSE_ID is set")
    except RuntimeError as e:
        st.error(str(e))
    catalog_in = st.text_input("Unity Catalog", value=_default_catalog())
    try:
        catalog = _validate_catalog(catalog_in)
    except ValueError as e:
        st.warning(str(e))
        catalog = _default_catalog()
    st.session_state["reviewer"] = st.text_input(
        "Reviewer name (stored on approve/reject)",
        value=st.session_state["reviewer"],
    )
    limit = st.number_input("Row limit", min_value=50, max_value=5000, value=500, step=50)
    st.divider()
    st.subheader("Filters")
    f_run = st.text_input("onboard_run_id contains", value="")
    f_phase = st.text_input("phase equals", value="")
    f_status = st.selectbox(
        "status",
        options=["(any)", "pending", "approved", "rejected"],
        index=0,
    )
    st.button("Refresh data", type="primary", help="Re-runs the query with current filters.")

try:
    get_warehouse_id()
except RuntimeError:
    st.stop()

# Client-side filter for "contains" on run id: load broader slice when using contains
use_contains = bool((f_run or "").strip())
onboard_exact = None if use_contains else ((f_run or "").strip() or None)
phase_f = (f_phase or "").strip() or None
status_f = None if f_status == "(any)" else f_status

try:
    df = load_hitl_rows(
        catalog,
        onboard_run_id=onboard_exact,
        phase=phase_f,
        status=status_f,
        limit=limit if not use_contains else min(limit, 5000),
    )
    if use_contains and (f_run or "").strip():
        needle = (f_run or "").strip().lower()
        df = df[df["onboard_run_id"].astype(str).str.lower().str.contains(needle, na=False)]
except Exception as e:  # noqa: BLE001 — show in UI
    st.exception(e)
    st.stop()

# Show ``artifact_path`` (silver) and optional ``institution_id`` from ``pipeline_runs``
_display_cols = [
    c
    for c in (
        "institution_id",
        "onboard_run_id",
        "phase",
        "artifact_type",
        "artifact_path",
        "status",
        "reviewer",
        "reviewed_at",
    )
    if c in df.columns
]
st.dataframe(
    df[_display_cols] if _display_cols else df,
    use_container_width=True,
    hide_index=True,
)

if df.empty:
    st.info("No rows match. Clear filters or raise the row limit.")
    st.stop()

st.subheader("Actions")
st.caption(
    "**IA grain** and **SMA** (cohort/course manifest): **Save JSON & approve UC** writes the "
    "silver JSON and approves the pending UC row. Other HITL JSON using the same radio layout "
    "uses the same button."
)

pending = df[df["status"].astype(str).str.lower() == "pending"].copy()
# Editor must stay available when UC is already approved but JSON still needs edits
# (e.g. only some IA grain items had "Save choice to JSON" clicked).
action_df = pending if not pending.empty else df
if pending.empty and not df.empty:
    n_all = len(df)
    st.info(
        f"**{n_all}** ``hitl_reviews`` row(s) match your filters, and **0** are ``status = pending``. "
        "That usually means these runs were already approved or rejected in Unity Catalog—**not** "
        "that the JSON is empty. Many rows here are normal (e.g. one row per manifest path). "
        "The silver JSON editor still appears under each group; **Save JSON & approve UC** writes "
        "JSON and only runs the UC approve SQL when that group is still pending. "
        "To list only gates awaiting UC, set sidebar **status** to **pending**."
    )
elif not pending.empty:
    st.success(f"{len(pending)} pending UC row(s) in the current result set.")

groups = (
    action_df[["onboard_run_id", "phase", "artifact_type"]]
    .drop_duplicates()
    .sort_values(["onboard_run_id", "phase", "artifact_type"])
    .itertuples(index=False)
)

for onboard_run_id, phase, artifact_type in groups:
    sub = action_df[
        (action_df["onboard_run_id"] == onboard_run_id)
        & (action_df["phase"] == phase)
        & (action_df["artifact_type"] == artifact_type)
    ]
    sub_pending = pending[
        (pending["onboard_run_id"] == onboard_run_id)
        & (pending["phase"] == phase)
        & (pending["artifact_type"] == artifact_type)
    ]
    # Bordered container (not st.expander): nested expanders are forbidden in Streamlit, but
    # the HITL editor and SMA/IA helpers use expanders internally.
    with st.container(border=True):
        _n_paths = len(sub)
        _hdr = f"`{onboard_run_id}` · `{phase}` · `{artifact_type}`"
        if _n_paths != 1:
            _hdr += f" · {_n_paths} paths"
        st.markdown(_hdr)
        raw_s = sub["artifact_path"].dropna().astype(str).str.strip()
        raw_paths = [p for p in raw_s.tolist() if p]
        if not raw_paths:
            st.warning(
                "No ``artifact_path`` on these rows; the pipeline has nothing to point the editor at. "
                "Re-run registration from ``edvise_genai_ia`` / ``edvise_genai_sma`` onboard."
            )
        else:
            default_path = raw_paths[0]
            if len(set(raw_paths)) > 1:
                with st.expander("Multiple artifact paths for this group", expanded=False):
                    st.caption(
                        "Defaulting the editor to the first path. Pick another in the path field if needed."
                    )
                    st.code("\n".join(sorted(set(raw_paths))), language="text")
            render_silver_hitl_editor(
                catalog=catalog,
                default_artifact_path=default_path,
                onboard_run_id=str(onboard_run_id),
                phase=str(phase),
                artifact_type=str(artifact_type),
                pending_df=pending if not pending.empty else action_df,
                uc_group_pending=not sub_pending.empty,
            )
        st.divider()
        is_ia_grain_row = is_ia_grain_phase(str(phase), str(artifact_type))
        is_sma_row = _is_sma_hitl_context(str(phase), str(artifact_type))
        if sub_pending.empty:
            st.caption(
                "This UC group is not **pending** (already approved/rejected or filtered out). "
                "Use the JSON editor above; **Save JSON & approve UC** still saves JSON and does not "
                "change ``hitl_reviews`` while the row is not pending."
            )
        elif is_ia_grain_row or is_sma_row:
            st.caption(
                "IA / SMA: **Save JSON & approve UC** in the editor approves this pending row. "
                "Use **Reject UC** only if you intend to block the gate."
            )
            if st.button("Reject UC", key=f"r-{onboard_run_id}-{phase}-{artifact_type}"):
                try:
                    approve_or_reject(
                        catalog,
                        str(onboard_run_id),
                        str(phase),
                        str(artifact_type),
                        st.session_state["reviewer"],
                        "rejected",
                    )
                    st.toast("UC row rejected.", icon="⛔")
                    st.rerun()
                except Exception as ex:  # noqa: BLE001
                    st.error(str(ex))
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button(
                    "Approve UC",
                    key=f"a-{onboard_run_id}-{phase}-{artifact_type}",
                    type="primary",
                ):
                    try:
                        approve_or_reject(
                            catalog,
                            str(onboard_run_id),
                            str(phase),
                            str(artifact_type),
                            st.session_state["reviewer"],
                            "approved",
                        )
                        st.toast("UC row approved.", icon="✅")
                        st.rerun()
                    except Exception as ex:  # noqa: BLE001
                        st.error(str(ex))
            with c2:
                if st.button("Reject UC", key=f"r-{onboard_run_id}-{phase}-{artifact_type}-sma"):
                    try:
                        approve_or_reject(
                            catalog,
                            str(onboard_run_id),
                            str(phase),
                            str(artifact_type),
                            st.session_state["reviewer"],
                            "rejected",
                        )
                        st.toast("UC row rejected.", icon="⛔")
                        st.rerun()
                    except Exception as ex:  # noqa: BLE001
                        st.error(str(ex))
            with c3:
                st.caption("Updates ``hitl_reviews`` only (not the JSON file).")
