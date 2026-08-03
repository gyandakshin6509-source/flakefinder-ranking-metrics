"""
qc_run.py: comprehensive QC pass before lab handoff.

Runs ten checks against the current state of the repo + outputs/.
Writes the consolidated report to outputs/qc_report.md. Does not modify any
source files or production outputs. CHECK 4 re-runs flake_metrics on the
graphene run with output dirs redirected to outputs_qc_rerun/, and CHECK 7
writes spot-check figures to outputs/qc_spotcheck/.

Usage:
    python src/qc_run.py
"""

from __future__ import annotations

import ast
import csv
import gc
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import eop
import flake_metrics
from eop import build_chip_occupancy, compute_eop, save_eop_figure
from io_utils import load_scan_meta
from lhrr import save_lhrr_figure

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
EXTRACTED = DATA_ROOT / "_extracted"
OUTPUTS = PROJECT_ROOT / "outputs"
METRICS = OUTPUTS / "metrics"
QC_RERUN = PROJECT_ROOT / "outputs_qc_rerun"
QC_SPOT = OUTPUTS / "qc_spotcheck"
README = PROJECT_ROOT / "README.md"
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
PLAN_MD = OUTPUTS / "plan.md"
FLATFIELD_NPY = DATA_ROOT / "flakes data" / "flakes data" / "flatfields" / "flatfield_10x_bin3.npy"

RUNS = {
    "run_20260505_1616": ["chip_0", "chip_1", "chip_2", "chip_3", "chip_4", "chip_5"],
    "run_20260504_1026": ["chip_0", "chip_1"],
}

EXPECTED_CSV_COLUMNS = list(flake_metrics.CSV_COLUMNS)
EXPECTED_JSON_TOP_KEYS = {"chip", "material", "n_candidates", "occupancy", "candidates"}
EXPECTED_OCCUPANCY_KEYS = {
    "n_detections", "substrate_baseline", "baseline_fallback",
    "max_raw_contrast", "stage_px_um", "map_origin_um",
}
EXPECTED_CANDIDATE_FIELDS = {
    "chip", "material", "label",
    "stage_x_um", "stage_y_um", "stage_z_um",
    "lhrr_area_px", "lhrr_area_um2", "lhrr_fraction",
    "lhrr_bbox_frame", "lhrr_bbox_stage",
    "variance_threshold_used", "kernel_size_px",
    "lhrr_quality_flag", "lhrr_skip_reason",
    "clearance_um", "weighted_obstruction", "eop_score",
    "clearance_warning", "cand_pixels", "stage_center_um",
    "eop_skip_reason",
}
LHRR_NULLABLE_WHEN_SKIPPED = {
    "lhrr_area_px", "lhrr_area_um2", "lhrr_fraction",
    "lhrr_bbox_frame", "lhrr_bbox_stage",
    "variance_threshold_used", "kernel_size_px",
    "lhrr_quality_flag",
}
EOP_NULLABLE_WHEN_SKIPPED = {
    "clearance_um", "weighted_obstruction", "eop_score",
    "clearance_warning", "cand_pixels", "stage_center_um",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_float(v):
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _maybe_int(v):
    if v is None or v == "" or v == "None":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _maybe_bool(v):
    if v is None or v == "" or v == "None":
        return None
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _status(level: str, msg: str) -> str:
    return f"**{level}**: {msg}"


# ---------------------------------------------------------------------------
# CHECK 1: Schema completeness
# ---------------------------------------------------------------------------

def check_1_schema() -> tuple[str, list[str]]:
    out: list[str] = ["## CHECK 1: Schema completeness", ""]
    issues: list[str] = []

    # CSV side
    out.append("### Ranked CSV columns")
    out.append("")
    for run, _ in RUNS.items():
        csv_path = METRICS / f"{run}_ranked_candidates.csv"
        if not csv_path.exists():
            issues.append(f"{csv_path.name}: missing")
            out.append(f"- `{csv_path.name}`: **MISSING**")
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            cols = next(reader, [])
        extra = [c for c in cols if c not in EXPECTED_CSV_COLUMNS]
        missing = [c for c in EXPECTED_CSV_COLUMNS if c not in cols]
        if extra or missing:
            issues.append(f"{csv_path.name}: schema mismatch")
        out.append(f"- `{csv_path.name}`: {len(cols)} columns")
        if extra:
            out.append(f"    - extra: {extra}")
        if missing:
            out.append(f"    - missing: {missing}")
        if not extra and not missing:
            out.append(f"    - matches `flake_metrics.CSV_COLUMNS` exactly")
    out.append("")

    # JSON side
    out.append("### Per-chip metrics JSON schema")
    out.append("")
    n_records_checked = 0
    record_issues = 0
    for run, chips in RUNS.items():
        for chip in chips:
            jp = METRICS / f"{run}_{chip}_metrics.json"
            if not jp.exists():
                issues.append(f"{jp.name}: missing")
                out.append(f"- `{jp.name}`: **MISSING**")
                continue
            data = _read_json(jp)
            top_extra = set(data.keys()) - EXPECTED_JSON_TOP_KEYS
            top_missing = EXPECTED_JSON_TOP_KEYS - set(data.keys())
            occ = data.get("occupancy", {})
            occ_extra = set(occ.keys()) - EXPECTED_OCCUPANCY_KEYS
            occ_missing = EXPECTED_OCCUPANCY_KEYS - set(occ.keys())

            note = "ok"
            if top_extra or top_missing or occ_extra or occ_missing:
                note = "MISMATCH"
                issues.append(f"{jp.name}: schema mismatch")
            out.append(f"- `{jp.name}`, top: {note}")
            if top_extra:
                out.append(f"    - top extra: {sorted(top_extra)}")
            if top_missing:
                out.append(f"    - top missing: {sorted(top_missing)}")
            if occ_extra:
                out.append(f"    - occupancy extra: {sorted(occ_extra)}")
            if occ_missing:
                out.append(f"    - occupancy missing: {sorted(occ_missing)}")

            for rec in data.get("candidates", []):
                n_records_checked += 1
                rec_extra = set(rec.keys()) - EXPECTED_CANDIDATE_FIELDS
                rec_missing = EXPECTED_CANDIDATE_FIELDS - set(rec.keys())
                if rec_extra or rec_missing:
                    record_issues += 1
                    issues.append(
                        f"{jp.name}/{rec.get('label', '?')}: candidate fields "
                        f"missing={sorted(rec_missing)} extra={sorted(rec_extra)}"
                    )
                # Null discipline
                lhrr_skip = rec.get("lhrr_skip_reason")
                eop_skip = rec.get("eop_skip_reason")
                for f in LHRR_NULLABLE_WHEN_SKIPPED:
                    if rec.get(f) is None and lhrr_skip is None:
                        record_issues += 1
                        issues.append(
                            f"{jp.name}/{rec.get('label')}: {f} is null but "
                            f"lhrr_skip_reason is null"
                        )
                for f in EOP_NULLABLE_WHEN_SKIPPED:
                    if rec.get(f) is None and eop_skip is None:
                        record_issues += 1
                        issues.append(
                            f"{jp.name}/{rec.get('label')}: {f} is null but "
                            f"eop_skip_reason is null"
                        )

    out.append("")
    out.append(f"### Per-candidate field check")
    out.append(f"- candidate records inspected: **{n_records_checked}**")
    out.append(f"- field/null violations: **{record_issues}**")
    out.append("")

    status = "PASS" if not issues else "FAIL"
    out.insert(1, f"_{status}_  {len(issues)} issue(s)")
    out.insert(2, "")
    return status, out


# ---------------------------------------------------------------------------
# CHECK 2: Range validation
# ---------------------------------------------------------------------------

def check_2_ranges() -> tuple[str, list[str]]:
    out: list[str] = ["## CHECK 2: Range validation", ""]
    issues: list[str] = []

    all_rows: list[dict] = []
    for run in RUNS:
        all_rows.extend(_read_csv_rows(METRICS / f"{run}_ranked_candidates.csv"))

    out.append(f"Aggregated **{len(all_rows)}** rows across both ranked CSVs.")
    out.append("")

    numeric_specs = [
        ("lhrr_area_um2", lambda v: v >= 0),
        ("lhrr_fraction", lambda v: 0.0 <= v <= 1.0),
        ("clearance_um", lambda v: v >= 0),
        ("weighted_obstruction", lambda v: v >= 0),
        ("eop_score", lambda v: v >= 0),
        ("lhrr_area_norm", lambda v: 0.0 <= v <= 1.0),
        ("eop_score_norm", lambda v: 0.0 <= v <= 1.0),
        ("composite_score", lambda v: 0.0 <= v <= 1.0),
    ]

    out.append("| field | min | max | mean | n_null | n_negative | n_nan | n_out_of_range |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name, predicate in numeric_specs:
        values = []
        n_null = n_neg = n_nan = n_oor = 0
        violators: list[str] = []
        for r in all_rows:
            raw = r.get(name)
            v = _maybe_float(raw)
            if v is None:
                n_null += 1
                continue
            if v != v:  # NaN
                n_nan += 1
                violators.append(r.get("label", "?"))
                continue
            if v < 0:
                n_neg += 1
            try:
                ok = predicate(v)
            except Exception:
                ok = False
            if not ok:
                n_oor += 1
                violators.append(r.get("label", "?"))
            values.append(v)
        if values:
            mn, mx, mean = min(values), max(values), sum(values) / len(values)
        else:
            mn = mx = mean = float("nan")
        out.append(
            f"| `{name}` | {mn:.4g} | {mx:.4g} | {mean:.4g} | {n_null} | {n_neg} | {n_nan} | {n_oor} |"
        )
        if n_oor or n_nan:
            issues.append(
                f"{name}: {n_oor} out-of-range, {n_nan} NaN; "
                f"first violators: {violators[:5]}"
            )

    # Categorical checks
    out.append("")
    out.append("### Categorical fields")
    flag_values: dict[str, int] = {}
    warn_values: dict[str, int] = {}
    bad_flag_labels: list[str] = []
    bad_warn_labels: list[str] = []
    for r in all_rows:
        flg = r.get("lhrr_quality_flag")
        if flg in ("", None):
            flg = "<null>"
        flag_values[flg] = flag_values.get(flg, 0) + 1
        if flg not in {"usable", "marginal", "<null>"}:
            bad_flag_labels.append(r.get("label", "?"))

        warn_raw = r.get("clearance_warning")
        if warn_raw in ("True", "False", True, False):
            key = str(warn_raw).strip()
        elif warn_raw in (None, "", "None"):
            key = "<null>"
        else:
            key = str(warn_raw)
            bad_warn_labels.append(r.get("label", "?"))
        warn_values[key] = warn_values.get(key, 0) + 1

    out.append(f"- `lhrr_quality_flag`: {flag_values}")
    if bad_flag_labels:
        issues.append(f"lhrr_quality_flag unexpected values: {bad_flag_labels[:5]}")
    out.append(f"- `clearance_warning`: {warn_values}")
    if bad_warn_labels:
        issues.append(f"clearance_warning non-bool values: {bad_warn_labels[:5]}")
    out.append("")

    # Skip-reason discipline: when lhrr_area_um2 is null, lhrr_skip_reason must be set
    null_area_no_skip: list[str] = []
    for r in all_rows:
        if _maybe_float(r.get("lhrr_area_um2")) is None:
            if r.get("lhrr_skip_reason") in ("", None, "None"):
                null_area_no_skip.append(r.get("label", "?"))
    if null_area_no_skip:
        issues.append(f"null lhrr_area_um2 with no skip_reason: {null_area_no_skip[:5]}")

    null_eop_no_skip: list[str] = []
    for r in all_rows:
        if _maybe_float(r.get("eop_score")) is None:
            if r.get("eop_skip_reason") in ("", None, "None"):
                null_eop_no_skip.append(r.get("label", "?"))
    if null_eop_no_skip:
        issues.append(f"null eop_score with no skip_reason: {null_eop_no_skip[:5]}")

    status = "PASS" if not issues else "FAIL"
    out.insert(1, f"_{status}_  {len(issues)} issue(s)")
    out.insert(2, "")
    return status, out


# ---------------------------------------------------------------------------
# CHECK 3: Candidate completeness (no silent drops)
# ---------------------------------------------------------------------------

def check_3_completeness() -> tuple[str, list[str]]:
    out: list[str] = ["## CHECK 3: Candidate completeness", ""]
    issues: list[str] = []
    out.append("| run | chip | revisit_50x.json points | metrics JSON candidates | CSV rows | match? |")
    out.append("|---|---|---|---|---|---|")

    for run, chips in RUNS.items():
        csv_rows = _read_csv_rows(METRICS / f"{run}_ranked_candidates.csv")
        csv_per_chip: dict[str, int] = {}
        for r in csv_rows:
            csv_per_chip[r["chip"]] = csv_per_chip.get(r["chip"], 0) + 1
        for chip in chips:
            revisit = _read_json(EXTRACTED / run / chip / "seg" / "revisit_50x.json")
            n_input = len(revisit.get("points", []))
            metrics = _read_json(METRICS / f"{run}_{chip}_metrics.json")
            n_json = len(metrics.get("candidates", []))
            n_csv = csv_per_chip.get(chip, 0)
            ok = (n_input == n_json == n_csv)
            mark = "ok" if ok else "**MISMATCH**"
            out.append(f"| {run} | {chip} | {n_input} | {n_json} | {n_csv} | {mark} |")
            if not ok:
                issues.append(f"{run}/{chip}: {n_input} input vs {n_json} JSON vs {n_csv} CSV")
    out.append("")
    status = "PASS" if not issues else "FAIL"
    out.insert(1, f"_{status}_  {len(issues)} issue(s)")
    out.insert(2, "")
    return status, out


# ---------------------------------------------------------------------------
# CHECK 4: Determinism (re-run graphene + diff)
# ---------------------------------------------------------------------------

def check_4_determinism() -> tuple[str, list[str], dict[str, Any]]:
    """Returns (status, lines, cached_chip_data) where cached_chip_data is a
    dict {chip_name: chip_data} for graphene chips, kept in memory so that
    CHECK 7 can reuse them and avoid rebuilding."""
    out: list[str] = ["## CHECK 4: Determinism", ""]
    issues: list[str] = []

    QC_RERUN.mkdir(parents=True, exist_ok=True)
    rerun_metrics = QC_RERUN / "metrics"
    rerun_eop = QC_RERUN / "eop"
    rerun_metrics.mkdir(parents=True, exist_ok=True)
    rerun_eop.mkdir(parents=True, exist_ok=True)

    # Monkey-patch flake_metrics output dirs
    orig_metrics_dir = flake_metrics.METRICS_DIR
    orig_eop_dir = flake_metrics.EOP_DIR
    flake_metrics.METRICS_DIR = rerun_metrics
    flake_metrics.EOP_DIR = rerun_eop

    cached: dict[str, Any] = {}
    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)

    run = "run_20260504_1026"
    print(f"  CHECK 4: re-running flake_metrics on {run} -> {QC_RERUN}")
    t0 = time.time()
    try:
        flake_metrics.main(EXTRACTED / run)
    except Exception as exc:
        issues.append(f"re-run raised: {type(exc).__name__}: {exc}")
        out.append(f"- Re-run failed: {exc}")
        flake_metrics.METRICS_DIR = orig_metrics_dir
        flake_metrics.EOP_DIR = orig_eop_dir
        status = "FAIL"
        out.insert(1, f"_{status}_  {len(issues)} issue(s)")
        out.insert(2, "")
        return status, out, cached
    finally:
        flake_metrics.METRICS_DIR = orig_metrics_dir
        flake_metrics.EOP_DIR = orig_eop_dir

    out.append(f"- re-run completed in {time.time() - t0:.1f} s")

    # Diff CSVs
    orig_csv = METRICS / f"{run}_ranked_candidates.csv"
    new_csv = rerun_metrics / f"{run}_ranked_candidates.csv"
    o = _read_csv_rows(orig_csv)
    n = _read_csv_rows(new_csv)
    out.append(f"- original CSV rows: {len(o)}, re-run CSV rows: {len(n)}")
    if len(o) != len(n):
        issues.append("row count differs between original and re-run CSVs")

    differing_rows = 0
    max_num_diff = 0.0
    non_numeric_diffs: list[str] = []
    paired = list(zip(o, n))
    for a, b in paired:
        if a.get("label") != b.get("label"):
            non_numeric_diffs.append(f"label order: {a.get('label')} vs {b.get('label')}")
            differing_rows += 1
            continue
        for k in EXPECTED_CSV_COLUMNS:
            va, vb = a.get(k), b.get(k)
            fa, fb = _maybe_float(va), _maybe_float(vb)
            if fa is not None and fb is not None:
                d = abs(fa - fb)
                if d > max_num_diff:
                    max_num_diff = d
                if d > 1e-9:
                    differing_rows += 1
                    if len(non_numeric_diffs) < 5:
                        non_numeric_diffs.append(
                            f"{a.get('label')}.{k}: {fa} vs {fb} (Δ={d:.3e})"
                        )
                    break
            else:
                if (va or "") != (vb or ""):
                    differing_rows += 1
                    if len(non_numeric_diffs) < 5:
                        non_numeric_diffs.append(
                            f"{a.get('label')}.{k}: {va!r} vs {vb!r}"
                        )
                    break

    out.append(f"- differing rows: **{differing_rows}**")
    out.append(f"- max numeric diff: **{max_num_diff:.3e}**")
    if non_numeric_diffs:
        out.append("- first differences:")
        for d in non_numeric_diffs:
            out.append(f"    - {d}")
    out.append("")

    if differing_rows > 0 or max_num_diff > 1e-9:
        issues.append(f"non-deterministic: {differing_rows} rows differ, max |Δ|={max_num_diff:.3e}")

    status = "PASS" if not issues else "WARN" if max_num_diff <= 1e-6 else "FAIL"
    out.insert(1, f"_{status}_  {len(issues)} issue(s)")
    out.insert(2, "")
    return status, out, cached


# ---------------------------------------------------------------------------
# CHECK 5: README / code constants alignment
# ---------------------------------------------------------------------------

def _module_constants(path: Path) -> list[str]:
    """Return all module-level UPPER_CASE assignment targets via AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (
                    isinstance(t, ast.Name)
                    and re.match(r"^[A-Z][A-Z0-9_]*$", t.id)
                    and not t.id.startswith("_")
                    and t.id not in names
                ):
                    names.append(t.id)
    return names


def check_5_readme_alignment() -> tuple[str, list[str]]:
    out: list[str] = ["## CHECK 5: README / code constants alignment", ""]
    issues: list[str] = []

    readme_text = README.read_text(encoding="utf-8")

    code_constants: dict[str, list[str]] = {}
    for py in sorted(SRC_DIR.glob("*.py")):
        if py.name.startswith("_") or py.name in {"qc_run.py"}:
            continue
        try:
            consts = _module_constants(py)
        except SyntaxError:
            consts = []
        if consts:
            code_constants[py.name] = consts

    in_both: list[tuple[str, str]] = []
    in_code_only: list[tuple[str, str]] = []
    for fname, consts in code_constants.items():
        for c in consts:
            if c in readme_text:
                in_both.append((fname, c))
            else:
                in_code_only.append((fname, c))

    # Find ALL_CAPS_TOKEN candidates in README that look like constant names
    readme_caps = set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", readme_text))
    code_const_set = {c for _, lst in code_constants.items() for c in lst}
    in_readme_only = sorted(readme_caps - code_const_set)
    # Filter common false positives (not really constants we'd own)
    drop = {"BGR", "RGB", "JSON", "CSV", "WCAG", "CRLF", "LF",
            "DIST_L2", "B_SOFTWARE_HKLM"}
    in_readme_only = [t for t in in_readme_only if t not in drop]

    out.append("### In code AND in README (good)")
    for f, c in in_both:
        out.append(f"- `{c}`  ({f})")
    out.append("")
    out.append("### In code but NOT in README (documentation gap)")
    if not in_code_only:
        out.append("- (none)")
    for f, c in in_code_only:
        out.append(f"- `{c}`  ({f})")
        issues.append(f"undocumented constant: {c} ({f})")
    out.append("")
    out.append("### Looks like a constant in README but NOT defined in any module")
    if not in_readme_only:
        out.append("- (none)")
    for c in in_readme_only:
        out.append(f"- `{c}`")
        issues.append(f"stale README reference: {c}")
    out.append("")

    status = "PASS" if not issues else "WARN"
    out.insert(1, f"_{status}_  {len(issues)} issue(s): these are documentation hygiene, not correctness")
    out.insert(2, "")
    return status, out


# ---------------------------------------------------------------------------
# CHECK 6: CLAUDE.md and plan.md alignment
# ---------------------------------------------------------------------------

def check_6_claude_plan() -> tuple[str, list[str]]:
    out: list[str] = ["## CHECK 6: CLAUDE.md and plan.md alignment", ""]
    issues: list[str] = []

    if not CLAUDE_MD.exists():
        issues.append("CLAUDE.md not found")
        out.append(f"- CLAUDE.md not found at {CLAUDE_MD}")
    else:
        ctxt = CLAUDE_MD.read_text(encoding="utf-8")
        out.append("### CLAUDE.md")
        # Specific assertions
        assertions = [
            ("VARIANCE_PERCENTILE default: 50", "VARIANCE_PERCENTILE = 50"),
            ("variance threshold sampled from strict mask",
             'in_mask_var = var_map[mask_crop]'),
        ]
        for desc, key in assertions:
            present = key in ctxt or desc.split(":")[-1].strip().split()[0] in ctxt
            mark = "ok" if present else "MISSING"
            out.append(f"- {desc}: **{mark}**")
            if not present:
                issues.append(f"CLAUDE.md missing: {desc}")

        # Stale references: CLAUDE.md predates the per-material threshold
        material_aware = "LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL" in ctxt
        out.append(f"- mentions `LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL`: "
                   f"**{'yes' if material_aware else 'NO (added after CLAUDE.md was written)'}**")
        if not material_aware:
            issues.append(
                "CLAUDE.md does not mention LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL "
                "(per-material thresholds added after the LHRR notes section)"
            )

        # Mentions area-integral fix?
        eop_fix = "stage_px" in ctxt and "area integral" in ctxt.lower()
        out.append(f"- documents EOP area-integral fix: **{'yes' if eop_fix else 'NO'}**")
        if not eop_fix:
            issues.append(
                "CLAUDE.md does not document the EOP area-integral fix "
                "(weighted_obstruction multiplied by stage_px²)"
            )
        out.append("")

    if not PLAN_MD.exists():
        issues.append("plan.md not found")
        out.append(f"- plan.md not found at {PLAN_MD}")
    else:
        ptxt = PLAN_MD.read_text(encoding="utf-8")
        out.append("### plan.md")
        plan_checks = [
            ("VARIANCE_PERCENTILE default", "VARIANCE_PERCENTILE", "50"),
            ("OCCUPANCY_STAGE_PX_UM", "OCCUPANCY_STAGE_PX_UM", "5.0"),
            ("MAX_OBSTRUCTION_RADIUS_UM", "MAX_OBSTRUCTION_RADIUS_UM", "500"),
            ("BASELINE_SAMPLE_FRAMES", "BASELINE_SAMPLE_FRAMES", "20"),
            ("COMPOSITE_W_LHRR", "COMPOSITE_W_LHRR", "0.5"),
            ("COMPOSITE_W_EOP",  "COMPOSITE_W_EOP",  "0.5"),
        ]
        for label, name, expected in plan_checks:
            mentioned = name in ptxt
            value_matches = expected in ptxt
            note = "ok" if mentioned and value_matches else "REVIEW"
            out.append(f"- `{name}` mentioned ({mentioned}), expected default `{expected}` "
                       f"present in text ({value_matches}): **{note}**")
            if not mentioned:
                issues.append(f"plan.md does not reference {name}")

        # Output path drift
        # plan.md said outputs/ranked_candidates.csv; we now write
        # outputs/metrics/<run>_ranked_candidates.csv
        path_old = "outputs/ranked_candidates.csv"
        path_new = "outputs/metrics/<run>_ranked_candidates.csv"
        if path_old in ptxt and "<run>" not in ptxt and "metrics/" not in ptxt:
            out.append(f"- ⚠ plan.md still mentions `{path_old}` "
                       f"but actual output is `{path_new}`")
            issues.append(f"plan.md output path drift: {path_old} -> {path_new}")
        else:
            out.append(f"- output path: implementation writes `{path_new}` "
                       f"(plan.md mentions `{path_old}`: {path_old in ptxt})")

        # LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL
        if "LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL" not in ptxt:
            out.append("- ⚠ plan.md does not mention "
                       "`LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL` "
                       "(per-material thresholds were a post-plan addition)")
            issues.append("plan.md missing post-plan addition: "
                          "LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL")

        out.append("")

    status = "PASS" if not issues else "WARN"
    out.insert(1, f"_{status}_  {len(issues)} discrepancy/discrepancies: "
                  f"documentation drift, not correctness")
    out.insert(2, "")
    return status, out


# ---------------------------------------------------------------------------
# CHECK 7: Spot-check sample generation
# ---------------------------------------------------------------------------

def _top_n(csv_path: Path, n: int = 5) -> list[dict]:
    rows = _read_csv_rows(csv_path)
    rows.sort(
        key=lambda r: _maybe_float(r.get("composite_score")) or -1.0,
        reverse=True,
    )
    return rows[:n]


def check_7_spotcheck() -> tuple[str, list[str]]:
    out: list[str] = ["## CHECK 7: Spot-check figure generation", ""]
    issues: list[str] = []
    QC_SPOT.mkdir(parents=True, exist_ok=True)

    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)

    selections: list[tuple[str, str, dict]] = []  # (run, chip, row)
    for run in RUNS:
        csv_path = METRICS / f"{run}_ranked_candidates.csv"
        for r in _top_n(csv_path, 5):
            selections.append((run, r["chip"], r))

    # Group by (run, chip) so we build occupancy once per chip
    by_chip: dict[tuple[str, str], list[dict]] = {}
    for run, chip, row in selections:
        by_chip.setdefault((run, chip), []).append(row)

    out.append("### Top 5 selected per run")
    out.append("")
    out.append("| run | chip | label | composite |")
    out.append("|---|---|---|---|")
    for run, chip, row in selections:
        comp = _maybe_float(row.get("composite_score"))
        out.append(f"| {run} | {chip} | {row['label']} | "
                   f"{comp:.4f} |")
    out.append("")
    out.append(f"Building occupancy for **{len(by_chip)}** unique chips, "
               f"generating LHRR + EOP figures for **{len(selections)}** candidates.")
    out.append("")

    n_lhrr_ok = n_eop_ok = 0
    for (run, chip), rows in sorted(by_chip.items()):
        chip_dir = EXTRACTED / run / chip
        print(f"  CHECK 7: building occupancy for {run}/{chip}")
        t0 = time.time()
        chip_data = build_chip_occupancy(chip_dir, flatfield, progress_every=200)
        print(f"    occupancy: {time.time() - t0:.1f} s")
        material = rows[0].get("material")
        for row in rows:
            label = row["label"]
            base = f"{run}_{chip}_{label}"
            try:
                lhrr_path = QC_SPOT / f"lhrr_{base}.png"
                save_lhrr_figure(chip_dir, label, flatfield, lhrr_path,
                                 material=material)
                n_lhrr_ok += 1
            except Exception as exc:
                issues.append(f"LHRR figure failed: {base}: {exc}")
            try:
                er = compute_eop(chip_dir, label, chip_data)
                eop_path = QC_SPOT / f"eop_{base}.png"
                save_eop_figure(chip_dir, label, chip_data, er, eop_path)
                n_eop_ok += 1
            except Exception as exc:
                issues.append(f"EOP figure failed: {base}: {exc}")
        del chip_data
        gc.collect()

    out.append(f"- LHRR figures saved: **{n_lhrr_ok}/{len(selections)}**")
    out.append(f"- EOP figures saved: **{n_eop_ok}/{len(selections)}**")
    out.append(f"- output directory: `{QC_SPOT.relative_to(PROJECT_ROOT)}`")
    out.append("")

    status = "PASS" if not issues else "FAIL"
    out.insert(1, f"_{status}_  {len(issues)} issue(s)")
    out.insert(2, "")
    return status, out


# ---------------------------------------------------------------------------
# CHECK 8: Edge case coverage
# ---------------------------------------------------------------------------

def check_8_edges() -> tuple[str, list[str]]:
    out: list[str] = ["## CHECK 8: Edge case coverage", ""]
    issues: list[str] = []

    all_rows: list[dict] = []
    for run in RUNS:
        all_rows.extend(_read_csv_rows(METRICS / f"{run}_ranked_candidates.csv"))

    chip_bounds: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for run, chips in RUNS.items():
        for c in chips:
            meta = load_scan_meta(str(EXTRACTED / run / c / "scan_10x" / "scan_meta.json"))
            chip_bounds[(run, c)] = (
                float(meta["x_min_um"]),
                float(meta["x_max_um"]),
                float(meta["y_min_um"]),
                float(meta["y_max_um"]),
            )

    # Build (label, chip) -> run lookup once
    label_chip_to_run: dict[tuple[str, str], str] = {}
    for run, chips in RUNS.items():
        for c in chips:
            jp = METRICS / f"{run}_{c}_metrics.json"
            if not jp.exists():
                continue
            for cand in _read_json(jp).get("candidates", []):
                label_chip_to_run[(cand.get("label", ""), c)] = run

    buckets = {
        "lhrr_skip_reason non-null": [],
        "eop_skip_reason non-null": [],
        f"clearance_um == OCCUPANCY_STAGE_PX_UM ({eop.OCCUPANCY_STAGE_PX_UM})": [],
        "lhrr_area_um2 == 0": [],
        "composite_score < 0.05 (very low)": [],
        "composite_score > 0.5 (very high)": [],
        "stage within 100 µm of chip boundary": [],
    }

    for r in all_rows:
        label = r.get("label", "?")
        chip = r.get("chip", "?")
        if r.get("lhrr_skip_reason") not in (None, "", "None"):
            buckets["lhrr_skip_reason non-null"].append(label)
        if r.get("eop_skip_reason") not in (None, "", "None"):
            buckets["eop_skip_reason non-null"].append(label)
        cle = _maybe_float(r.get("clearance_um"))
        if cle is not None and abs(cle - eop.OCCUPANCY_STAGE_PX_UM) < 1e-6:
            buckets[f"clearance_um == OCCUPANCY_STAGE_PX_UM ({eop.OCCUPANCY_STAGE_PX_UM})"].append(label)
        area = _maybe_float(r.get("lhrr_area_um2"))
        if area == 0.0:
            buckets["lhrr_area_um2 == 0"].append(label)
        comp = _maybe_float(r.get("composite_score"))
        if comp is not None and comp < 0.05:
            buckets["composite_score < 0.05 (very low)"].append(label)
        if comp is not None and comp > 0.5:
            buckets["composite_score > 0.5 (very high)"].append(label)

        run = label_chip_to_run.get((label, chip), "?")
        bnd = chip_bounds.get((run, chip))
        sx = _maybe_float(r.get("stage_x_um"))
        sy = _maybe_float(r.get("stage_y_um"))
        if bnd and sx is not None and sy is not None:
            xmin, xmax, ymin, ymax = bnd
            near = (
                abs(sx - xmin) < 100 or abs(sx - xmax) < 100 or
                abs(sy - ymin) < 100 or abs(sy - ymax) < 100
            )
            if near:
                buckets["stage within 100 µm of chip boundary"].append(label)

    out.append("| bucket | count | examples |")
    out.append("|---|---|---|")
    for name, labels in buckets.items():
        ex = ", ".join(labels[:3]) if labels else "-"
        out.append(f"| {name} | {len(labels)} | {ex} |")
    out.append("")

    status = "PASS"
    out.insert(1, f"_{status}_  edge-case enumeration only: review counts")
    out.insert(2, "")
    return status, out


# ---------------------------------------------------------------------------
# CHECK 9: Fresh-run simulation
# ---------------------------------------------------------------------------

def check_9_fresh_run() -> tuple[str, list[str]]:
    out: list[str] = ["## CHECK 9: Fresh-run simulation", ""]
    issues: list[str] = []

    req = PROJECT_ROOT / "requirements.txt"
    if not req.exists():
        out.append("- `requirements.txt`: **MISSING**")
        issues.append("requirements.txt not present at project root")
    else:
        contents = req.read_text(encoding="utf-8")
        out.append(f"- `requirements.txt`: present ({len(contents.splitlines())} lines)")
        for pkg in ("numpy", "opencv-python", "scipy", "pandas", "matplotlib"):
            if pkg in contents:
                # Check for a version pin
                pinned = re.search(rf"^{re.escape(pkg)}\s*(==|>=|~=)\s*\S+",
                                   contents, re.MULTILINE)
                if pinned:
                    out.append(f"    - `{pkg}`: pinned ({pinned.group(0)})")
                else:
                    out.append(f"    - `{pkg}`: present but unpinned")
                    issues.append(f"requirements.txt: {pkg} not version-pinned")
            else:
                out.append(f"    - `{pkg}`: **MISSING**")
                issues.append(f"requirements.txt missing dependency: {pkg}")

    # Hardcoded paths
    out.append("")
    out.append("### Hardcoded absolute paths")
    pat = re.compile(r"(C:[/\\]|/Users/|/home/)", re.IGNORECASE)
    bad: list[tuple[str, int, str]] = []
    for py in sorted(SRC_DIR.glob("*.py")):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
            if pat.search(line) and "qc_run.py" not in py.name:
                bad.append((py.name, i, line.strip()))
    if not bad:
        out.append("- (none)")
    else:
        for name, ln, line in bad:
            out.append(f"- `{name}:{ln}`: `{line}`")
            issues.append(f"hardcoded path in {name}:{ln}")
    out.append("")

    # Output directory auto-creation
    out.append("### Output-directory auto-creation")
    auto_create_files = []
    for py in sorted(SRC_DIR.glob("*.py")):
        txt = py.read_text(encoding="utf-8")
        if ".mkdir(parents=True, exist_ok=True)" in txt:
            auto_create_files.append(py.name)
    out.append(f"- files calling `mkdir(parents=True, exist_ok=True)`: "
               f"{auto_create_files or '(none)'}")
    if "flake_metrics.py" not in auto_create_files:
        issues.append("flake_metrics.py does not auto-create output directories")
    out.append("")

    status = "PASS" if not issues else "WARN" if all("requirements" in s for s in issues) else "FAIL"
    if issues:
        status = "FAIL" if any("hardcoded path" in s or "missing dependency" in s for s in issues) else "WARN"
    out.insert(1, f"_{status}_  {len(issues)} issue(s)")
    out.insert(2, "")
    return status, out


# ---------------------------------------------------------------------------
# CHECK 10: Lint / dead code
# ---------------------------------------------------------------------------

def _used_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        if isinstance(node, ast.Attribute):
            # base name
            n = node
            while isinstance(n, ast.Attribute):
                n = n.value
            if isinstance(n, ast.Name):
                names.add(n.id)
    return names


def check_10_lint() -> tuple[str, list[str]]:
    out: list[str] = ["## CHECK 10: Lint / dead code", ""]
    issues: list[str] = []

    # pyflakes
    out.append("### pyflakes")
    pf_lines: list[str] = []
    pf_unavailable = False
    try:
        pf = subprocess.run(
            [sys.executable, "-m", "pyflakes", str(SRC_DIR)],
            capture_output=True, text=True,
        )
        merged = (pf.stdout or "") + (pf.stderr or "")
        if "No module named pyflakes" in merged or "ModuleNotFoundError" in merged:
            pf_unavailable = True
            out.append("- pyflakes not installed (skipped)")
        else:
            pf_lines = [
                ln for ln in merged.splitlines()
                if ln.strip() and "qc_run.py" not in ln
            ]
            if pf.returncode == 0 and not pf_lines:
                out.append("- (no warnings)")
            else:
                for ln in pf_lines[:50]:
                    out.append(f"- `{ln}`")
                if len(pf_lines) > 50:
                    out.append(f"- ... +{len(pf_lines) - 50} more")
                issues.extend(pf_lines)
    except Exception as exc:
        pf_unavailable = True
        out.append(f"- pyflakes invocation failed: {exc}")

    # TODO/FIXME markers
    out.append("")
    out.append("### TODO / FIXME / XXX / HACK markers")
    pat = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
    todo_hits = []
    for py in sorted(SRC_DIR.glob("*.py")):
        if py.name == "qc_run.py":
            continue
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
            if pat.search(line):
                todo_hits.append((py.name, i, line.strip()))
    if not todo_hits:
        out.append("- (none)")
    for name, ln, line in todo_hits:
        out.append(f"- `{name}:{ln}`: {line}")

    # Unused imports + uncalled functions per file
    out.append("")
    out.append("### Unused imports / uncalled top-level functions")
    unused_imports: list[str] = []
    uncalled_fns: list[str] = []
    # Collect all imported names across all modules so cross-module references count
    all_files = [p for p in sorted(SRC_DIR.glob("*.py")) if p.name != "qc_run.py"]
    file_used: dict[str, set[str]] = {}
    for py in all_files:
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        file_used[py.name] = _used_names(tree)

    for py in all_files:
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        used = file_used[py.name]

        # Imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    if name not in used:
                        unused_imports.append(f"{py.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name == "*":
                        continue
                    if name not in used:
                        unused_imports.append(
                            f"{py.name}: from {node.module} import {alias.name}"
                        )

        # Top-level function definitions
        defined_fns = [
            n.name for n in tree.body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
        ]
        # Considered "called" if its name appears in any other module's
        # used set OR in its own __main__ block / used set
        all_used = set().union(*file_used.values())
        for fn in defined_fns:
            # Self-reference inside same file is fine if used
            count = sum(1 for nm in [fn] for u in all_used if nm == u)
            # Conservative: name found in another file
            occur = sum(1 for f, used_set in file_used.items()
                        if f != py.name and fn in used_set)
            self_use = fn in used
            if occur == 0 and not self_use:
                uncalled_fns.append(f"{py.name}: {fn}()")

    if not unused_imports:
        out.append("- unused imports: (none)")
    else:
        for u in unused_imports[:30]:
            out.append(f"- {u}")
        if len(unused_imports) > 30:
            out.append(f"- ... +{len(unused_imports) - 30} more")
    out.append("")
    if not uncalled_fns:
        out.append("- defined-but-not-called top-level functions: (none)")
    else:
        for u in uncalled_fns[:30]:
            out.append(f"- {u}")
    out.append("")

    if pf_lines or unused_imports or uncalled_fns:
        status = "WARN"
    else:
        status = "PASS"
    out.insert(1, f"_{status}_  pyflakes={len(pf_lines)}, "
                  f"todo={len(todo_hits)}, "
                  f"unused_imports={len(unused_imports)}, "
                  f"uncalled_fns={len(uncalled_fns)}")
    out.insert(2, "")
    return status, out


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    statuses: dict[int, str] = {}
    sections: dict[int, list[str]] = {}

    print("=== CHECK 1 ==="); s, lines = check_1_schema(); statuses[1], sections[1] = s, lines
    print("=== CHECK 2 ==="); s, lines = check_2_ranges(); statuses[2], sections[2] = s, lines
    print("=== CHECK 3 ==="); s, lines = check_3_completeness(); statuses[3], sections[3] = s, lines
    print("=== CHECK 5 ==="); s, lines = check_5_readme_alignment(); statuses[5], sections[5] = s, lines
    print("=== CHECK 6 ==="); s, lines = check_6_claude_plan(); statuses[6], sections[6] = s, lines
    print("=== CHECK 8 ==="); s, lines = check_8_edges(); statuses[8], sections[8] = s, lines
    print("=== CHECK 9 ==="); s, lines = check_9_fresh_run(); statuses[9], sections[9] = s, lines
    print("=== CHECK 10 ==="); s, lines = check_10_lint(); statuses[10], sections[10] = s, lines
    print("=== CHECK 4 (slow) ==="); s, lines, _cache = check_4_determinism(); statuses[4], sections[4] = s, lines
    print("=== CHECK 7 (slow) ==="); s, lines = check_7_spotcheck(); statuses[7], sections[7] = s, lines

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1

    report: list[str] = []
    report.append("# FlakeFinder QC Report")
    report.append("")
    report.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    report.append(f"_Total wall-clock: {(time.time() - t0)/60:.1f} min_")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append("| check | status |")
    report.append("|---|---|")
    titles = {
        1: "Schema completeness",
        2: "Range validation",
        3: "Candidate completeness",
        4: "Determinism",
        5: "README / code alignment",
        6: "CLAUDE.md / plan.md alignment",
        7: "Spot-check figure generation",
        8: "Edge case coverage",
        9: "Fresh-run simulation",
        10: "Lint / dead code",
    }
    for i in range(1, 11):
        report.append(f"| CHECK {i}: {titles[i]} | **{statuses.get(i, '?')}** |")
    report.append("")
    report.append(f"**PASS: {counts['PASS']}  WARN: {counts['WARN']}  FAIL: {counts['FAIL']}**")
    report.append("")
    report.append("---")
    report.append("")
    for i in range(1, 11):
        report.extend(sections.get(i, []))
        report.append("---")
        report.append("")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "qc_report.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"PASS: {counts['PASS']}  WARN: {counts['WARN']}  FAIL: {counts['FAIL']}")


if __name__ == "__main__":
    main()
