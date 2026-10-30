"""Interactive Cardiac Research Dashboard & Multi-Cohort Exploration Server.

Provides a research-grade web dashboard for clinical CT cohorts (including Stanford
COCA, Harvard/MGH CardiacFatQuant, and multi-center clinical trials):
  - Cohort Overview & Clinical KPI Scorecards
  - High-impact interactive research correlation plots (EAT & PVAT vs Agatston Calcium Score)
  - Cohort Case Explorer with live search, risk filtering, and status tracking
  - Extensible DICE & multi-compartment segmentation benchmarking engine
  - Direct integration with local OHIF 2D MPR / 3D Slicer viewers and clinical PDF reports
"""

from __future__ import annotations

import io
import json
import math
import os
import threading
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file

from cardiac_fat_fl.cohort_research import CohortDataset, CohortParticipant
from cardiac_fat_fl.io_metatensor import read_nrrd_as_metatensor
from cardiac_fat_fl.measurement_schema import (
    ADIPOSE_HU_MAX,
    ADIPOSE_HU_MIN,
    MEASUREMENT_CSV_COLUMNS,
    REPORTED_REGIONS,
    SEGMENTATION_COMPARTMENTS,
)
from cardiac_fat_fl.segmentation_evaluator import (
    CohortBenchmarkReport,
    SegmentationBenchmarkEngine,
)

# Global holder for active cohort and benchmark report
_ACTIVE_COHORT: Optional[CohortDataset] = None
_LATEST_BENCHMARK: Optional[CohortBenchmarkReport] = None


def get_default_cohort_dir() -> Path:
    """Resolve default cohort directory from env var, or fallback to example_source_dir."""
    env_dir = os.environ.get("CARDIAC_COHORT_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir).resolve()

    example_dir = Path("example_source_dir").resolve()
    if example_dir.exists():
        return example_dir

    return Path(".").resolve()


def sanitize_nans(obj: Any) -> Any:
    """Recursively convert float NaN/Inf to None so JSON outputs are strictly RFC 8259 compliant."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nans(v) for v in obj]
    return obj


def create_dashboard_app(cohort_dir: Optional[Union[str, Path]] = None) -> Flask:
    """Create and configure the Flask Cardiac Research Dashboard application."""
    global _ACTIVE_COHORT

    app = Flask(__name__)
    target_dir = Path(cohort_dir).resolve() if cohort_dir else get_default_cohort_dir()

    try:
        _ACTIVE_COHORT = CohortDataset(target_dir)
    except Exception as e:
        print(f"Warning: Failed to initialize CohortDataset from {target_dir}: {e}")
        _ACTIVE_COHORT = None

    @app.route("/")
    def index() -> str:
        return render_template_string(DASHBOARD_HTML_TEMPLATE)

    @app.route("/api/cohort/summary", methods=["GET"])
    def get_cohort_summary():
        if _ACTIVE_COHORT is None:
            return jsonify({"error": "No active cohort loaded"}), 400
        summary = _ACTIVE_COHORT.compute_summary_analytics()
        summary["root_dir"] = str(_ACTIVE_COHORT.root_dir)
        return jsonify(sanitize_nans(summary))

    @app.route("/api/cohort/scatter", methods=["GET"])
    def get_scatter_data():
        if _ACTIVE_COHORT is None:
            return jsonify([]), 400
        return jsonify(sanitize_nans(_ACTIVE_COHORT.get_scatter_data()))

    @app.route("/api/cohort/cases", methods=["GET"])
    def get_cohort_cases():
        if _ACTIVE_COHORT is None:
            return jsonify([]), 400

        search_q = request.args.get("search", "").lower().strip()
        tier_filter = request.args.get("tier", "").strip()
        has_seg_only = request.args.get("seg_only", "").lower() in ["true", "1"]
        inflam_only = request.args.get("inflam_only", "").lower() in ["true", "1"]

        cases = []
        for p in _ACTIVE_COHORT.participants.values():
            if search_q and search_q not in p.participant_id.lower():
                continue
            if tier_filter and p.calcium_risk_tier != tier_filter:
                continue
            if has_seg_only and p.seg_path is None:
                continue
            if inflam_only and p.crisp_ct_status != "Active Vascular Inflammation":
                continue

            cases.append({
                "participant_id": p.participant_id,
                "scan_type": p.scan_type,
                "has_seg": p.seg_path is not None,
                "calcium_score": p.calcium_score,
                "calcium_risk_tier": p.calcium_risk_tier,
                "crisp_ct_status": p.crisp_ct_status,
                "eat_volume_cm3": p.metrics.get("EAT Total Volume (cm3)"),
                "pvat_volume_cm3": p.metrics.get("PVAT Total Volume (cm3)"),
                "pvat_mean_hu": p.metrics.get("PVAT Total Mean Attenuation (HU)"),
                "has_report_pdf": p.report_pdf is not None,
                "has_report_html": p.report_html is not None,
                "has_ohif": p.ohif_export_dir is not None,
                "available_segs": list(p.available_segmentations.keys()),
            })

        return jsonify(sanitize_nans(cases))

    @app.route("/api/case/<pid>", methods=["GET"])
    def get_case_details(pid: str):
        if _ACTIVE_COHORT is None or pid not in _ACTIVE_COHORT.participants:
            return jsonify({"error": f"Participant {pid} not found"}), 404
        p = _ACTIVE_COHORT.participants[pid]
        return jsonify(sanitize_nans(p.to_dict()))

    @app.route("/api/cohort/set_directory", methods=["POST"])
    def set_cohort_directory():
        global _ACTIVE_COHORT
        data = request.get_json(silent=True) or {}
        new_path = data.get("cohort_dir")
        if not new_path:
            return jsonify({"error": "cohort_dir parameter is required"}), 400

        target = Path(new_path).resolve()
        if not target.exists():
            return jsonify({"error": f"Directory not found: {target}"}), 404

        try:
            _ACTIVE_COHORT = CohortDataset(target)
            return jsonify({
                "success": True,
                "root_dir": str(target),
                "total_count": _ACTIVE_COHORT.total_count,
                "segmented_count": _ACTIVE_COHORT.segmented_count,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/benchmark/run", methods=["POST"])
    def run_benchmark():
        global _LATEST_BENCHMARK, _ACTIVE_COHORT
        if _ACTIVE_COHORT is None:
            return jsonify({"error": "No active cohort loaded"}), 400

        data = request.get_json(silent=True) or {}
        candidate_name = data.get("candidate_seg", "eat_pvat.seg.nrrd")
        reference_name = data.get("reference_seg", "ground_truth.seg.nrrd")
        limit = int(data.get("limit", 50))

        engine = SegmentationBenchmarkEngine(
            candidate_name=candidate_name,
            reference_name=reference_name,
            compute_surface_distances=False,
        )

        pairs: List[Any] = []
        for pid, p in _ACTIVE_COHORT.participants.items():
            if len(pairs) >= limit:
                break
            # Find candidate and reference files in patient folder
            p_dir = p.study_dir
            cand_path = p_dir / candidate_name if not (p_dir / candidate_name).is_file() else (p_dir / candidate_name)
            ref_path = p_dir / reference_name if not (p_dir / reference_name).is_file() else (p_dir / reference_name)

            if not cand_path.is_file():
                # check aliases
                for fname in p.available_segmentations.values():
                    if candidate_name.lower() in fname.name.lower():
                        cand_path = fname
                        break
            if not ref_path.is_file():
                for fname in p.available_segmentations.values():
                    if reference_name.lower() in fname.name.lower():
                        ref_path = fname
                        break

            if cand_path.is_file() and ref_path.is_file():
                try:
                    c_mt = read_nrrd_as_metatensor(cand_path)
                    r_mt = read_nrrd_as_metatensor(ref_path)
                    c_arr = c_mt.detach().cpu().numpy()
                    r_arr = r_mt.detach().cpu().numpy()
                    sp = tuple(c_mt.meta.get("spacing", (1.0, 1.0, 1.0)))
                    pairs.append((pid, c_arr, r_arr, sp))
                except Exception as e:
                    print(f"Error loading {pid} masks: {e}")

        if not pairs:
            return jsonify({
                "error": f"No matching cases found containing both '{candidate_name}' and '{reference_name}'.",
                "searched_cohort_cases": len(_ACTIVE_COHORT.participants),
            }), 404

        report = engine.evaluate_cohort(pairs)
        _LATEST_BENCHMARK = report
        return jsonify(report.to_dict())

    @app.route("/api/benchmark/results", methods=["GET"])
    def get_benchmark_results():
        if _LATEST_BENCHMARK is None:
            return jsonify({"status": "no_benchmark_run"}), 200
        return jsonify(_LATEST_BENCHMARK.to_dict())

    @app.route("/api/benchmark/export", methods=["GET"])
    def export_benchmark():
        if _LATEST_BENCHMARK is None:
            return jsonify({"error": "No benchmark results available to export"}), 404

        fmt = request.args.get("format", "csv").lower()
        if fmt == "json":
            return Response(
                json.dumps(_LATEST_BENCHMARK.to_dict(), indent=2),
                mimetype="application/json",
                headers={"Content-Disposition": "attachment; filename=cardiac_segmentation_benchmark.json"},
            )

        # CSV format
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "cardiac_segmentation_benchmark.csv"
        SegmentationBenchmarkEngine.export_summary_csv(_LATEST_BENCHMARK, tmp)
        return send_file(
            str(tmp),
            mimetype="text/csv",
            as_attachment=True,
            download_name="cardiac_segmentation_benchmark.csv",
        )

    return app


def launch_dashboard(
    cohort_dir: Optional[Union[str, Path]] = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    auto_open: bool = True,
) -> None:
    """Launch the interactive Cardiac Research Dashboard server."""
    app = create_dashboard_app(cohort_dir=cohort_dir)
    url = f"http://{host}:{port}/"
    print(f"\n========================================================")
    print(f"  Cardiac Research Dashboard (MONAI / FLARE / OHIF)")
    print(f"  Server URL: {url}")
    print(f"========================================================\n")

    if auto_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False, threaded=True)


DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cardiac Research Dashboard | EAT & PVAT Analytics</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .tab-active { border-bottom: 2px solid #0284c7; color: #0284c7; font-weight: 600; }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col">

  <!-- Top Navigation Header -->
  <header class="bg-slate-800 border-b border-slate-700 px-6 py-4 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-50 shadow-md">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center text-white shadow-lg">
        <i class="fa-solid fa-heart-pulse text-xl"></i>
      </div>
      <div>
        <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
          Cardiac Fat Research Dashboard
          <span class="text-xs px-2 py-0.5 rounded bg-sky-900/60 border border-sky-500/40 text-sky-300 font-mono">v2.5+</span>
        </h1>
        <p class="text-xs text-slate-400">Automated EAT/PVAT Quantification, Agatston CAC Risk & DICE Benchmarking</p>
      </div>
    </div>

    <!-- Active Cohort Info & Actions -->
    <div class="flex items-center gap-3">
      <div class="bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-1.5 text-xs text-slate-300 flex items-center gap-2">
        <i class="fa-regular fa-folder-open text-sky-400"></i>
        <span class="font-mono truncate max-w-xs" id="activeCohortPathDisplay">Loading cohort...</span>
        <button onclick="openCohortModal()" class="ml-1 text-sky-400 hover:text-sky-300 font-medium underline">Switch</button>
      </div>
      <button onclick="refreshData()" class="bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs px-3 py-2 rounded-lg flex items-center gap-1.5 transition">
        <i class="fa-solid fa-arrows-rotate"></i> Refresh
      </button>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="bg-slate-800/60 border-b border-slate-700/60 px-6">
    <nav class="flex gap-8 text-sm">
      <button onclick="switchTab('analytics')" id="tab-analytics" class="py-3 px-1 tab-active transition flex items-center gap-2">
        <i class="fa-solid fa-chart-line"></i> Cohort Research Analytics
      </button>
      <button onclick="switchTab('cases')" id="tab-cases" class="py-3 px-1 text-slate-400 hover:text-slate-200 transition flex items-center gap-2">
        <i class="fa-solid fa-hospital-user"></i> Case Explorer (<span id="casesCountPill">0</span>)
      </button>
      <button onclick="switchTab('benchmark')" id="tab-benchmark" class="py-3 px-1 text-slate-400 hover:text-slate-200 transition flex items-center gap-2">
        <i class="fa-solid fa-scale-balanced"></i> DICE & Method Benchmarking
      </button>
    </nav>
  </div>

  <!-- Main Container -->
  <main class="flex-1 p-6 max-w-7xl mx-auto w-full">

    <!-- ================= TAB 1: COHORT ANALYTICS ================= -->
    <section id="section-analytics" class="space-y-6">
      
      <!-- KPI Scorecards Row 1: Pipeline & Coverage (4 cards) -->
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-3.5">
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3.5 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Total Cohort (N)</div>
          <div class="text-2xl font-bold text-white mt-1" id="kpiTotal">--</div>
          <div class="text-[11px] text-slate-400 mt-1">Discovered CT Cases</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3.5 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Quantified (N)</div>
          <div class="text-2xl font-bold text-emerald-400 mt-1" id="kpiQuantified">--</div>
          <div class="text-[11px] text-emerald-500/80 mt-1" id="kpiQuantifiedPct">--% metrics in CSV</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3.5 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">3D Masks on Disk (N)</div>
          <div class="text-2xl font-bold text-cyan-400 mt-1" id="kpiMasks">--</div>
          <div class="text-[11px] text-cyan-500/80 mt-1" id="kpiMasksPct">--% .seg.nrrd</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3.5 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">CAC Scored (N)</div>
          <div class="text-2xl font-bold text-amber-400 mt-1" id="kpiCac">--</div>
          <div class="text-[11px] text-amber-500/80 mt-1" id="kpiCacPct">--% of cohort</div>
        </div>
      </div>

      <!-- KPI Scorecards Row 2: Adipose Biomarkers (5 cards: Mean EAT Vol, Mean EAT HU, Mean PVAT Vol, Mean PVAT HU, Inflammation) -->
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3.5">
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3.5 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold flex items-center justify-between">
            <span>Mean EAT Volume</span>
            <span class="text-[10px] text-slate-500 font-mono">cm³</span>
          </div>
          <div class="text-2xl font-bold text-sky-400 mt-1" id="kpiEatVol">--</div>
          <div class="text-[11px] text-slate-400 mt-1 truncate" id="kpiEatVolSub">Median: -- (IQR: --)</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3.5 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold flex items-center justify-between">
            <span>Mean EAT HU</span>
            <span class="text-[10px] text-slate-500 font-mono">HU</span>
          </div>
          <div class="text-2xl font-bold text-sky-300 mt-1" id="kpiEatHu">--</div>
          <div class="text-[11px] text-slate-400 mt-1 truncate" id="kpiEatHuSub">SD: ±-- • Med: --</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3.5 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold flex items-center justify-between">
            <span>Mean PVAT Volume</span>
            <span class="text-[10px] text-slate-500 font-mono">cm³</span>
          </div>
          <div class="text-2xl font-bold text-indigo-400 mt-1" id="kpiPvatVol">--</div>
          <div class="text-[11px] text-slate-400 mt-1 truncate" id="kpiPvatVolSub">Median: -- (IQR: --)</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3.5 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold flex items-center justify-between">
            <span>Mean PVAT HU</span>
            <span class="text-[10px] text-slate-500 font-mono">HU</span>
          </div>
          <div class="text-2xl font-bold text-indigo-300 mt-1" id="kpiPvatHu">--</div>
          <div class="text-[11px] text-slate-400 mt-1 truncate" id="kpiPvatHuSub">SD: ±-- • Med: --</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3.5 shadow col-span-2 md:col-span-1">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold flex items-center justify-between">
            <span>Vascular Inflammation (N)</span>
            <i class="fa-solid fa-fire text-rose-400 text-xs"></i>
          </div>
          <div class="text-2xl font-bold text-rose-400 mt-1" id="kpiInflam">--</div>
          <div class="text-[11px] text-rose-500/80 mt-1" id="kpiInflamPct">CRISP-CT > -70.1 HU</div>
        </div>
      </div>

      <!-- Clinical Research Correlation Summary -->
      <div class="bg-slate-800/70 border border-slate-700 rounded-xl p-4 shadow flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-lg bg-sky-500/20 text-sky-400 flex items-center justify-center">
            <i class="fa-solid fa-flask"></i>
          </div>
          <div>
            <h3 class="text-sm font-semibold text-white">Adipose Phenotyping & Calcium Correlation Statistics</h3>
            <p class="text-xs text-slate-400">Pearson and Spearman rank association with coronary calcification</p>
          </div>
        </div>
        <div class="flex items-center gap-6 text-xs">
          <div>
            <span class="text-slate-400">EAT vs. Agatston CAC:</span>
            <span class="font-mono font-bold text-sky-300 ml-1" id="corrEatCac">--</span>
          </div>
          <div>
            <span class="text-slate-400">PVAT HU vs. Agatston CAC:</span>
            <span class="font-mono font-bold text-indigo-300 ml-1" id="corrPvatCac">--</span>
          </div>
        </div>
      </div>

      <!-- Charts Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        <!-- Chart 1: Agatston Calcium Score vs EAT Volume -->
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow flex flex-col">
          <div class="flex items-center justify-between mb-3 flex-wrap gap-2">
            <div>
              <h3 class="text-sm font-bold text-white flex items-center gap-2" id="titleChartEat">
                <i class="fa-solid fa-chart-line text-sky-400"></i>
                Agatston CAC vs. Total EAT Volume
              </h3>
              <p class="text-xs text-slate-400" id="subChartEat">Coronary calcification as a function of epicardial fat volume</p>
            </div>
            <div class="flex items-center gap-1.5 text-xs">
              <!-- Swap Axes -->
              <button onclick="toggleAxisSwap('eat')" title="Swap X and Y Axes" class="px-2 py-1 bg-slate-700/80 hover:bg-slate-700 text-slate-300 rounded text-[11px] flex items-center gap-1 transition">
                <i class="fa-solid fa-right-left text-[10px]"></i> Swap
              </button>
              <!-- Scale toggle -->
              <div class="inline-flex rounded-lg bg-slate-900 p-0.5 border border-slate-700">
                <button id="btnScaleEatLin" onclick="setScaleMode('eat', 'linear')" class="px-2 py-0.5 rounded text-[11px] font-medium bg-sky-600 text-white transition">Linear</button>
                <button id="btnScaleEatLog" onclick="setScaleMode('eat', 'logarithmic')" class="px-2 py-0.5 rounded text-[11px] font-medium text-slate-400 hover:text-white transition">Log₁₀</button>
              </div>
              <!-- Reset Zoom -->
              <button onclick="resetChartZoom('eat')" title="Reset Zoom" class="px-2 py-1 bg-slate-700/80 hover:bg-slate-700 text-slate-300 rounded text-[11px] flex items-center gap-1 transition">
                <i class="fa-solid fa-arrows-rotate text-[10px]"></i> Reset
              </button>
              <!-- Pop-out Studio -->
              <button onclick="openChartModal('eat')" title="Pop-out Studio View" class="p-1.5 bg-slate-700/80 hover:bg-slate-700 text-slate-300 hover:text-white rounded text-xs transition">
                <i class="fa-solid fa-expand"></i>
              </button>
            </div>
          </div>
          <div class="relative flex-1 min-h-[300px]">
            <canvas id="chartEatVsCac"></canvas>
          </div>
        </div>

        <!-- Chart 2: Agatston CAC vs PVAT Attenuation -->
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow flex flex-col">
          <div class="flex items-center justify-between mb-3 flex-wrap gap-2">
            <div>
              <h3 class="text-sm font-bold text-white flex items-center gap-2" id="titleChartPvat">
                <i class="fa-solid fa-fire-flame-curved text-rose-400"></i>
                Agatston CAC vs. PVAT Attenuation
              </h3>
              <p class="text-xs text-slate-400" id="subChartPvat">Coronary calcification stratified by pericoronary inflammation</p>
            </div>
            <div class="flex items-center gap-1.5 text-xs">
              <!-- Swap Axes -->
              <button onclick="toggleAxisSwap('pvat')" title="Swap X and Y Axes" class="px-2 py-1 bg-slate-700/80 hover:bg-slate-700 text-slate-300 rounded text-[11px] flex items-center gap-1 transition">
                <i class="fa-solid fa-right-left text-[10px]"></i> Swap
              </button>
              <!-- Scale toggle -->
              <div class="inline-flex rounded-lg bg-slate-900 p-0.5 border border-slate-700">
                <button id="btnScalePvatLin" onclick="setScaleMode('pvat', 'linear')" class="px-2 py-0.5 rounded text-[11px] font-medium bg-rose-600 text-white transition">Linear</button>
                <button id="btnScalePvatLog" onclick="setScaleMode('pvat', 'logarithmic')" class="px-2 py-0.5 rounded text-[11px] font-medium text-slate-400 hover:text-white transition">Log₁₀</button>
              </div>
              <!-- Reset Zoom -->
              <button onclick="resetChartZoom('pvat')" title="Reset Zoom" class="px-2 py-1 bg-slate-700/80 hover:bg-slate-700 text-slate-300 rounded text-[11px] flex items-center gap-1 transition">
                <i class="fa-solid fa-arrows-rotate text-[10px]"></i> Reset
              </button>
              <!-- Pop-out Studio -->
              <button onclick="openChartModal('pvat')" title="Pop-out Studio View" class="p-1.5 bg-slate-700/80 hover:bg-slate-700 text-slate-300 hover:text-white rounded text-xs transition">
                <i class="fa-solid fa-expand"></i>
              </button>
            </div>
          </div>
          <div class="relative flex-1 min-h-[300px]">
            <canvas id="chartPvatVsCac"></canvas>
          </div>
        </div>

        <!-- Chart 3: Compartment Volumes -->
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow flex flex-col">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h3 class="text-sm font-bold text-white flex items-center gap-2">
                <i class="fa-solid fa-chart-column text-indigo-400"></i>
                Adipose Volume Distribution by Compartment
              </h3>
              <p class="text-xs text-slate-400">Mean volume (cm³) across chambers and coronary vessels</p>
            </div>
          </div>
          <div class="relative flex-1 min-h-[280px]">
            <canvas id="chartCompartments"></canvas>
          </div>
        </div>

        <!-- Chart 4: CAC Risk Tier Breakdown -->
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow flex flex-col">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h3 class="text-sm font-bold text-white flex items-center gap-2">
                <i class="fa-solid fa-chart-pie text-amber-400"></i>
                Cohort Coronary Calcium Risk Distribution
              </h3>
              <p class="text-xs text-slate-400">Clinical categorization across cohort</p>
            </div>
          </div>
          <div class="relative flex-1 min-h-[280px] flex items-center justify-center">
            <canvas id="chartRiskTiers"></canvas>
          </div>
        </div>

      </div>

      <!-- Explanatory Scientific Guide: Vascular Inflammation & Pericoronary Fat Phenotyping -->
      <div class="bg-slate-800/90 border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex items-center justify-between flex-wrap gap-3 pb-3 border-b border-slate-700/80">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-rose-500/20 text-rose-400 flex items-center justify-center text-lg shadow">
              <i class="fa-solid fa-fire-flame-curved"></i>
            </div>
            <div>
              <h3 class="text-base font-bold text-white flex items-center gap-2">
                Clinical Reference: Vascular Inflammation & Pericoronary Attenuation (CRISP-CT)
                <span class="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30 font-semibold">Biomarker Standard</span>
              </h3>
              <p class="text-xs text-slate-400">How perivascular adipose attenuation (HU) detects coronary plaque vulnerability and cardiac risk</p>
            </div>
          </div>
          <div class="flex items-center gap-2 text-xs">
            <span class="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700 text-slate-300 font-mono">
              Cutoff: <strong class="text-rose-400">&gt; -70.1 HU</strong>
            </span>
            <span class="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700 text-slate-300 font-mono">
              Lancet 2018 (CRISP-CT)
            </span>
          </div>
        </div>

        <!-- 3-Column Explanatory Grid -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs text-slate-300">
          
          <!-- Column 1: Biological Mechanism -->
          <div class="bg-slate-900/60 border border-slate-700/70 rounded-xl p-4 space-y-2">
            <div class="font-bold text-sky-400 flex items-center gap-2 text-sm">
              <i class="fa-solid fa-dna text-xs"></i> 1. Biological Mechanism
            </div>
            <p class="leading-relaxed text-slate-300">
              Coronary atherosclerosis is driven by vascular wall inflammation. Through <strong class="text-white">paracrine vascular-adipose crosstalk</strong>, inflammatory cytokines (such as TNF-α, IL-6, and IFN-γ) diffuse outward from the inflamed coronary vessel into adjacent pericoronary fat (PVAT).
            </p>
            <p class="leading-relaxed text-slate-400">
              This inflammatory gradient inhibits preadipocyte lipid accumulation (adipogenesis) and triggers local microvascular remodeling and interstitial edema, reducing adipocyte size and lipid density.
            </p>
          </div>

          <!-- Column 2: Radiologic Derivation on CT -->
          <div class="bg-slate-900/60 border border-slate-700/70 rounded-xl p-4 space-y-2">
            <div class="font-bold text-emerald-400 flex items-center gap-2 text-sm">
              <i class="fa-solid fa-microscope text-xs"></i> 2. Automated 3D CT Derivation
            </div>
            <p class="leading-relaxed text-slate-300">
              Quiescent mature fat is lipid-dense and attenuates CT X-rays strongly negatively (typically <span class="font-mono text-emerald-400">-90 to -75 HU</span>). Water-rich edema and cellular infiltration shift attenuation upwards toward zero (<span class="font-mono text-white">0 HU</span>).
            </p>
            <ul class="space-y-1 text-slate-400 list-disc list-inside">
              <li>Geometric derivation of LAD, LCX, and RCA coronary centerlines.</li>
              <li>Anisotropic 3D Euclidean Distance Transform (EDT) radial dilation of <strong class="text-slate-300">10 mm</strong> around each vessel.</li>
              <li>Voxel extraction restricted strictly to adipose HU window <span class="font-mono text-slate-300">[-190, -30] HU</span>.</li>
            </ul>
          </div>

          <!-- Column 3: Clinical Validation & Interpretation -->
          <div class="bg-slate-900/60 border border-slate-700/70 rounded-xl p-4 space-y-2">
            <div class="font-bold text-rose-400 flex items-center gap-2 text-sm">
              <i class="fa-solid fa-heart-pulse text-xs"></i> 3. Clinical Trial Evidence (CRISP-CT)
            </div>
            <p class="leading-relaxed text-slate-300">
              In the landmark Oxford <strong class="text-white">CRISP-CT study</strong> (<em>The Lancet</em> 2018; 392: 1314–21; N=3,912), pericoronary attenuation <strong class="text-rose-400">&gt; -70.1 HU</strong> independently predicted fatal and non-fatal myocardial infarction, identifying vulnerable plaque biology before stenosis occurs.
            </p>
            <div class="pt-1.5 border-t border-slate-800 text-[11px] text-slate-400">
              In this cohort: <strong class="text-rose-400 font-mono text-xs" id="explActiveCount">--</strong> cases (<span id="explActivePct" class="font-mono">--%</span>) exceed -70.1 HU, flagging active coronaritis independent of anatomical calcium score.
            </div>
          </div>

        </div>
      </div>
    </section>

    <!-- ================= TAB 2: CASE EXPLORER ================= -->
    <section id="section-cases" class="hidden space-y-4">
      <!-- Search & Filters -->
      <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center gap-3 flex-1 min-w-[280px]">
          <div class="relative flex-1">
            <i class="fa-solid fa-magnifying-glass absolute left-3 top-3 text-slate-400 text-xs"></i>
            <input type="text" id="caseSearchInput" oninput="applyCaseFilters()" placeholder="Search by participant ID..." 
                   class="bg-slate-900 border border-slate-700 rounded-lg pl-8 pr-4 py-1.5 text-xs text-white w-full focus:outline-none focus:border-sky-500">
          </div>
        </div>
        <div class="flex items-center gap-2 text-xs">
          <select id="caseTierFilter" onchange="applyCaseFilters()" class="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-300 focus:outline-none">
            <option value="">All CAC Risk Tiers</option>
            <option value="Zero (0)">Zero (0)</option>
            <option value="Minimal (1-10)">Minimal (1-10)</option>
            <option value="Mild (11-100)">Mild (11-100)</option>
            <option value="Moderate (101-400)">Moderate (101-400)</option>
            <option value="Severe (>400)">Severe (>400)</option>
          </select>
          <label class="flex items-center gap-1.5 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-300 cursor-pointer">
            <input type="checkbox" id="caseSegOnly" onchange="applyCaseFilters()" class="rounded bg-slate-800 text-sky-500">
            <span>Segmented Only</span>
          </label>
          <label class="flex items-center gap-1.5 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-300 cursor-pointer">
            <input type="checkbox" id="caseInflamOnly" onchange="applyCaseFilters()" class="rounded bg-slate-800 text-rose-500">
            <span>Inflammation Only</span>
          </label>
        </div>
      </div>

      <!-- Cases Table -->
      <div class="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow">
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-900/60 uppercase tracking-wider font-semibold text-slate-400 border-b border-slate-700">
              <tr>
                <th class="px-4 py-3">Participant ID</th>
                <th class="px-4 py-3">Scan Type</th>
                <th class="px-4 py-3">Agatston CAC</th>
                <th class="px-4 py-3">EAT Volume</th>
                <th class="px-4 py-3">PVAT Volume</th>
                <th class="px-4 py-3">PVAT Mean HU</th>
                <th class="px-4 py-3">CRISP-CT Status</th>
                <th class="px-4 py-3">Artifacts</th>
                <th class="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody id="casesTableBody" class="divide-y divide-slate-700/60">
              <tr><td colspan="9" class="text-center py-8 text-slate-400">Loading cases...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- ================= TAB 3: DICE & BENCHMARKING ================= -->
    <section id="section-benchmark" class="hidden space-y-6">
      <!-- Benchmark Configuration Card -->
      <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow">
        <h3 class="text-sm font-bold text-white mb-2 flex items-center gap-2">
          <i class="fa-solid fa-vial text-sky-400"></i>
          Extensible Segmentation Method Benchmarking
        </h3>
        <p class="text-xs text-slate-400 mb-4">
          Compare multi-compartment candidate segmentations against ground truth or alternative algorithms (e.g. Heuristic vs DeepHeartSeg ONNX, or SegResNet candidate vs Ground Truth).
        </p>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          <div>
            <label class="block text-slate-300 mb-1 font-semibold">Candidate Segmentation Mask</label>
            <input type="text" id="bmCandidateName" value="eat_pvat.seg.nrrd" 
                   class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-sky-500">
          </div>
          <div>
            <label class="block text-slate-300 mb-1 font-semibold">Reference / Ground Truth Mask</label>
            <input type="text" id="bmReferenceName" value="ground_truth.seg.nrrd" 
                   class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-sky-500">
          </div>
          <div>
            <label class="block text-slate-300 mb-1 font-semibold">Max Case Limit</label>
            <input type="number" id="bmCaseLimit" value="50" min="1" max="1000" 
                   class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-sky-500">
          </div>
          <div class="flex items-end gap-2">
            <button onclick="runBenchmarkEvaluation()" id="btnRunBenchmark" class="flex-1 bg-sky-600 hover:bg-sky-500 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition shadow">
              <i class="fa-solid fa-play"></i> Run Benchmark
            </button>
          </div>
        </div>
      </div>

      <!-- Benchmark Results View -->
      <div id="benchmarkResultsContainer" class="space-y-6">
        <div class="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-8 text-center text-slate-400 text-xs">
          <i class="fa-solid fa-scale-balanced text-3xl mb-2 text-slate-500"></i>
          <p>No benchmark run yet. Select candidate and reference segmentation filenames above and click "Run Benchmark".</p>
        </div>
      </div>
    </section>

  </main>

  <!-- Modal: Case Detail Deep Dive -->
  <div id="caseModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm hidden z-50 flex items-center justify-center p-4">
    <div class="bg-slate-800 border border-slate-700 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4 max-h-[90vh] overflow-y-auto">
      <div class="flex items-center justify-between border-b border-slate-700 pb-3">
        <h3 class="text-base font-bold text-white flex items-center gap-2">
          <i class="fa-solid fa-user-doctor text-sky-400"></i>
          Case Deep Dive: <span id="modalCaseId" class="font-mono text-sky-300">--</span>
        </h3>
        <button onclick="closeCaseModal()" class="text-slate-400 hover:text-white text-lg"><i class="fa-solid fa-xmark"></i></button>
      </div>

      <div class="grid grid-cols-3 gap-3 text-xs">
        <div class="bg-slate-900 p-3 rounded-lg border border-slate-700/60">
          <div class="text-slate-400">Scan Type</div>
          <div class="text-white font-semibold mt-0.5" id="modalScanType">--</div>
        </div>
        <div class="bg-slate-900 p-3 rounded-lg border border-slate-700/60">
          <div class="text-slate-400">Agatston Score</div>
          <div class="text-amber-400 font-semibold mt-0.5" id="modalCacScore">--</div>
        </div>
        <div class="bg-slate-900 p-3 rounded-lg border border-slate-700/60">
          <div class="text-slate-400">CRISP-CT Status</div>
          <div class="text-white font-semibold mt-0.5" id="modalCrispStatus">--</div>
        </div>
      </div>

      <div class="border-t border-slate-700 pt-3">
        <h4 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Quantification Scorecard (19 Metrics)</h4>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-900/60 text-slate-400">
              <tr>
                <th class="p-2">Region</th>
                <th class="p-2 text-right">Volume (cm³)</th>
                <th class="p-2 text-right">Mean Attenuation (HU)</th>
              </tr>
            </thead>
            <tbody id="modalMetricsBody" class="divide-y divide-slate-700/50">
              <!-- populated dynamically -->
            </tbody>
          </table>
        </div>
      </div>

      <div class="flex items-center justify-end gap-3 pt-4 border-t border-slate-700 text-xs">
        <button onclick="closeCaseModal()" class="px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200">Close</button>
      </div>
    </div>
  </div>

  <!-- Modal: Switch Cohort Directory -->
  <div id="cohortModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm hidden z-50 flex items-center justify-center p-4">
    <div class="bg-slate-800 border border-slate-700 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4">
      <div class="flex items-center justify-between border-b border-slate-700 pb-3">
        <h3 class="text-base font-bold text-white flex items-center gap-2">
          <i class="fa-regular fa-folder-open text-sky-400"></i> Switch Cohort Directory
        </h3>
        <button onclick="closeCohortModal()" class="text-slate-400 hover:text-white text-lg"><i class="fa-solid fa-xmark"></i></button>
      </div>
      <p class="text-xs text-slate-400">
        Enter the absolute path to a clinical cohort root (e.g. Stanford COCA, Harvard CardiacFatQuant, or institutional dataset folder).
      </p>
      <input type="text" id="inputCohortPath" placeholder="C:/path/to/cohort or E:/CT/..." 
             class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-sky-500">
      <div id="cohortSwitchError" class="text-xs text-rose-400 hidden"></div>
      <div class="flex items-center justify-end gap-3 pt-3">
        <button onclick="closeCohortModal()" class="px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs">Cancel</button>
        <button onclick="submitCohortPath()" class="px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-semibold text-xs">Load Cohort</button>
      </div>
    </div>
  </div>

  <!-- Modal: High-Res Interactive Chart Studio Pop-Out -->
  <div id="chartStudioModal" class="fixed inset-0 bg-black/80 backdrop-blur-md hidden z-50 flex items-center justify-center p-3 md:p-6">
    <div class="bg-slate-800 border border-slate-700 rounded-2xl w-full max-w-6xl max-h-[94vh] flex flex-col shadow-2xl overflow-hidden">
      <!-- Studio Header -->
      <div class="p-4 border-b border-slate-700 flex flex-wrap items-center justify-between gap-3 bg-slate-800/90">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-xl bg-sky-500/20 text-sky-400 flex items-center justify-center text-sm font-bold">
            <i class="fa-solid fa-chart-line" id="studioModalIcon"></i>
          </div>
          <div>
            <h2 class="text-base font-bold text-white flex items-center gap-2">
              <span id="studioModalTitle">Interactive Correlation Studio</span>
              <span id="studioCaseCountPill" class="text-[11px] px-2 py-0.5 rounded-full bg-slate-700 text-slate-300 font-mono">--</span>
            </h2>
            <p class="text-xs text-slate-400" id="studioModalSubtitle">Multi-dimensional cardiac adipose phenotyping & Agatston risk</p>
          </div>
        </div>

        <!-- Toolbar Controls -->
        <div class="flex items-center flex-wrap gap-2 text-xs">
          <!-- Scale Toggle -->
          <div class="inline-flex rounded-lg bg-slate-900 p-0.5 border border-slate-700">
            <button id="btnStudioScaleLin" onclick="setStudioScale('linear')" class="px-2.5 py-1 rounded text-xs font-medium bg-sky-600 text-white transition">Linear</button>
            <button id="btnStudioScaleLog" onclick="setStudioScale('logarithmic')" class="px-2.5 py-1 rounded text-xs font-medium text-slate-400 hover:text-white transition">Log₁₀ CAC</button>
          </div>

          <!-- Axis Swap -->
          <button onclick="toggleStudioAxisSwap()" class="px-2.5 py-1 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 transition flex items-center gap-1" title="Swap X and Y Axes">
            <i class="fa-solid fa-right-left text-[10px]"></i> Swap
          </button>

          <!-- Color Mode -->
          <div class="flex items-center gap-1.5 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-300">
            <span class="text-slate-400 text-[11px]">Color:</span>
            <select id="studioColorModeSelect" onchange="setStudioColorMode(this.value)" class="bg-transparent text-xs text-white focus:outline-none cursor-pointer">
              <option value="tier">Clinical CAC Tier</option>
              <option value="inflammation">CRISP-CT Status</option>
              <option value="single">Single Accent</option>
            </select>
          </div>

          <!-- Zoom & Reset -->
          <button onclick="zoomStudio(1.2)" class="px-2.5 py-1 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 transition" title="Zoom In"><i class="fa-solid fa-magnifying-glass-plus"></i></button>
          <button onclick="zoomStudio(0.8)" class="px-2.5 py-1 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 transition" title="Zoom Out"><i class="fa-solid fa-magnifying-glass-minus"></i></button>
          <button onclick="resetStudioZoom()" class="px-2.5 py-1 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 transition flex items-center gap-1" title="Reset Pan & Zoom"><i class="fa-solid fa-arrows-rotate text-[10px]"></i> Reset</button>

          <!-- Export PNG -->
          <button onclick="downloadStudioImage()" class="px-3 py-1 rounded-lg bg-emerald-600/80 hover:bg-emerald-600 text-white font-medium transition flex items-center gap-1.5" title="Export Publication-Quality PNG"><i class="fa-solid fa-download"></i> PNG</button>

          <!-- Close -->
          <button onclick="closeChartModal()" class="text-slate-400 hover:text-white text-lg ml-2"><i class="fa-solid fa-xmark"></i></button>
        </div>
      </div>

      <!-- Studio Chart Body -->
      <div class="p-5 flex-1 min-h-[460px] relative bg-slate-900/60">
        <canvas id="chartStudioCanvas"></canvas>
      </div>

      <!-- Studio Footer: Dynamic Legend & User Guidance -->
      <div class="p-3 border-t border-slate-700 bg-slate-800/90 flex flex-wrap items-center justify-between text-xs gap-3">
        <div class="flex items-center flex-wrap gap-4 text-slate-400" id="studioLegendContainer"></div>
        <div class="text-[11px] text-slate-400 italic">
          💡 Use mousewheel or trackpad to zoom freely on both axes; click and drag to pan through dense clusters.
        </div>
      </div>
    </div>
  </div>

  <!-- JavaScript Logic & Chart.js Visualizations -->
  <script>
    let activeTab = 'analytics';
    let cohortSummary = null;
    let rawCases = [];
    let chartEatCac = null;
    let chartPvatCac = null;
    let chartComps = null;
    let chartTiers = null;

    function switchTab(tab) {
      activeTab = tab;
      ['analytics', 'cases', 'benchmark'].forEach(t => {
        document.getElementById(`section-${t}`).classList.add('hidden');
        document.getElementById(`tab-${t}`).classList.remove('tab-active');
        document.getElementById(`tab-${t}`).classList.add('text-slate-400');
      });
      document.getElementById(`section-${tab}`).classList.remove('hidden');
      document.getElementById(`tab-${tab}`).classList.add('tab-active');
      document.getElementById(`tab-${tab}`).classList.remove('text-slate-400');

      if (tab === 'cases' && rawCases.length === 0) {
        fetchCases();
      }
    }

    async function refreshData() {
      await fetchSummary();
      await fetchScatterData();
      if (activeTab === 'cases') await fetchCases();
    }

    async function fetchSummary() {
      try {
        const res = await fetch('/api/cohort/summary');
        if (!res.ok) throw new Error("Failed to load cohort summary");
        cohortSummary = await res.json();

        document.getElementById('activeCohortPathDisplay').innerText = cohortSummary.root_dir || "Loaded";
        document.getElementById('casesCountPill').innerText = cohortSummary.total_count || 0;
        document.getElementById('kpiTotal').innerText = cohortSummary.total_count || 0;
        const qCount = cohortSummary.quantified_count || cohortSummary.segmented_count || 0;
        const qPct = cohortSummary.quantified_pct || cohortSummary.segmented_pct || 0;
        const maskCount = cohortSummary.mask_saved_count !== undefined ? cohortSummary.mask_saved_count : cohortSummary.segmented_count;
        const maskPct = cohortSummary.mask_saved_pct !== undefined ? cohortSummary.mask_saved_pct : cohortSummary.segmented_pct;
        document.getElementById('kpiQuantified').innerText = qCount;
        document.getElementById('kpiQuantifiedPct').innerText = `${qPct}% metrics`;
        document.getElementById('kpiMasks').innerText = maskCount;
        document.getElementById('kpiMasksPct').innerText = `${maskPct}% on disk`;
        document.getElementById('kpiCac').innerText = cohortSummary.cac_scored_count || 0;
        document.getElementById('kpiCacPct').innerText = `${cohortSummary.cac_scored_pct || 0}% of cohort`;
        // Row 2 Biomarkers: EAT Vol, EAT HU, PVAT Vol, PVAT HU, Inflammation
        const ev = cohortSummary.eat_volume_stats;
        if (ev && ev.mean !== undefined) {
          document.getElementById('kpiEatVol').innerText = `${ev.mean.toFixed(1)}`;
          document.getElementById('kpiEatVolSub').innerText = `N=${ev.count} • Med ${ev.median.toFixed(1)} (IQR ${ev.iqr}) • Skew +${ev.skewness}`;
        } else {
          document.getElementById('kpiEatVol').innerText = '--';
          document.getElementById('kpiEatVolSub').innerText = 'No data';
        }

        const eh = cohortSummary.eat_attenuation_stats;
        if (eh && eh.mean !== undefined) {
          document.getElementById('kpiEatHu').innerText = `${eh.mean.toFixed(1)}`;
          document.getElementById('kpiEatHuSub').innerText = `N=${eh.count} • SD ±${eh.std.toFixed(1)} • Med ${eh.median.toFixed(1)}`;
        } else {
          document.getElementById('kpiEatHu').innerText = '--';
          document.getElementById('kpiEatHuSub').innerText = 'No data';
        }

        const pv = cohortSummary.pvat_volume_stats;
        if (pv && pv.mean !== undefined) {
          document.getElementById('kpiPvatVol').innerText = `${pv.mean.toFixed(1)}`;
          document.getElementById('kpiPvatVolSub').innerText = `N=${pv.count} • Med ${pv.median.toFixed(1)} (IQR ${pv.iqr}) • Skew +${pv.skewness}`;
        } else {
          document.getElementById('kpiPvatVol').innerText = '--';
          document.getElementById('kpiPvatVolSub').innerText = 'No data';
        }

        const ph = cohortSummary.pvat_attenuation_stats;
        if (ph && ph.mean !== undefined) {
          document.getElementById('kpiPvatHu').innerText = `${ph.mean.toFixed(1)}`;
          document.getElementById('kpiPvatHuSub').innerText = `N=${ph.count} • SD ±${ph.std.toFixed(1)} • Med ${ph.median.toFixed(1)}`;
        } else {
          document.getElementById('kpiPvatHu').innerText = '--';
          document.getElementById('kpiPvatHuSub').innerText = 'No data';
        }

        document.getElementById('kpiInflam').innerText = cohortSummary.active_inflammation_count || 0;
        document.getElementById('kpiInflamPct').innerText = `${cohortSummary.active_inflammation_pct || 0}% CRISP-CT > -70.1 HU`;
        const explCountEl = document.getElementById('explActiveCount');
        if (explCountEl) explCountEl.innerText = `${cohortSummary.active_inflammation_count || 0}`;
        const explPctEl = document.getElementById('explActivePct');
        if (explPctEl) explPctEl.innerText = `${cohortSummary.active_inflammation_pct || 0}%`;

        // Correlation labels
        if (cohortSummary.correlations?.eat_vs_cac) {
          const c = cohortSummary.correlations.eat_vs_cac;
          document.getElementById('corrEatCac').innerText = `r=${c.pearson_r} (p=${c.pearson_p < 0.001 ? '<0.001' : c.pearson_p.toFixed(3)})`;
        } else {
          document.getElementById('corrEatCac').innerText = 'N/A';
        }

        if (cohortSummary.correlations?.pvat_hu_vs_cac) {
          const c = cohortSummary.correlations.pvat_hu_vs_cac;
          document.getElementById('corrPvatCac').innerText = `r=${c.pearson_r} (p=${c.pearson_p < 0.001 ? '<0.001' : c.pearson_p.toFixed(3)})`;
        } else {
          document.getElementById('corrPvatCac').innerText = 'N/A';
        }

        renderRiskTiersChart(cohortSummary.risk_tier_counts || {});
        renderCompartmentChart(cohortSummary.chamber_volume_summaries || {});
      } catch (err) {
        console.error("fetchSummary error:", err);
      }
    }

    let rawScatterPoints = [];
    let scaleModes = { eat: 'linear', pvat: 'linear' };
    let axisSwap = { eat: false, pvat: false };
    let studioChart = null;
    let currentStudioMetric = 'eat';
    let studioScale = 'linear';
    let studioColorMode = 'tier';
    let studioAxisSwap = false;

    const TIER_COLORS = {
      'Zero (0)': '#94a3b8',
      'Minimal (1-10)': '#34d399',
      'Mild (11-100)': '#38bdf8',
      'Moderate (101-400)': '#fbbf24',
      'Severe (>400)': '#f43f5e',
      'Unknown': '#64748b'
    };

    function getPointColor(p, mode, defaultColor) {
      if (mode === 'tier') return TIER_COLORS[p.calcium_risk_tier] || '#64748b';
      if (mode === 'inflammation') {
        return p.crisp_ct_status === 'Active Vascular Inflammation' ? '#f43f5e' : '#38bdf8';
      }
      return defaultColor;
    }

    async function fetchScatterData() {
      try {
        const res = await fetch('/api/cohort/scatter');
        if (!res.ok) return;
        rawScatterPoints = await res.json();
        renderScatterCharts(rawScatterPoints);
      } catch (err) {
        console.error("fetchScatterData error:", err);
      }
    }

    function toggleAxisSwap(metric) {
      axisSwap[metric] = !axisSwap[metric];
      const isEat = (metric === 'eat');
      const swapped = axisSwap[metric];
      
      const titleEl = document.getElementById(isEat ? 'titleChartEat' : 'titleChartPvat');
      const subEl = document.getElementById(isEat ? 'subChartEat' : 'subChartPvat');
      
      if (isEat) {
        titleEl.innerHTML = swapped 
          ? '<i class="fa-solid fa-chart-line text-sky-400"></i> Total EAT Volume vs. Agatston CAC'
          : '<i class="fa-solid fa-chart-line text-sky-400"></i> Agatston CAC vs. Total EAT Volume';
        subEl.innerText = swapped
          ? 'Epicardial adipose volume as a function of coronary calcification'
          : 'Coronary calcification as a function of epicardial fat volume';
      } else {
        titleEl.innerHTML = swapped
          ? '<i class="fa-solid fa-fire-flame-curved text-rose-400"></i> PVAT Attenuation vs. Agatston CAC'
          : '<i class="fa-solid fa-fire-flame-curved text-rose-400"></i> Agatston CAC vs. PVAT Attenuation';
        subEl.innerText = swapped
          ? 'Pericoronary vascular inflammation (HU) vs. calcification'
          : 'Coronary calcification stratified by pericoronary inflammation';
      }

      renderScatterCharts(rawScatterPoints);
    }

    function setScaleMode(metric, mode) {
      scaleModes[metric] = mode;
      const isEat = (metric === 'eat');
      const linBtn = document.getElementById(isEat ? 'btnScaleEatLin' : 'btnScalePvatLin');
      const logBtn = document.getElementById(isEat ? 'btnScaleEatLog' : 'btnScalePvatLog');
      const activeBg = isEat ? 'bg-sky-600' : 'bg-rose-600';

      if (mode === 'linear') {
        linBtn.className = `px-2 py-0.5 rounded text-[11px] font-medium ${activeBg} text-white transition`;
        logBtn.className = 'px-2 py-0.5 rounded text-[11px] font-medium text-slate-400 hover:text-white transition';
      } else {
        linBtn.className = 'px-2 py-0.5 rounded text-[11px] font-medium text-slate-400 hover:text-white transition';
        logBtn.className = `px-2 py-0.5 rounded text-[11px] font-medium ${activeBg} text-white transition`;
      }

      renderScatterCharts(rawScatterPoints);
    }

    function resetChartZoom(metric) {
      if (metric === 'eat' && chartEatCac) chartEatCac.resetZoom();
      if (metric === 'pvat' && chartPvatCac) chartPvatCac.resetZoom();
    }

    function buildScatterConfig(metric, pts, scaleMode, colorMode, inModal = false) {
      const isEat = (metric === 'eat');
      const isLog = (scaleMode === 'logarithmic');
      const isSwapped = (inModal ? studioAxisSwap : axisSwap[metric]);

      const filtered = pts.filter(p => {
        const fatVal = isEat ? p.eat_volume_cm3 : p.pvat_mean_hu;
        return fatVal !== null && p.calcium_score !== null;
      });

      const chartPoints = filtered.map(p => {
        const rawCac = p.calcium_score;
        const cacVal = isLog ? Math.max(1, rawCac) : rawCac;
        const fatVal = isEat ? p.eat_volume_cm3 : p.pvat_mean_hu;

        // Default (!isSwapped): X = Fat (EAT/PVAT), Y = CAC
        // Swapped (isSwapped):  X = CAC, Y = Fat (EAT/PVAT)
        const xCoord = isSwapped ? cacVal : fatVal;
        const yCoord = isSwapped ? fatVal : cacVal;

        return {
          x: xCoord,
          y: yCoord,
          originalCAC: rawCac,
          fatValue: fatVal,
          label: p.participant_id,
          tier: p.calcium_risk_tier,
          crisp: p.crisp_ct_status,
        };
      });

      const defaultColor = isEat ? '#38bdf8' : '#fb7185';
      const colors = filtered.map(p => getPointColor(p, colorMode, defaultColor));

      const datasets = [{
        label: isEat ? 'Participant EAT' : 'Participant PVAT',
        data: chartPoints,
        backgroundColor: colors,
        borderColor: colors,
        pointRadius: inModal ? 5 : 4,
        pointHoverRadius: inModal ? 8 : 6,
      }];

      if (!isEat && chartPoints.length > 0) {
        if (!isSwapped) {
          // Vertical cutoff line at X = -70.1 HU
          const minY = Math.min(...chartPoints.map(p => p.y));
          const maxY = Math.max(...chartPoints.map(p => p.y));
          datasets.push({
            type: 'line',
            label: 'CRISP-CT Cutoff (-70.1 HU)',
            data: [{ x: -70.1, y: minY }, { x: -70.1, y: maxY }],
            borderColor: '#f43f5e',
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 0,
            fill: false,
          });
        } else {
          // Horizontal cutoff line at Y = -70.1 HU
          const minX = Math.min(...chartPoints.map(p => p.x));
          const maxX = Math.max(...chartPoints.map(p => p.x));
          datasets.push({
            type: 'line',
            label: 'CRISP-CT Cutoff (-70.1 HU)',
            data: [{ x: minX, y: -70.1 }, { x: maxX, y: -70.1 }],
            borderColor: '#f43f5e',
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 0,
            fill: false,
          });
        }
      }

      const cacScaleConfig = {
        type: isLog ? 'logarithmic' : 'linear',
        min: isLog ? 1 : undefined,
        title: {
          display: true,
          text: isLog ? 'Agatston Calcium Score (Log₁₀)' : 'Agatston Calcium Score (Linear)',
          color: '#94a3b8'
        },
        grid: { color: '#334155' },
        ticks: {
          color: '#94a3b8',
          callback: function(val) {
            const num = Number(val);
            if (isLog) {
              if ([1, 10, 100, 1000, 10000, 100000, 1000000, 4000000].includes(num)) {
                if (num >= 1000000) return (num / 1000000) + 'M';
                if (num >= 1000) return (num / 1000) + 'k';
                return num;
              }
              return '';
            } else {
              if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
              if (num >= 1000) return (num / 1000).toFixed(0) + 'k';
              return num;
            }
          }
        }
      };

      const fatScaleConfig = {
        type: 'linear',
        title: {
          display: true,
          text: isEat ? 'Total EAT Volume (cm³)' : 'PVAT Mean Attenuation (HU)',
          color: '#94a3b8'
        },
        grid: { color: '#334155' },
        ticks: { color: '#94a3b8' }
      };

      const xScale = isSwapped ? cacScaleConfig : fatScaleConfig;
      const yScale = isSwapped ? fatScaleConfig : cacScaleConfig;

      return {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: { x: xScale, y: yScale },
          plugins: {
            legend: {
              display: !isEat,
              labels: {
                color: '#94a3b8',
                filter: (item) => item.text && item.text.includes('Cutoff')
              }
            },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const r = ctx.raw;
                  if (!r || r.originalCAC === undefined) return ctx.dataset.label;
                  const cacStr = r.originalCAC;
                  const fatStr = Number(r.fatValue).toFixed(1);
                  const unit = isEat ? 'cm³' : 'HU';
                  return `Patient ${r.label} [${r.tier}]: ${isEat ? 'EAT' : 'PVAT'}=${fatStr} ${unit} ➔ CAC=${cacStr} (${r.crisp})`;
                }
              }
            },
            zoom: {
              pan: { enabled: true, mode: 'xy' },
              zoom: {
                wheel: { enabled: true, speed: 0.08 },
                pinch: { enabled: true },
                mode: 'xy'
              }
            }
          }
        }
      };
    }

    function renderScatterCharts(pts) {
      if (chartEatCac) chartEatCac.destroy();
      const ctxEat = document.getElementById('chartEatVsCac').getContext('2d');
      chartEatCac = new Chart(ctxEat, buildScatterConfig('eat', pts, scaleModes.eat, 'single', false));

      if (chartPvatCac) chartPvatCac.destroy();
      const ctxPvat = document.getElementById('chartPvatVsCac').getContext('2d');
      chartPvatCac = new Chart(ctxPvat, buildScatterConfig('pvat', pts, scaleModes.pvat, 'single', false));
    }

    // ================= POP-OUT STUDIO MODAL LOGIC =================
    function openChartModal(metric) {
      currentStudioMetric = metric;
      studioScale = scaleModes[metric] || 'linear';
      studioAxisSwap = axisSwap[metric] || false;
      document.getElementById('chartStudioModal').classList.remove('hidden');

      const isEat = (metric === 'eat');
      updateStudioTitles();
      document.getElementById('studioModalIcon').className = isEat
        ? 'fa-solid fa-chart-line text-sky-400'
        : 'fa-solid fa-fire-flame-curved text-rose-400';

      const validN = rawScatterPoints.filter(p => (isEat ? p.eat_volume_cm3 : p.pvat_mean_hu) !== null && p.calcium_score !== null).length;
      document.getElementById('studioCaseCountPill').innerText = `${validN} paired cases`;

      updateStudioScaleButtons();
      renderStudioChart();
      renderStudioLegend();
    }

    function updateStudioTitles() {
      const isEat = (currentStudioMetric === 'eat');
      document.getElementById('studioModalTitle').innerText = isEat
        ? (studioAxisSwap ? 'Total EAT Volume vs. Agatston CAC' : 'Agatston CAC vs. Total EAT Volume')
        : (studioAxisSwap ? 'PVAT Attenuation vs. Agatston CAC' : 'Agatston CAC vs. PVAT Attenuation');
      document.getElementById('studioModalSubtitle').innerText = isEat
        ? 'Epicardial adipose volume burden vs. patient coronary calcification scores'
        : 'Pericoronary vascular inflammation cutoff (-70.1 HU) vs. Agatston calcification';
    }

    function toggleStudioAxisSwap() {
      studioAxisSwap = !studioAxisSwap;
      updateStudioTitles();
      renderStudioChart();
    }

    function closeChartModal() {
      document.getElementById('chartStudioModal').classList.add('hidden');
      if (studioChart) {
        studioChart.destroy();
        studioChart = null;
      }
    }

    function setStudioScale(mode) {
      studioScale = mode;
      updateStudioScaleButtons();
      renderStudioChart();
    }

    function updateStudioScaleButtons() {
      const isEat = currentStudioMetric === 'eat';
      const activeBg = isEat ? 'bg-sky-600' : 'bg-rose-600';
      const linBtn = document.getElementById('btnStudioScaleLin');
      const logBtn = document.getElementById('btnStudioScaleLog');
      if (studioScale === 'linear') {
        linBtn.className = `px-2.5 py-1 rounded text-xs font-medium ${activeBg} text-white transition`;
        logBtn.className = 'px-2.5 py-1 rounded text-xs font-medium text-slate-400 hover:text-white transition';
      } else {
        linBtn.className = 'px-2.5 py-1 rounded text-xs font-medium text-slate-400 hover:text-white transition';
        logBtn.className = `px-2.5 py-1 rounded text-xs font-medium ${activeBg} text-white transition`;
      }
    }

    function setStudioColorMode(mode) {
      studioColorMode = mode;
      renderStudioChart();
      renderStudioLegend();
    }

    function zoomStudio(factor) {
      if (studioChart) studioChart.zoom(factor);
    }

    function resetStudioZoom() {
      if (studioChart) studioChart.resetZoom();
    }

    function downloadStudioImage() {
      if (!studioChart) return;
      const url = studioChart.toBase64Image('image/png', 1.0);
      const a = document.createElement('a');
      a.download = `cardiac_fat_${currentStudioMetric}_vs_cac_${studioScale}.png`;
      a.href = url;
      a.click();
    }

    function renderStudioChart() {
      if (studioChart) studioChart.destroy();
      const ctx = document.getElementById('chartStudioCanvas').getContext('2d');
      studioChart = new Chart(ctx, buildScatterConfig(currentStudioMetric, rawScatterPoints, studioScale, studioColorMode, true));
    }

    function renderStudioLegend() {
      const container = document.getElementById('studioLegendContainer');
      if (studioColorMode === 'tier') {
        container.innerHTML = `
          <span class="font-semibold text-slate-300">Plaque Burden:</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#94a3b8]"></span> Zero (0)</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#34d399]"></span> Minimal (1-10)</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#38bdf8]"></span> Mild (11-100)</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#fbbf24]"></span> Moderate (101-400)</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#f43f5e]"></span> Severe (>400)</span>
        `;
      } else if (studioColorMode === 'inflammation') {
        container.innerHTML = `
          <span class="font-semibold text-slate-300">CRISP-CT Cutoff:</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#38bdf8]"></span> Normal Fat (&le; -70.1 HU)</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#f43f5e]"></span> Active Inflammation (> -70.1 HU)</span>
        `;
      } else {
        container.innerHTML = `<span class="text-slate-400">Single accent color mode</span>`;
      }
    }

    function renderCompartmentChart(summaries) {
      const labels = ['EAT LA', 'EAT LV', 'EAT RA', 'EAT RV', 'PVAT LAD', 'PVAT LCX', 'PVAT RCA'];
      const means = labels.map(l => summaries[l]?.mean || 0);

      if (chartComps) chartComps.destroy();
      const ctx = document.getElementById('chartCompartments').getContext('2d');
      chartComps = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Mean Volume (cm³)',
            data: means,
            backgroundColor: ['#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8', '#f43f5e', '#ec4899', '#d946ef'],
            borderRadius: 6,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { title: { display: true, text: 'Volume (cm³)', color: '#94a3b8' }, grid: { color: '#334155' } },
            x: { grid: { display: false } }
          },
          plugins: { legend: { display: false } }
        }
      });
    }

    function renderRiskTiersChart(tierCounts) {
      const keys = Object.keys(tierCounts);
      const vals = keys.map(k => tierCounts[k]);

      if (chartTiers) chartTiers.destroy();
      const ctx = document.getElementById('chartRiskTiers').getContext('2d');
      chartTiers = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: keys,
          datasets: [{
            data: vals,
            backgroundColor: ['#10b981', '#06b6d4', '#f59e0b', '#f97316', '#ef4444', '#64748b'],
            borderWidth: 2,
            borderColor: '#1e293b'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right', labels: { color: '#cbd5e1', font: { size: 11 } } }
          }
        }
      });
    }

    async function fetchCases() {
      try {
        const res = await fetch('/api/cohort/cases');
        if (!res.ok) return;
        rawCases = await res.json();
        document.getElementById('casesCountPill').innerText = rawCases.length;
        renderCasesTable(rawCases);
      } catch (err) {
        console.error("fetchCases error:", err);
      }
    }

    function applyCaseFilters() {
      const q = document.getElementById('caseSearchInput').value.toLowerCase().trim();
      const tier = document.getElementById('caseTierFilter').value;
      const segOnly = document.getElementById('caseSegOnly').checked;
      const inflamOnly = document.getElementById('caseInflamOnly').checked;

      const filtered = rawCases.filter(c => {
        if (q && !c.participant_id.toLowerCase().includes(q)) return false;
        if (tier && c.calcium_risk_tier !== tier) return false;
        if (segOnly && !c.has_seg) return false;
        if (inflamOnly && c.crisp_ct_status !== 'Active Vascular Inflammation') return false;
        return true;
      });

      renderCasesTable(filtered);
    }

    function renderCasesTable(cases) {
      const tbody = document.getElementById('casesTableBody');
      if (cases.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="text-center py-8 text-slate-400">No matching cases found</td></tr>';
        return;
      }

      tbody.innerHTML = cases.map(c => {
        const segBadge = c.has_seg 
          ? '<span class="px-2 py-0.5 rounded bg-emerald-900/60 text-emerald-400 border border-emerald-500/40 text-[10px]">Segmented</span>'
          : '<span class="px-2 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px]">Unsegmented</span>';

        const inflamBadge = c.crisp_ct_status === 'Active Vascular Inflammation'
          ? '<span class="px-2 py-0.5 rounded bg-rose-900/60 text-rose-400 border border-rose-500/40 text-[10px]"><i class="fa-solid fa-fire text-[9px] mr-1"></i>Active</span>'
          : `<span class="text-slate-400 text-[11px]">${c.crisp_ct_status}</span>`;

        return `
          <tr class="hover:bg-slate-700/40 transition">
            <td class="px-4 py-2.5 font-mono font-medium text-white">${c.participant_id}</td>
            <td class="px-4 py-2.5 text-slate-400">${c.scan_type}</td>
            <td class="px-4 py-2.5 font-mono text-amber-300">${c.calcium_score !== null ? c.calcium_score : '--'} <span class="text-[10px] text-slate-400 ml-1">(${c.calcium_risk_tier})</span></td>
            <td class="px-4 py-2.5 font-mono text-sky-300">${c.eat_volume_cm3 !== null && c.eat_volume_cm3 !== undefined ? c.eat_volume_cm3.toFixed(2) + ' cm³' : '--'}</td>
            <td class="px-4 py-2.5 font-mono text-slate-300">${c.pvat_volume_cm3 !== null && c.pvat_volume_cm3 !== undefined ? c.pvat_volume_cm3.toFixed(2) + ' cm³' : '--'}</td>
            <td class="px-4 py-2.5 font-mono text-indigo-300">${c.pvat_mean_hu !== null && c.pvat_mean_hu !== undefined ? c.pvat_mean_hu.toFixed(1) + ' HU' : '--'}</td>
            <td class="px-4 py-2.5">${inflamBadge}</td>
            <td class="px-4 py-2.5 flex items-center gap-1.5">${segBadge}</td>
            <td class="px-4 py-2.5 text-right space-x-2">
              <button onclick="openCaseModal('${c.participant_id}')" class="text-sky-400 hover:text-sky-300 text-xs font-medium">Deep Dive</button>
            </td>
          </tr>
        `;
      }).join('');
    }

    async function openCaseModal(pid) {
      try {
        const res = await fetch(`/api/case/${pid}`);
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById('modalCaseId').innerText = data.participant_id;
        document.getElementById('modalScanType').innerText = data.scan_type;
        document.getElementById('modalCacScore').innerText = data.calcium_score !== null ? `${data.calcium_score} (${data.calcium_risk_tier})` : 'Not available';
        document.getElementById('modalCrispStatus').innerText = data.crisp_ct_status;

        const tbody = document.getElementById('modalMetricsBody');
        const metrics = data.metrics || {};
        const regions = ["EAT Total", "EAT LA", "EAT LV", "EAT RA", "EAT RV", "PVAT LAD", "PVAT LCX", "PVAT RCA", "PVAT Total"];

        tbody.innerHTML = regions.map(reg => {
          const vol = metrics[`${reg} Volume (cm3)`];
          const hu = metrics[`${reg} Mean Attenuation (HU)`];
          return `
            <tr>
              <td class="p-2 font-medium text-white">${reg}</td>
              <td class="p-2 text-right font-mono text-sky-300">${vol !== undefined && !isNaN(vol) ? vol.toFixed(2) : '--'}</td>
              <td class="p-2 text-right font-mono text-indigo-300">${hu !== undefined && !isNaN(hu) ? hu.toFixed(1) : '--'}</td>
            </tr>
          `;
        }).join('');

        document.getElementById('caseModal').classList.remove('hidden');
      } catch (err) {
        console.error("openCaseModal error:", err);
      }
    }

    function closeCaseModal() {
      document.getElementById('caseModal').classList.add('hidden');
    }

    async function runBenchmarkEvaluation() {
      const cand = document.getElementById('bmCandidateName').value.trim();
      const ref = document.getElementById('bmReferenceName').value.trim();
      const limit = document.getElementById('bmCaseLimit').value;
      const btn = document.getElementById('btnRunBenchmark');

      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Evaluating...';

      try {
        const res = await fetch('/api/benchmark/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ candidate_seg: cand, reference_seg: ref, limit: limit })
        });
        const data = await res.json();
        if (!res.ok) {
          alert(data.error || "Benchmark evaluation failed");
          return;
        }

        renderBenchmarkResults(data);
      } catch (err) {
        alert("Error executing benchmark: " + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Benchmark';
      }
    }

    function renderBenchmarkResults(rep) {
      const c = document.getElementById('benchmarkResultsContainer');
      const comps = rep.compartment_summary || {};

      const compRows = Object.keys(comps).map(k => {
        const s = comps[k];
        return `
          <tr class="hover:bg-slate-700/40">
            <td class="px-4 py-2 font-medium text-white">${k}</td>
            <td class="px-4 py-2 text-right font-mono text-emerald-400 font-bold">${(s.mean_dice * 100).toFixed(1)}%</td>
            <td class="px-4 py-2 text-right font-mono text-slate-300">${(s.median_dice * 100).toFixed(1)}%</td>
            <td class="px-4 py-2 text-right font-mono text-slate-400">${(s.iqr_dice * 100).toFixed(1)}%</td>
            <td class="px-4 py-2 text-right font-mono text-sky-300">${(s.mean_iou * 100).toFixed(1)}%</td>
            <td class="px-4 py-2 text-right font-mono text-amber-300">${s.mean_abs_volume_diff_cm3} cm³</td>
            <td class="px-4 py-2 text-right font-mono text-indigo-300">${s.mean_relative_volume_error_pct}%</td>
          </tr>
        `;
      }).join('');

      c.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow">
            <div class="text-xs text-slate-400 uppercase font-semibold">Cases Evaluated</div>
            <div class="text-2xl font-bold text-white mt-1">${rep.participant_count}</div>
            <div class="text-xs text-slate-400 mt-1">${rep.candidate_name} vs. ${rep.reference_name}</div>
          </div>
          <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow">
            <div class="text-xs text-slate-400 uppercase font-semibold">Mean Overall Dice</div>
            <div class="text-2xl font-bold text-emerald-400 mt-1">${(rep.mean_overall_dice * 100).toFixed(2)}%</div>
            <div class="text-xs text-emerald-500/80 mt-1">Cohort Multi-Compartment Agreement</div>
          </div>
          <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow">
            <div class="text-xs text-slate-400 uppercase font-semibold">Median Overall Dice</div>
            <div class="text-2xl font-bold text-sky-400 mt-1">${(rep.median_overall_dice * 100).toFixed(2)}%</div>
            <div class="text-xs text-slate-400 mt-1">Robust Non-Parametric Central Tendency</div>
          </div>
        </div>

        <div class="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow">
          <div class="p-4 bg-slate-900/60 border-b border-slate-700 flex items-center justify-between">
            <h4 class="text-xs font-bold text-white uppercase tracking-wider">Multi-Compartment Segmentation Scorecard</h4>
            <div class="flex items-center gap-2">
              <a href="/api/benchmark/export?format=csv" class="bg-slate-700 hover:bg-slate-600 text-xs text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition">
                <i class="fa-solid fa-file-csv text-emerald-400"></i> Export CSV
              </a>
              <a href="/api/benchmark/export?format=json" class="bg-slate-700 hover:bg-slate-600 text-xs text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition">
                <i class="fa-solid fa-file-code text-sky-400"></i> Export JSON
              </a>
            </div>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs text-slate-300">
              <thead class="bg-slate-900/40 text-slate-400 uppercase tracking-wider">
                <tr>
                  <th class="px-4 py-2.5">Compartment</th>
                  <th class="px-4 py-2.5 text-right">Mean Dice</th>
                  <th class="px-4 py-2.5 text-right">Median Dice</th>
                  <th class="px-4 py-2.5 text-right">Dice IQR</th>
                  <th class="px-4 py-2.5 text-right">Mean IoU</th>
                  <th class="px-4 py-2.5 text-right">Mean Abs Vol Diff</th>
                  <th class="px-4 py-2.5 text-right">Mean Rel Vol Err</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-700/60">
                ${compRows}
              </tbody>
            </table>
          </div>
        </div>
      `;
    }

    function openCohortModal() {
      document.getElementById('cohortModal').classList.remove('hidden');
    }

    function closeCohortModal() {
      document.getElementById('cohortModal').classList.add('hidden');
      document.getElementById('cohortSwitchError').classList.add('hidden');
    }

    async function submitCohortPath() {
      const p = document.getElementById('inputCohortPath').value.trim();
      const errEl = document.getElementById('cohortSwitchError');
      errEl.classList.add('hidden');

      if (!p) return;

      try {
        const res = await fetch('/api/cohort/set_directory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cohort_dir: p })
        });
        const data = await res.json();
        if (!res.ok) {
          errEl.innerText = data.error || "Failed to switch cohort";
          errEl.classList.remove('hidden');
          return;
        }

        closeCohortModal();
        await refreshData();
      } catch (err) {
        errEl.innerText = "Error: " + err;
        errEl.classList.remove('hidden');
      }
    }

    // Initial load
    window.addEventListener('DOMContentLoaded', () => {
      refreshData();
    });
  </script>
</body>
</html>
"""

