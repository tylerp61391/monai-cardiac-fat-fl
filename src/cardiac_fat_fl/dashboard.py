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
        return jsonify(summary)

    @app.route("/api/cohort/scatter", methods=["GET"])
    def get_scatter_data():
        if _ACTIVE_COHORT is None:
            return jsonify([]), 400
        return jsonify(_ACTIVE_COHORT.get_scatter_data())

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

        return jsonify(cases)

    @app.route("/api/case/<pid>", methods=["GET"])
    def get_case_details(pid: str):
        if _ACTIVE_COHORT is None or pid not in _ACTIVE_COHORT.participants:
            return jsonify({"error": f"Participant {pid} not found"}), 404
        p = _ACTIVE_COHORT.participants[pid]
        return jsonify(p.to_dict())

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
      
      <!-- KPI Scorecards -->
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Total Cohort</div>
          <div class="text-2xl font-bold text-white mt-1" id="kpiTotal">--</div>
          <div class="text-[11px] text-slate-400 mt-1">Discovered Cases</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Quantified</div>
          <div class="text-2xl font-bold text-emerald-400 mt-1" id="kpiQuantified">--</div>
          <div class="text-[11px] text-emerald-500/80 mt-1" id="kpiQuantifiedPct">--% analyzed</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Agatston CAC</div>
          <div class="text-2xl font-bold text-amber-400 mt-1" id="kpiCac">--</div>
          <div class="text-[11px] text-amber-500/80 mt-1" id="kpiCacPct">--% scored</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Median EAT</div>
          <div class="text-2xl font-bold text-sky-400 mt-1" id="kpiMedianEat">--</div>
          <div class="text-[11px] text-slate-400 mt-1">cm³ Total Volume</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Mean PVAT HU</div>
          <div class="text-2xl font-bold text-indigo-400 mt-1" id="kpiMeanPvatHu">--</div>
          <div class="text-[11px] text-slate-400 mt-1">Hounsfield Units</div>
        </div>
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-4 shadow">
          <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Inflammation</div>
          <div class="text-2xl font-bold text-rose-400 mt-1" id="kpiInflam">--</div>
          <div class="text-[11px] text-rose-500/80 mt-1">CRISP-CT > -70.1 HU</div>
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
        
        <!-- Chart 1: EAT Volume vs Calcium Score -->
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow flex flex-col">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h3 class="text-sm font-bold text-white flex items-center gap-2">
                <i class="fa-solid fa-chart-line text-sky-400"></i>
                Total EAT Volume vs. Agatston Calcium Score
              </h3>
              <p class="text-xs text-slate-400">Stratified by clinical plaque burden tiers</p>
            </div>
          </div>
          <div class="relative flex-1 min-h-[300px]">
            <canvas id="chartEatVsCac"></canvas>
          </div>
        </div>

        <!-- Chart 2: PVAT HU vs Calcium Score -->
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow flex flex-col">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h3 class="text-sm font-bold text-white flex items-center gap-2">
                <i class="fa-solid fa-fire-flame-curved text-rose-400"></i>
                PVAT Attenuation vs. Agatston CAC (CRISP-CT)
              </h3>
              <p class="text-xs text-slate-400">Red line denotes -70.1 HU vascular inflammation threshold</p>
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
        document.getElementById('kpiQuantified').innerText = qCount;
        document.getElementById('kpiQuantifiedPct').innerText = `${qPct}% (${maskCount} 3D masks)`;
        document.getElementById('kpiCac').innerText = cohortSummary.cac_scored_count || 0;
        document.getElementById('kpiCacPct').innerText = `${cohortSummary.cac_scored_pct || 0}% scored`;
        document.getElementById('kpiMedianEat').innerText = cohortSummary.eat_volume_stats?.median ? `${cohortSummary.eat_volume_stats.median}` : '--';
        document.getElementById('kpiMeanPvatHu').innerText = cohortSummary.pvat_attenuation_stats?.mean ? `${cohortSummary.pvat_attenuation_stats.mean}` : '--';
        document.getElementById('kpiInflam').innerText = cohortSummary.active_inflammation_count || 0;

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

    async function fetchScatterData() {
      try {
        const res = await fetch('/api/cohort/scatter');
        if (!res.ok) return;
        const pts = await res.json();
        renderScatterCharts(pts);
      } catch (err) {
        console.error("fetchScatterData error:", err);
      }
    }

    function renderScatterCharts(pts) {
      // 1. EAT Volume vs Calcium Score
      const eatPts = pts
        .filter(p => p.eat_volume_cm3 !== null && p.calcium_score !== null)
        .map(p => ({ x: p.calcium_score, y: p.eat_volume_cm3, label: p.participant_id }));

      if (chartEatCac) chartEatCac.destroy();
      const ctxEat = document.getElementById('chartEatVsCac').getContext('2d');
      chartEatCac = new Chart(ctxEat, {
        type: 'scatter',
        data: {
          datasets: [{
            label: 'Participant CT',
            data: eatPts,
            backgroundColor: '#38bdf8',
            borderColor: '#0284c7',
            pointRadius: 4,
            pointHoverRadius: 6,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: 'Agatston Calcium Score', color: '#94a3b8' }, grid: { color: '#334155' } },
            y: { title: { display: true, text: 'Total EAT Volume (cm³)', color: '#94a3b8' }, grid: { color: '#334155' } },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => `Patient ${ctx.raw.label}: CAC=${ctx.raw.x}, EAT=${ctx.raw.y} cm³`
              }
            }
          }
        }
      });

      // 2. PVAT HU vs Calcium Score
      const pvatPts = pts
        .filter(p => p.pvat_mean_hu !== null && p.calcium_score !== null)
        .map(p => ({ x: p.calcium_score, y: p.pvat_mean_hu, label: p.participant_id }));

      if (chartPvatCac) chartPvatCac.destroy();
      const ctxPvat = document.getElementById('chartPvatVsCac').getContext('2d');
      chartPvatCac = new Chart(ctxPvat, {
        type: 'scatter',
        data: {
          datasets: [{
            label: 'Participant PVAT',
            data: pvatPts,
            backgroundColor: '#fb7185',
            borderColor: '#e11d48',
            pointRadius: 4,
            pointHoverRadius: 6,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: 'Agatston Calcium Score', color: '#94a3b8' }, grid: { color: '#334155' } },
            y: { title: { display: true, text: 'PVAT Mean Attenuation (HU)', color: '#94a3b8' }, min: -140, max: -40, grid: { color: '#334155' } },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => `Patient ${ctx.raw.label}: CAC=${ctx.raw.x}, PVAT=${ctx.raw.y} HU`
              }
            }
          }
        }
      });
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

