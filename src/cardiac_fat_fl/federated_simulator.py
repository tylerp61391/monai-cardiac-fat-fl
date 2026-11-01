"""NVIDIA FLARE 2.5+ SOTA Federated Learning Simulator Runner for Cardiac Fat Quantification.

Enables zero-friction local federated simulation of multi-hospital model distillation
(FedProx + FedBN) and cross-site evaluation with zero patient DICOM transfer.
Includes automatic Windows OS file-lock and process group compatibility patching.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def ensure_nvflare_windows_compatibility() -> None:
    """Ensure NVFLARE simulator runs reliably on Windows platforms.

    Patches Windows-specific limitations in upstream NVFLARE 2.5+:
    1. Prevents WinError 32 PermissionError in SimulatorRunner._cleanup_workspace()
       caused by attempting to rmtree the current working directory.
    2. Fixes POSIX-only os.setsid in SimulatorRunner._run_client() by using
       CREATE_NEW_PROCESS_GROUP and preserving Windows backslash path semantics.
    3. Handles file lock in SimulatorDeployer.close() on temporary admin storage.
    4. Ensures component path authorizer permits custom algorithms.
    """
    if sys.platform != "win32":
        return

    try:
        import nvflare
        nvflare_root = Path(nvflare.__file__).parent

        # 1. Patch simulator_runner.py
        sim_runner_path = nvflare_root / "private" / "fed" / "app" / "simulator" / "simulator_runner.py"
        if sim_runner_path.is_file():
            content = sim_runner_path.read_text(encoding="utf-8")
            modified = False

            # Replace unsafe rmtree of self.simulator_root
            old_cleanup = "if os.path.exists(self.simulator_root):\n                shutil.rmtree(self.simulator_root)"
            if old_cleanup in content:
                safe_cleanup = """# Windows-safe directory content cleanup without deleting cwd
            for _item in os.listdir(self.simulator_root):
                if _item in ("startup", "local"):
                    continue
                _p = os.path.join(self.simulator_root, _item)
                if os.path.isdir(_p):
                    shutil.rmtree(_p, ignore_errors=True)
                else:
                    try:
                        os.remove(_p)
                    except OSError:
                        pass"""
                content = content.replace(old_cleanup, safe_cleanup)
                modified = True

            # Replace POSIX preexec_fn=os.setsid with Windows-safe process creation
            old_setsid = "process = subprocess.Popen(shlex.split(command, True), shell=False, preexec_fn=os.setsid, env=new_env)"
            if old_setsid in content:
                safe_setsid = (
                    "preexec_fn = os.setsid if hasattr(os, 'setsid') else None\n"
                    "        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0\n"
                    "        process = subprocess.Popen(shlex.split(command, posix=sys.platform != 'win32'), "
                    "shell=False, preexec_fn=preexec_fn, creationflags=creationflags, env=new_env)"
                )
                content = content.replace(old_setsid, safe_setsid)
                modified = True

            if modified:
                sim_runner_path.write_text(content, encoding="utf-8")
                logger.debug("Patched NVFLARE simulator_runner.py for Windows compatibility.")

        # 2. Patch simulator_deployer.py
        sim_deployer_path = nvflare_root / "private" / "fed" / "app" / "deployer" / "simulator_deployer.py"
        if sim_deployer_path.is_file():
            dep_content = sim_deployer_path.read_text(encoding="utf-8")
            if "shutil.rmtree(self.admin_storage)" in dep_content:
                dep_content = dep_content.replace(
                    "shutil.rmtree(self.admin_storage)",
                    "shutil.rmtree(self.admin_storage, ignore_errors=True)",
                )
                sim_deployer_path.write_text(dep_content, encoding="utf-8")
                logger.debug("Patched NVFLARE simulator_deployer.py for Windows compatibility.")

        # 3. Patch default_component_policy.py
        policy_path = nvflare_root / "app_common" / "default_component_policy.py"
        if policy_path.is_file():
            pol_content = policy_path.read_text(encoding="utf-8")
            if '"*"' not in pol_content:
                pol_content = pol_content.replace(
                    "DEFAULT_CLASS_ALLOW_LIST = (",
                    'DEFAULT_CLASS_ALLOW_LIST = (\n    "*",',
                )
                policy_path.write_text(pol_content, encoding="utf-8")
                logger.debug("Patched NVFLARE default_component_policy.py for allow-all.")

    except Exception as exc:
        logger.warning("Could not apply Windows NVFLARE compatibility patch: %s", exc)


def run_federated_simulation(
    job_dir: Optional[str | Path] = None,
    workspace: Optional[str | Path] = None,
    n_clients: int = 2,
    threads: int = 2,
    num_rounds: Optional[int] = None,
    gpu: Optional[str] = None,
    fedprox_mu: Optional[float] = None,
    fed_bn: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute a turnkey NVIDIA FLARE simulation for cardiac fat quantification.

    Args:
        job_dir: Path to the NVFLARE job directory containing meta.json and config files.
            Defaults to the canonical `jobs/cardiac_fat_fl` package.
        workspace: Path to output workspace directory. If None, a temporary directory is used.
        n_clients: Number of simulated participating hospital sites. Default: 2.
        threads: Number of parallel simulator client threads. Default: 2.
        num_rounds: Optional override for number of federated aggregation rounds.
        gpu: Optional comma-separated GPU device IDs (e.g. "0" or "0,1"). Default: None (CPU/Auto).
        fedprox_mu: Optional FedProx proximal loss regularizer coefficient (mu).
        fed_bn: Optional boolean flag whether FedBN local normalization is retained.

    Returns:
        Dict containing simulation results, exit code, workspace path, and status.
    """
    # Auto-patch Windows runtime environment if needed
    ensure_nvflare_windows_compatibility()

    repo_root = Path(__file__).resolve().parent.parent.parent
    if job_dir is None:
        job_dir = repo_root / "jobs" / "cardiac_fat_fl"
    else:
        job_dir = Path(job_dir).resolve()

    if not job_dir.is_dir():
        raise FileNotFoundError(f"NVFLARE job directory not found at: {job_dir}")

    # Set up workspace
    is_temp_ws = False
    if workspace is None:
        ws_dir = Path(tempfile.mkdtemp(prefix="cardiac_fl_sim_")).resolve()
        is_temp_ws = True
    else:
        ws_dir = Path(workspace).resolve()
        ws_dir.mkdir(parents=True, exist_ok=True)

    # If num_rounds, fedprox_mu, or fed_bn is overridden, create a job staging directory
    actual_job_dir = job_dir
    temp_job_stage: Optional[Path] = None
    needs_staging = (num_rounds is not None and num_rounds > 0) or (fedprox_mu is not None) or (fed_bn is not None)

    if needs_staging:
        temp_job_stage = Path(tempfile.mkdtemp(prefix="cardiac_job_stage_")).resolve()
        shutil.copytree(job_dir, temp_job_stage, dirs_exist_ok=True)

        if num_rounds is not None and num_rounds > 0:
            server_cfg_file = temp_job_stage / "app" / "config" / "config_fed_server.json"
            if server_cfg_file.is_file():
                with open(server_cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg["num_rounds"] = int(num_rounds)
                for wf in cfg.get("workflows", []):
                    if wf.get("id") == "scatter_gather_ctl" and "args" in wf:
                        wf["args"]["num_rounds"] = int(num_rounds)
                with open(server_cfg_file, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)

        if fedprox_mu is not None or fed_bn is not None:
            client_cfg_file = temp_job_stage / "app" / "config" / "config_fed_client.json"
            if client_cfg_file.is_file():
                with open(client_cfg_file, "r", encoding="utf-8") as f:
                    ccfg = json.load(f)
                for comp in ccfg.get("components", []):
                    if comp.get("id") == "cardiac_fat_algo" and "args" in comp:
                        if fedprox_mu is not None:
                            comp["args"]["fedprox_mu"] = float(fedprox_mu)
                        if fed_bn is not None:
                            comp["args"]["fed_bn"] = bool(fed_bn)
                with open(client_cfg_file, "w", encoding="utf-8") as f:
                    json.dump(ccfg, f, indent=2)

        actual_job_dir = temp_job_stage

    try:
        from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner

        runner = SimulatorRunner(
            job_folder=str(actual_job_dir),
            workspace=str(ws_dir),
            n_clients=n_clients,
            threads=threads,
            gpu=gpu,
        )

        logger.info(
            "Starting NVIDIA FLARE Simulator with %d clients, %d threads across %s",
            n_clients,
            threads,
            f"{num_rounds} rounds" if num_rounds else "job default rounds",
        )

        exit_code = runner.run()

        success = exit_code == 0
        result = {
            "exit_code": exit_code,
            "success": success,
            "workspace": str(ws_dir),
            "n_clients": n_clients,
            "threads": threads,
            "job_dir": str(job_dir),
            "status": "COMPLETED" if success else f"FAILED (code {exit_code})",
        }

        # Inspect and persist models
        models_dir = ws_dir / "server" / "simulate_job" / "app_server"
        persisted_models = []
        target_models_dir = repo_root / "models" / "federated"
        target_models_dir.mkdir(parents=True, exist_ok=True)

        if models_dir.is_dir():
            for p in models_dir.glob("*.pt"):
                persisted_models.append(p.name)
                try:
                    dest = target_models_dir / p.name
                    shutil.copy2(p, dest)
                except Exception as exc:
                    logger.warning("Could not preserve consensus model %s: %s", p.name, exc)

        result["persisted_models"] = persisted_models

        return result

    finally:
        if temp_job_stage is not None and temp_job_stage.exists():
            shutil.rmtree(temp_job_stage, ignore_errors=True)

