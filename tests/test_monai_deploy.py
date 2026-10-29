"""Unit tests for MONAI Deploy Application Package (MAP) operators and pipeline."""

from pathlib import Path
import json
import numpy as np
import pydicom
import pytest
import yaml
from typer.testing import CliRunner

from cardiac_fat_fl.cli import app
from cardiac_fat_fl.measurement_schema import (
    MEASUREMENT_CSV_COLUMNS,
    validate_measurement_dataframe,
)
from cardiac_fat_fl.monai_deploy import (
    CardiacCTStudyLoaderOp,
    CardiacFatQuantApplication,
    CardiacFatQuantifierOp,
    DicomSegAndSrWriterOp,
    run_monai_deploy_pipeline,
)

runner = CliRunner()


def test_cardiac_ct_study_loader_op(synthetic_nrrd_scan: Path) -> None:
    """Test CardiacCTStudyLoaderOp ingests volume correctly into MetaTensor."""
    loader = CardiacCTStudyLoaderOp()
    result = loader.compute(input_path=synthetic_nrrd_scan)

    assert "participant_id" in result
    assert "scan_path" in result
    assert "image" in result
    assert result["image"].ndim == 4  # (1, D, H, W)
    assert result["participant_id"] == synthetic_nrrd_scan.parent.name


def test_cardiac_fat_quantifier_op(synthetic_nrrd_scan: Path) -> None:
    """Test CardiacFatQuantifierOp executes segmentation and adipose quantification."""
    loader = CardiacCTStudyLoaderOp()
    study_data = loader.compute(input_path=synthetic_nrrd_scan)

    quantifier = CardiacFatQuantifierOp(models_dir=None, pvat_radius_mm=10.0)
    quant_data = quantifier.compute(study_dict=study_data)

    assert "pred" in quant_data
    assert "metrics" in quant_data
    assert quant_data["pred"].shape[0] == 8  # 8 compartments
    metrics = quant_data["metrics"]
    assert "EAT Total Volume (cm3)" in metrics
    assert "PVAT Total Volume (cm3)" in metrics


def test_dicom_seg_and_sr_writer_op(synthetic_nrrd_scan: Path, tmp_path: Path) -> None:
    """Test DicomSegAndSrWriterOp emits standard DICOM-SEG, DICOM-SR TID 1500, and metrics."""
    loader = CardiacCTStudyLoaderOp()
    study_data = loader.compute(input_path=synthetic_nrrd_scan)

    quantifier = CardiacFatQuantifierOp(models_dir=None)
    quant_data = quantifier.compute(study_dict=study_data)

    writer = DicomSegAndSrWriterOp()
    output_dir = tmp_path / "map_output"
    artifacts = writer.compute(quant_dict=quant_data, output_dir=output_dir)

    # 1. eat_pvat.seg.nrrd exists
    assert artifacts["eat_pvat_seg_nrrd"].exists()

    # 2. DICOM-SEG exists and has standard SOP Class UID
    seg_dcm = artifacts["dicom_seg"]
    assert seg_dcm.exists()
    ds_seg = pydicom.dcmread(str(seg_dcm))
    assert ds_seg.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
    assert hasattr(ds_seg, "SegmentSequence")
    assert len(ds_seg.SegmentSequence) == 8

    # 3. DICOM-SR TID 1500 exists and has standard SOP Class UID
    sr_dcm = artifacts["dicom_sr"]
    assert sr_dcm.exists()
    ds_sr = pydicom.dcmread(str(sr_dcm))
    assert ds_sr.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.34"

    # 4. JSON & CSV metrics exist and validate
    assert artifacts["metrics_json"].exists()
    json_data = json.loads(artifacts["metrics_json"].read_text(encoding="utf-8"))
    assert "EAT Total Volume (cm3)" in json_data

    assert artifacts["metrics_csv"].exists()
    import pandas as pd

    df = pd.read_csv(artifacts["metrics_csv"])
    assert list(df.columns) == MEASUREMENT_CSV_COLUMNS
    validate_measurement_dataframe(df)


def test_run_monai_deploy_pipeline_end_to_end(synthetic_nrrd_scan: Path, tmp_path: Path) -> None:
    """Test top-level run_monai_deploy_pipeline function."""
    output_dir = tmp_path / "pipeline_out"
    artifacts = run_monai_deploy_pipeline(
        input_path=synthetic_nrrd_scan,
        output_path=output_dir,
        pvat_radius_mm=10.0,
    )

    assert artifacts["dicom_seg"].exists()
    assert artifacts["dicom_sr"].exists()
    assert artifacts["metrics_csv"].exists()


def test_cli_map_run_command(synthetic_nrrd_scan: Path, tmp_path: Path) -> None:
    """Test CLI cardiac-fat-fl map-run execution."""
    output_dir = tmp_path / "cli_map_out"
    result = runner.invoke(
        app,
        [
            "map-run",
            "--input",
            str(synthetic_nrrd_scan),
            "--output",
            str(output_dir),
            "--pvat-radius",
            "10.0",
        ],
    )
    assert result.exit_code == 0, f"CLI output: {result.output}"
    assert "MONAI Deploy pipeline completed successfully!" in result.output
    assert (output_dir / "eat_pvat_seg.dcm").exists()
    assert (output_dir / "eat_pvat_sr.dcm").exists()
    assert (output_dir / "cardiac_fat_measurements.csv").exists()


def test_kubernetes_manifest_contract() -> None:
    """Test k8s/hospital-clinical-testbed.yaml syntax, resource kinds, and port mappings."""
    k8s_file = Path("k8s/hospital-clinical-testbed.yaml")
    assert k8s_file.exists(), "k8s manifest file must exist"

    docs = list(yaml.safe_load_all(k8s_file.read_text(encoding="utf-8")))
    assert len(docs) == 4, f"Expected 4 manifests (Namespace, Deployment, Service, Job), got {len(docs)}"

    kinds = {d["kind"]: d for d in docs}
    assert "Namespace" in kinds
    assert "Deployment" in kinds
    assert "Service" in kinds
    assert "Job" in kinds

    # 1. Namespace validation
    ns = kinds["Namespace"]
    assert ns["metadata"]["name"] == "clinical-ai"

    # 2. Deployment validation
    deploy = kinds["Deployment"]
    assert deploy["metadata"]["name"] == "orthanc-pacs"
    assert deploy["metadata"]["namespace"] == "clinical-ai"
    container = deploy["spec"]["template"]["spec"]["containers"][0]
    assert "orthancteam/orthanc" in container["image"]
    ports = {p["name"]: p["containerPort"] for p in container["ports"]}
    assert ports.get("http-web") == 8042
    assert ports.get("dicom-dimse") == 4242

    # 3. Service validation & label matching
    svc = kinds["Service"]
    assert svc["metadata"]["name"] == "orthanc-pacs"
    assert svc["metadata"]["namespace"] == "clinical-ai"
    assert svc["spec"]["selector"] == deploy["spec"]["selector"]["matchLabels"]
    svc_ports = {p["name"]: p["port"] for p in svc["spec"]["ports"]}
    assert svc_ports.get("http-web") == 8042
    assert svc_ports.get("dicom-dimse") == 4242

    # 4. Job validation
    job = kinds["Job"]
    assert job["metadata"]["name"] == "cardiac-fat-quant-job"
    assert job["metadata"]["namespace"] == "clinical-ai"
    assert job["spec"]["template"]["spec"]["restartPolicy"] == "OnFailure"


def test_docker_compose_manifest_contract() -> None:
    """Test monai_deploy_roadmap/docker-compose.monai-deploy.yml services, ports, and profile isolation."""
    compose_file = Path("monai_deploy_roadmap/docker-compose.monai-deploy.yml")
    assert compose_file.exists(), "docker-compose file must exist"

    data = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    assert "services" in data
    services = data["services"]

    # 1. Default standalone orthanc-pacs service
    assert "orthanc-pacs" in services
    orthanc = services["orthanc-pacs"]
    assert "8042:8042" in orthanc["ports"]
    assert "4242:4242" in orthanc["ports"]
    env = orthanc.get("environment", {})
    assert env.get("ORTHANC__DICOM_WEB__ENABLE") == "true"
    assert env.get("ORTHANC__AUTHENTICATION_ENABLED") == "false"

    # 2. Profiles on optional advanced services (so default up -d is never blocked)
    assert "monai-informatics-gateway" in services
    mig = services["monai-informatics-gateway"]
    assert "profiles" in mig
    assert "mig" in mig["profiles"]

    assert "cardiac-fat-map" in services
    map_worker = services["cardiac-fat-map"]
    assert "profiles" in map_worker
    assert "worker" in map_worker["profiles"]


def test_monai_deploy_app_yaml_contract() -> None:
    """Test monai_deploy_roadmap/app.yaml against MONAI Deploy and NVIDIA Holoscan SDK contracts."""
    app_yaml_file = Path("monai_deploy_roadmap/app.yaml")
    assert app_yaml_file.exists(), "app.yaml manifest must exist"

    manifest = yaml.safe_load(app_yaml_file.read_text(encoding="utf-8"))

    # 1. Application metadata
    app_meta = manifest.get("application", {})
    assert app_meta.get("name") == "monai-cardiac-fat-map"
    assert app_meta.get("version") == "1.0.0"

    # 2. Input/Output contracts conforming to /var/monai mount standard
    input_spec = manifest.get("input", {})
    assert input_spec.get("path") == "/var/monai/input"
    assert "dicom" in input_spec.get("formats", [])

    output_spec = manifest.get("output", {})
    assert output_spec.get("path") == "/var/monai/output"
    assert "dicom-seg" in output_spec.get("formats", [])
    assert "dicom-sr" in output_spec.get("formats", [])

    # 3. Model contracts
    models_spec = manifest.get("models", {})
    assert models_spec.get("path") == "/opt/holoscan/models"
    model_names = [m["name"] for m in models_spec.get("artifacts", [])]
    assert "deepheartseg_heart.onnx" in model_names
    assert "deepheartseg_loc.onnx" in model_names
    assert "wholeBody_ct_segmentation/models/model.pt" in model_names

