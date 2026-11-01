"""Tests validating CI/CD GitHub Actions workflows, Docker build manifests, and package dependencies."""

from pathlib import Path
import yaml
import tomllib


def test_ci_workflow_structure() -> None:
    """Validate .github/workflows/ci.yml structure, triggers, matrix, and jobs."""
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.exists(), "ci.yml must exist"

    data = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
    assert "name" in data
    triggers = data.get("on") or data.get(True)
    assert triggers is not None, "Workflow must define triggers ('on')"
    assert "push" in triggers
    assert "pull_request" in triggers
    assert "main" in triggers["push"]["branches"]
    assert "main" in triggers["pull_request"]["branches"]

    jobs = data.get("jobs", {})
    assert "lint" in jobs, "CI must contain a fast lint job"
    assert "test" in jobs, "CI must contain a test job"
    assert "docker-verify" in jobs, "CI must contain a docker verification job"

    # Verify lint job runs markdownlint and ruff
    lint_job = jobs["lint"]
    lint_steps = [s.get("name", "") for s in lint_job.get("steps", [])]
    assert any("markdownlint" in s.lower() for s in lint_steps)
    assert any("ruff" in s.lower() for s in lint_steps)

    # Verify test matrix
    test_job = jobs["test"]
    strategy = test_job.get("strategy", {})
    matrix = strategy.get("matrix", {})
    assert "ubuntu-latest" in matrix.get("os", [])
    assert "windows-latest" in matrix.get("os", [])
    assert "3.11" in matrix.get("python-version", [])
    assert "3.12" in matrix.get("python-version", [])


def test_docker_map_workflow_structure() -> None:
    """Validate .github/workflows/docker-map.yml for MONAI Deploy MAP image publication."""
    map_wf = Path(".github/workflows/docker-map.yml")
    assert map_wf.exists(), "docker-map.yml must exist"

    data = yaml.safe_load(map_wf.read_text(encoding="utf-8"))
    assert "jobs" in data
    assert "build-and-push-map" in data["jobs"]

    job = data["jobs"]["build-and-push-map"]
    steps = job.get("steps", [])
    build_step = next(s for s in steps if "build-push-action" in s.get("uses", ""))
    with_clause = build_step.get("with", {})
    assert with_clause.get("file") == "./deploy/Dockerfile"
    assert data["env"]["IMAGE_NAME"] == "${{ github.repository_owner }}/monai-cardiac-fat-map"


def test_docker_publish_workflow_structure() -> None:
    """Validate .github/workflows/docker-publish.yml for standard CLI image publication."""
    pub_wf = Path(".github/workflows/docker-publish.yml")
    assert pub_wf.exists(), "docker-publish.yml must exist"

    data = yaml.safe_load(pub_wf.read_text(encoding="utf-8"))
    assert "jobs" in data
    assert "build-and-push-image" in data["jobs"]


def test_security_workflow_structure() -> None:
    """Validate .github/workflows/security.yml security auditing jobs."""
    sec_wf = Path(".github/workflows/security.yml")
    assert sec_wf.exists(), "security.yml must exist"

    data = yaml.safe_load(sec_wf.read_text(encoding="utf-8"))
    jobs = data.get("jobs", {})
    assert "dependency-audit" in jobs
    assert "sast-scan" in jobs
    assert "trivy-fs-scan" in jobs


def test_pyproject_dependencies_completeness() -> None:
    """Verify that all core runtime modules (including flask and pymupdf) are declared."""
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "pyproject.toml must exist"

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])

    dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].strip() for d in deps]
    assert "flask" in dep_names, "flask must be declared in dependencies for dashboard and fhir server"
    assert "pymupdf" in dep_names, "pymupdf must be declared in dependencies for clinical report pdf"
    assert "nvflare" in dep_names
    assert "monai" in dep_names
    assert "torch" in dep_names
    assert "pydicom" in dep_names
    assert "pynetdicom" in dep_names

    # Verify ruff and coverage configurations
    tools = data.get("tool", {})
    assert "ruff" in tools, "ruff configuration must exist in pyproject.toml"
    assert "coverage" in tools, "coverage configuration must exist in pyproject.toml"
