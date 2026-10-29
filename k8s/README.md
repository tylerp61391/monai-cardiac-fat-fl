# Clinical Kubernetes Infrastructure (`k8s/`)

This directory provides turn-key, enterprise-grade Kubernetes manifests for deploying `monai-cardiac-fat-fl` within institutional hospital networks, academic medical centers, and cloud Kubernetes environments (AWS EKS, Azure AKS, Google Cloud GKE, Red Hat OpenShift, or local Docker Desktop / kind).

---

## Architecture Overview

The manifest [`hospital-clinical-testbed.yaml`](hospital-clinical-testbed.yaml) provisions an isolated, self-contained clinical AI staging environment:

| Resource | Kind | Namespace | Ports / Endpoints | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **`clinical-ai`** | `Namespace` | — | — | Logical isolation separating clinical AI workflows from standard hospital workloads. |
| **`orthanc-pacs`** | `Deployment` | `clinical-ai` | `8042` (HTTP / DICOMweb)<br>`4242` (DIMSE C-STORE) | Lightweight, standard DICOM PACS archive and web viewer pre-configured with DICOMweb enabled. |
| **`orthanc-pacs`** | `Service` | `clinical-ai` | ClusterIP (`8042`, `4242`) | Internal cluster DNS routing (`http://orthanc-pacs.clinical-ai.svc.cluster.local:8042`). |
| **`cardiac-fat-quant-job`** | `Job` | `clinical-ai` | — | Batch inference template executing the MONAI Deploy Application Package (MAP) DAG on incoming CT scans. |

---

## Quickstart Guide

### 1. Deploy the Clinical Testbed

Ensure your Kubernetes cluster is active (`kubectl get nodes`), then apply the manifest:

```bash
kubectl apply -f k8s/hospital-clinical-testbed.yaml
```

Verify deployment status:

```bash
kubectl get all -n clinical-ai
```

### 2. Access the PACS Web Explorer

Forward the internal PACS service to your local machine:

```bash
kubectl port-forward svc/orthanc-pacs -n clinical-ai 8043:8042
```

Open your browser to:
- **Orthanc PACS Explorer**: [http://localhost:8043](http://localhost:8043)
- **DICOMweb Endpoint**: `http://localhost:8043/dicom-web/studies`

### 3. Push a Study into the Kubernetes PACS

Transmit a patient scan (including raw CT series, 8-compartment DICOM-SEG, and DICOM-SR TID 1500) into the Kubernetes cluster:

```bash
# Via DICOMweb STOW-RS:
cardiac-fat-fl push "example_source_dir/0506" --url "http://localhost:8043/dicom-web/studies" --include-ct

# Or via traditional DICOM DIMSE C-STORE (after port-forwarding 4243:4242):
cardiac-fat-fl push "example_source_dir/0506" --host localhost --port 4243 --called-aet ORTHANC
```

Refresh [http://localhost:8043](http://localhost:8043) to inspect the 3 series indexed in the cluster database.

### 4. Tear Down & Clean Up

To cleanly delete the namespace and all associated pods, services, and jobs:

```bash
kubectl delete -f k8s/hospital-clinical-testbed.yaml
```

