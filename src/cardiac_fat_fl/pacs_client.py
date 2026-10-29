"""PACS Client for DICOMweb (STOW-RS) and DIMSE (C-STORE) Export.

Supports transmitting cardiac CT volumes and DICOM-SEG (SOP 1.2.840.10008.5.1.4.1.1.66.4)
to clinical and research PACS archives (Orthanc, dcm4chee-arc, Sectra, GE, Siemens, Philips).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pydicom
import requests


def find_dicom_instances_to_push(
    target_path: Union[str, Path],
    include_ct: bool = False,
) -> List[Path]:
    """Discover DICOM instances to transmit from a patient folder or export directory.

    Args:
        target_path: Path to participant folder, `ohif_export` directory, or `.dcm` file.
        include_ct: If True, include original or exported CT slice series in addition to DICOM-SEG.

    Returns:
        List of Path objects for valid DICOM files to push.
    """
    path = Path(target_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Target path does not exist: {path}")

    # Single DICOM file
    if path.is_file():
        if path.suffix.lower() == ".dcm" or _is_dicom_file(path):
            return [path]
        raise ValueError(f"Specified file is not a DICOM file: {path}")

    # Directory search
    seg_candidates = [
        path / "eat_pvat_seg.dcm",
        path / "ohif_export" / "eat_pvat_seg.dcm",
    ]
    found_seg: Optional[Path] = None
    for cand in seg_candidates:
        if cand.exists():
            found_seg = cand
            break

    # Deep glob if not found in standard paths
    if found_seg is None:
        matches = list(path.glob("*eat_pvat_seg*.dcm")) + list(path.glob("**/*eat_pvat_seg*.dcm"))
        if matches:
            found_seg = matches[0]

    instances: List[Path] = []

    if include_ct:
        # Search for CT slices in ohif_export or directory
        ct_dirs = [path / "ohif_export" / "ct", path / "ohif_export", path, path / "ct"]
        ct_files: List[Path] = []
        for d in ct_dirs:
            if d.exists() and d.is_dir():
                dcms = sorted(
                    [
                        f for f in d.glob("*.dcm")
                        if "eat_pvat_seg" not in f.name.lower() and _is_dicom_file(f)
                    ],
                    key=lambda p: p.name,
                )
                if dcms:
                    ct_files = dcms
                    break

        if not ct_files:
            # Check subdirectories
            for sub in path.iterdir():
                if sub.is_dir() and sub.name != "ohif_export":
                    dcms = sorted([f for f in sub.glob("*.dcm") if _is_dicom_file(f)])
                    if dcms:
                        ct_files = dcms
                        break

        instances.extend(ct_files)

    # Search for Structured Report (eat_pvat_sr.dcm)
    sr_candidates = [
        path / "eat_pvat_sr.dcm",
        path / "ohif_export" / "eat_pvat_sr.dcm",
    ]
    found_sr: Optional[Path] = None
    for cand in sr_candidates:
        if cand.exists():
            found_sr = cand
            break

    if found_sr is None:
        matches = list(path.glob("*eat_pvat_sr*.dcm")) + list(path.glob("**/*eat_pvat_sr*.dcm"))
        if matches:
            found_sr = matches[0]

    # Search for Encapsulated PDF Clinical Report (eat_pvat_report.dcm)
    report_candidates = [
        path / "eat_pvat_report.dcm",
        path / "ohif_export" / "eat_pvat_report.dcm",
    ]
    found_report: Optional[Path] = None
    for cand in report_candidates:
        if cand.exists():
            found_report = cand
            break

    if found_report is None:
        matches = list(path.glob("*eat_pvat_report*.dcm")) + list(path.glob("**/*eat_pvat_report*.dcm"))
        if matches:
            found_report = matches[0]

    # Always ensure segmentation, structured report, and clinical report are included if available
    if found_seg is not None and found_seg not in instances:
        instances.append(found_seg)
    if found_sr is not None and found_sr not in instances:
        instances.append(found_sr)
    if found_report is not None and found_report not in instances:
        instances.append(found_report)

    if not instances and found_seg is None and found_sr is None and found_report is None:
        # If no specific seg found, gather all .dcm files in path
        all_dcms = [f for f in path.glob("*.dcm") if _is_dicom_file(f)]
        if all_dcms:
            instances.extend(all_dcms)
        else:
            raise FileNotFoundError(
                f"No DICOM instances (CT or eat_pvat_seg.dcm) found in {path}. "
                "Ensure export-ohif or quantify --save-seg has been run."
            )

    return instances


def _is_dicom_file(file_path: Path) -> bool:
    """Check if file has DICM preamble or is readable by pydicom."""
    try:
        with open(file_path, "rb") as f:
            f.seek(128)
            preamble = f.read(4)
            if preamble == b"DICM":
                return True
        # Fallback quick read
        pydicom.dcmread(str(file_path), stop_before_pixels=True)
        return True
    except Exception:
        return False


def push_to_dicomweb_stow(
    files: List[Path],
    stow_url: str,
    auth: Optional[Tuple[str, str]] = None,
    token: Optional[str] = None,
    timeout: int = 120,
    ca_cert: Optional[Union[str, Path, bool]] = None,
) -> Dict[str, Any]:
    """Store DICOM instances to a DICOMweb STOW-RS server or Orthanc REST endpoint.

    Supports standard multipart/related STOW-RS:
      POST {stow_url}
      Content-Type: multipart/related; type="application/dicom"; boundary={boundary}

    Also supports direct Orthanc /instances endpoint:
      POST http://localhost:8042/instances
      Content-Type: application/dicom

    Args:
        files: List of DICOM file paths to store.
        stow_url: Target STOW-RS URL (e.g. `http://orthanc:8042/dicom-web/studies` or `/instances`).
        auth: Optional (username, password) tuple for HTTP Basic Auth.
        token: Optional Bearer token for Authorization header.
        timeout: Request timeout in seconds.
        ca_cert: Path to custom CA certificate bundle file/directory, or False to disable SSL verification.

    Returns:
        Dict with status summary (`success_count`, `failed_count`, `status_code`, `details`).
    """
    if not files:
        return {"success_count": 0, "failed_count": 0, "status_code": 200, "details": "No files provided"}

    verify_param: Union[bool, str] = True
    if ca_cert is not None:
        if isinstance(ca_cert, bool):
            verify_param = ca_cert
        else:
            verify_param = str(ca_cert)

    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    is_orthanc_raw = stow_url.rstrip("/").endswith("/instances")

    if is_orthanc_raw:
        # Direct Orthanc single-instance REST API
        success = 0
        failed = 0
        details = []
        for file_path in files:
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                resp = requests.post(
                    stow_url,
                    data=data,
                    headers={**headers, "Content-Type": "application/dicom"},
                    auth=auth,
                    timeout=timeout,
                    verify=verify_param,
                )
                if resp.status_code in [200, 201, 202]:
                    success += 1
                else:
                    failed += 1
                    details.append(f"{file_path.name}: HTTP {resp.status_code} - {resp.text[:200]}")
            except Exception as e:
                failed += 1
                details.append(f"{file_path.name}: {str(e)}")

        return {
            "success_count": success,
            "failed_count": failed,
            "status_code": 200 if failed == 0 else 207,
            "details": details or "All instances stored successfully via Orthanc REST",
        }

    # Standard DICOMweb STOW-RS multipart/related
    boundary = f"----DICOMWebBoundary{uuid.uuid4().hex}"
    body_parts = []

    for file_path in files:
        with open(file_path, "rb") as f:
            dicom_bytes = f.read()

        part_header = (
            f"--{boundary}\r\n"
            f"Content-Type: application/dicom\r\n"
            f"Content-Length: {len(dicom_bytes)}\r\n\r\n"
        ).encode("utf-8")
        body_parts.append(part_header + dicom_bytes + b"\r\n")

    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    payload = b"".join(body_parts)

    headers["Content-Type"] = f'multipart/related; type="application/dicom"; boundary={boundary}'

    resp = requests.post(
        stow_url,
        data=payload,
        headers=headers,
        auth=auth,
        timeout=timeout,
        verify=verify_param,
    )

    if resp.status_code in [200, 201, 202]:
        return {
            "success_count": len(files),
            "failed_count": 0,
            "status_code": resp.status_code,
            "details": f"Successfully stored {len(files)} instances to {stow_url}",
        }
    else:
        return {
            "success_count": 0,
            "failed_count": len(files),
            "status_code": resp.status_code,
            "details": f"STOW-RS failed with HTTP {resp.status_code}: {resp.text[:400]}",
        }


def push_to_dimse_cstore(
    files: List[Path],
    host: str,
    port: int,
    called_aet: str = "ANY-SCP",
    calling_aet: str = "CARDIAC_FAT_FL",
    timeout: int = 60,
) -> Dict[str, Any]:
    """Store DICOM instances to a standard PACS SCP using DICOM DIMSE C-STORE.

    Uses `pynetdicom` to negotiate association contexts (CT Image Storage,
    Segmentation Storage, etc.) and transfers files one by one.

    Args:
        files: List of DICOM file paths to store.
        host: Target PACS hostname or IP address.
        port: Target PACS port (e.g. 104, 4242, 11112).
        called_aet: Target AE Title (Application Entity Title).
        calling_aet: Local AE Title.
        timeout: Association and response timeout in seconds.

    Returns:
        Dict with status summary (`success_count`, `failed_count`, `status_code`, `details`).
    """
    try:
        from pynetdicom import AE, StoragePresentationContexts
        from pynetdicom.sop_class import SegmentationStorage, CTImageStorage
    except ImportError:
        raise ImportError(
            "pynetdicom is required for DIMSE C-STORE push. "
            "Install it with: pip install pynetdicom"
        )

    ae = AE(ae_title=calling_aet)
    # Support all standard storage SOP classes (CT, MR, SEG, SC, etc.)
    ae.requested_contexts = StoragePresentationContexts

    assoc = ae.associate(host, port, ae_title=called_aet, evt_handlers=None)
    if not assoc.is_established:
        return {
            "success_count": 0,
            "failed_count": len(files),
            "status_code": 0x0122,  # Association rejected
            "details": f"Failed to establish DICOM association with {called_aet}@{host}:{port}",
        }

    success = 0
    failed = 0
    details = []

    try:
        for file_path in files:
            ds = pydicom.dcmread(str(file_path))
            status = assoc.send_c_store(ds)
            if status and status.Status in [0x0000, 0xB000, 0xB006, 0xB007]:
                success += 1
            else:
                failed += 1
                status_hex = hex(status.Status) if status else "No response"
                details.append(f"{file_path.name}: C-STORE status {status_hex}")
    finally:
        assoc.release()

    return {
        "success_count": success,
        "failed_count": failed,
        "status_code": 0x0000 if failed == 0 else 0x0111,
        "details": details or f"Successfully transmitted {success} instances via C-STORE",
    }

