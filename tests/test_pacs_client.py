"""Unit and integration tests for PACS export client (STOW-RS & C-STORE)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from cardiac_fat_fl.pacs_client import (
    find_dicom_instances_to_push,
    push_to_dicomweb_stow,
    push_to_dimse_cstore,
)


def _create_minimal_dicom(file_path: Path, sop_class_uid: str = SecondaryCaptureImageStorage) -> Path:
    """Helper to generate a minimal valid DICOM file on disk."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = "TEST_PATIENT"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()

    file_path.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(str(file_path), ds, write_like_original=False)
    return file_path


def test_find_dicom_instances_single_file(tmp_path: Path):
    dcm = _create_minimal_dicom(tmp_path / "test.dcm")
    found = find_dicom_instances_to_push(dcm)
    assert len(found) == 1
    assert found[0] == dcm


def test_find_dicom_instances_in_ohif_export(tmp_path: Path):
    export_dir = tmp_path / "ohif_export"
    seg = _create_minimal_dicom(export_dir / "eat_pvat_seg.dcm")
    ct1 = _create_minimal_dicom(export_dir / "slice_001.dcm")
    ct2 = _create_minimal_dicom(export_dir / "slice_002.dcm")

    # Default: only seg
    found_seg_only = find_dicom_instances_to_push(tmp_path, include_ct=False)
    assert len(found_seg_only) == 1
    assert found_seg_only[0] == seg

    # With CT
    found_all = find_dicom_instances_to_push(tmp_path, include_ct=True)
    assert len(found_all) == 3
    assert seg in found_all
    assert ct1 in found_all
    assert ct2 in found_all


def test_push_to_dicomweb_stow_multipart_success(tmp_path: Path):
    dcm1 = _create_minimal_dicom(tmp_path / "inst1.dcm")
    dcm2 = _create_minimal_dicom(tmp_path / "inst2.dcm")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "OK"

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = push_to_dicomweb_stow(
            [dcm1, dcm2],
            stow_url="http://orthanc.hospital.internal:8042/dicom-web/studies",
            auth=("dr_smith", "secret"),
        )

        assert res["success_count"] == 2
        assert res["failed_count"] == 0
        assert res["status_code"] == 200
        assert mock_post.called

        # Verify multipart headers
        called_headers = mock_post.call_args[1]["headers"]
        assert "multipart/related" in called_headers["Content-Type"]
        assert "application/dicom" in called_headers["Content-Type"]


def test_push_to_dicomweb_stow_bearer_token(tmp_path: Path):
    dcm = _create_minimal_dicom(tmp_path / "eat_pvat_seg.dcm")

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = push_to_dicomweb_stow(
            [dcm],
            stow_url="http://cloud-pacs.io/dicom-web/studies",
            token="eyJh...jwt",
        )
        assert res["success_count"] == 1
        called_headers = mock_post.call_args[1]["headers"]
        assert called_headers["Authorization"] == "Bearer eyJh...jwt"


def test_push_to_dicomweb_orthanc_raw(tmp_path: Path):
    dcm = _create_minimal_dicom(tmp_path / "eat_pvat_seg.dcm")

    mock_resp = MagicMock()
    mock_resp.status_code = 201

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = push_to_dicomweb_stow(
            [dcm],
            stow_url="http://localhost:8042/instances",
        )
        assert res["success_count"] == 1
        called_headers = mock_post.call_args[1]["headers"]
        assert called_headers["Content-Type"] == "application/dicom"


def test_push_to_dimse_cstore_mock(tmp_path: Path):
    dcm = _create_minimal_dicom(tmp_path / "eat_pvat_seg.dcm")

    mock_status = MagicMock()
    mock_status.Status = 0x0000

    mock_assoc = MagicMock()
    mock_assoc.is_established = True
    mock_assoc.send_c_store.return_value = mock_status

    with patch("pynetdicom.AE.associate", return_value=mock_assoc):
        res = push_to_dimse_cstore(
            [dcm],
            host="192.168.1.100",
            port=104,
            called_aet="CENTRICITY",
            calling_aet="CARDIAC_FAT_FL",
        )

        assert res["success_count"] == 1
        assert res["failed_count"] == 0
        assert res["status_code"] == 0x0000
        assert mock_assoc.release.called
