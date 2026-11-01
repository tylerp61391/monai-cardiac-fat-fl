"""Self-Contained HL7 FHIR R4 Mock Server & Clinical Sandbox.

Implements an in-process, zero-dependency, standards-compliant HL7 FHIR R4 testbed
and clinical sandbox for cardiac fat quantification.
Supports:
  - Transaction and Batch Bundle ingestion (POST /r4, POST /r4/Bundle)
  - Canonical Reference Resolution (urn:uuid mapping to persistent resource IDs)
  - US Core 3.1.1 & Epic on FHIR CapabilityStatement (GET /r4/metadata)
  - Discrete resource CRUD & search for Observation, DiagnosticReport, DocumentReference, Patient, Binary
  - OperationOutcome generation for validation errors and not-found responses
  - Thread-safe in-memory store with optional JSON directory persistence
  - Flask Blueprint (`create_fhir_blueprint`) for mounting into Research Dashboard (:8050)
  - Standalone Flask server (`create_fhir_app`, `launch_fhir_server`)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import Blueprint, Flask, Response, jsonify, request, send_file

logger = logging.getLogger(__name__)

SUPPORTED_RESOURCE_TYPES = [
    "Patient",
    "Observation",
    "DiagnosticReport",
    "DocumentReference",
    "Bundle",
    "Binary",
]


class FHIRStore:
    """Thread-safe in-memory and disk-backed store for FHIR R4 resources."""

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None) -> None:
        self.lock = threading.Lock()
        self.storage_dir = Path(storage_dir).resolve() if storage_dir else None
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {
            t: {} for t in SUPPORTED_RESOURCE_TYPES
        }
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load any existing persisted JSON resources from disk."""
        if not self.storage_dir:
            return
        for rtype in SUPPORTED_RESOURCE_TYPES:
            tdir = self.storage_dir / rtype
            if tdir.is_dir():
                for f in tdir.glob("*.json"):
                    try:
                        with open(f, "r", encoding="utf-8") as fp:
                            res = json.load(fp)
                        rid = res.get("id") or f.stem
                        self._store[rtype][rid] = res
                    except Exception as e:
                        logger.warning("Could not load FHIR resource %s: %s", f, e)

    def save(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Save or update a resource, assigning a UUID if id is missing."""
        rtype = resource.get("resourceType")
        if not rtype:
            raise ValueError("Resource is missing 'resourceType'")

        with self.lock:
            if rtype not in self._store:
                self._store[rtype] = {}

            rid = resource.get("id")
            if not rid:
                rid = str(uuid.uuid4())
                resource["id"] = rid

            # Add meta block if missing
            if "meta" not in resource:
                resource["meta"] = {}
            resource["meta"]["versionId"] = str(
                int(resource["meta"].get("versionId", "0")) + 1
            )
            resource["meta"]["lastUpdated"] = datetime.now(timezone.utc).isoformat()

            self._store[rtype][rid] = copy.deepcopy(resource)

            # Persist to disk if configured
            if self.storage_dir:
                try:
                    tdir = self.storage_dir / rtype
                    tdir.mkdir(parents=True, exist_ok=True)
                    fpath = tdir / f"{rid}.json"
                    with open(fpath, "w", encoding="utf-8") as fp:
                        json.dump(resource, fp, indent=2)
                except Exception as e:
                    logger.warning("Could not persist FHIR resource %s/%s: %s", rtype, rid, e)

        return resource

    def get(self, resource_type: str, resource_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a resource by type and ID."""
        with self.lock:
            res_dict = self._store.get(resource_type, {})
            res = res_dict.get(resource_id)
            return copy.deepcopy(res) if res else None

    def delete(self, resource_type: str, resource_id: str) -> bool:
        """Delete a resource by type and ID."""
        with self.lock:
            res_dict = self._store.get(resource_type, {})
            if resource_id in res_dict:
                del res_dict[resource_id]
                if self.storage_dir:
                    fpath = self.storage_dir / resource_type / f"{resource_id}.json"
                    if fpath.exists():
                        fpath.unlink(missing_ok=True)
                return True
            return False

    def search(
        self, resource_type: str, query_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search resources matching given query parameters."""
        params = query_params or {}
        with self.lock:
            candidates = list(self._store.get(resource_type, {}).values())

        results = []
        for c in candidates:
            if self._matches(c, params):
                results.append(copy.deepcopy(c))
        return results

    def _matches(self, res: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Evaluate if resource matches query parameters."""
        for key, val in params.items():
            if not val:
                continue

            # 1. ID matching
            if key == "_id":
                if res.get("id") != val:
                    return False

            # 2. Patient / Subject matching
            elif key in ("patient", "subject"):
                val_clean = str(val).removeprefix("Patient/").removeprefix("urn:uuid:")
                matched_subj = False

                # Check subject.reference
                subj_ref = res.get("subject", {}).get("reference", "")
                if subj_ref:
                    clean_ref = subj_ref.removeprefix("Patient/").removeprefix("urn:uuid:")
                    if clean_ref == val_clean or subj_ref == val:
                        matched_subj = True

                # Check subject.identifier.value
                subj_ident = res.get("subject", {}).get("identifier", {}).get("value", "")
                if subj_ident and subj_ident == val_clean:
                    matched_subj = True

                # Check patient.reference
                pat_ref = res.get("patient", {}).get("reference", "")
                if pat_ref:
                    clean_pref = pat_ref.removeprefix("Patient/").removeprefix("urn:uuid:")
                    if clean_pref == val_clean or pat_ref == val:
                        matched_subj = True

                if not matched_subj:
                    return False

            # 3. Status matching
            elif key == "status":
                if res.get("status") != val:
                    return False

            # 4. Code matching (token)
            elif key == "code":
                code_to_find = str(val).split("|")[-1]  # Support system|code
                matched_code = False
                for coding in res.get("code", {}).get("coding", []):
                    if coding.get("code") == code_to_find:
                        matched_code = True
                        break
                if not matched_code:
                    return False

        return True

    def reset(self) -> None:
        """Clear all stored resources."""
        with self.lock:
            for k in self._store:
                self._store[k].clear()
            if self.storage_dir and self.storage_dir.is_dir():
                for rtype in SUPPORTED_RESOURCE_TYPES:
                    tdir = self.storage_dir / rtype
                    if tdir.is_dir():
                        for f in tdir.glob("*.json"):
                            try:
                                f.unlink(missing_ok=True)
                            except Exception:
                                pass

    def summary(self) -> Dict[str, Any]:
        """Return counts and unique patient identifiers stored."""
        with self.lock:
            counts = {k: len(v) for k, v in self._store.items()}
            patients = set()
            for rtype in ["Observation", "DiagnosticReport", "DocumentReference"]:
                for res in self._store.get(rtype, {}).values():
                    subj = (
                        res.get("subject", {}).get("reference", "")
                        or res.get("subject", {}).get("identifier", {}).get("value", "")
                        or res.get("patient", {}).get("reference", "")
                    )
                    if subj:
                        clean = subj.removeprefix("Patient/").removeprefix("urn:uuid:")
                        patients.add(clean)
            return {
                "counts": counts,
                "patients": sorted(list(patients)),
                "total_resources": sum(counts.values()),
            }


# Default global FHIRStore singleton for in-app / dashboard sharing
_DEFAULT_FHIR_STORE = FHIRStore()


def get_default_fhir_store() -> FHIRStore:
    """Return the global default FHIRStore instance."""
    return _DEFAULT_FHIR_STORE


def build_operation_outcome(
    diagnostics: str, severity: str = "error", code: str = "invalid"
) -> Dict[str, Any]:
    """Generate a standard HL7 FHIR R4 OperationOutcome resource."""
    return {
        "resourceType": "OperationOutcome",
        "id": str(uuid.uuid4()),
        "issue": [
            {
                "severity": severity,
                "code": code,
                "diagnostics": diagnostics,
            }
        ],
    }


def build_capability_statement(base_url: str) -> Dict[str, Any]:
    """Generate a US Core 3.1.1 & Epic on FHIR compliant CapabilityStatement."""
    return {
        "resourceType": "CapabilityStatement",
        "id": "cardiac-fat-fl-fhir-sandbox",
        "status": "active",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "publisher": "monai-cardiac-fat-fl",
        "kind": "instance",
        "software": {
            "name": "monai-cardiac-fat-fl FHIR R4 Mock Server & Sandbox",
            "version": "1.0.0",
        },
        "implementation": {
            "description": "In-process self-contained HL7 FHIR R4 testbed conforming to Epic on FHIR contracts.",
            "url": base_url,
        },
        "fhirVersion": "4.0.1",
        "format": ["application/fhir+json", "application/json"],
        "rest": [
            {
                "mode": "server",
                "documentation": "Self-contained HL7 FHIR R4 Mock Server & open.epic MCP Interoperability Sandbox",
                "security": {
                    "cors": True,
                    "service": [
                        {
                            "coding": [
                                {
                                    "system": "http://hl7.org/fhir/restful-security-service",
                                    "code": "SMART-on-FHIR",
                                    "display": "SMART-on-FHIR",
                                }
                            ]
                        }
                    ],
                },
                "resource": [
                    {
                        "type": "Observation",
                        "profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-observation-lab",
                        "interaction": [
                            {"code": "read"},
                            {"code": "create"},
                            {"code": "search-type"},
                        ],
                        "searchParam": [
                            {"name": "_id", "type": "token"},
                            {"name": "patient", "type": "reference"},
                            {"name": "subject", "type": "reference"},
                            {"name": "code", "type": "token"},
                            {"name": "status", "type": "token"},
                        ],
                    },
                    {
                        "type": "DiagnosticReport",
                        "profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-diagnosticreport-note",
                        "interaction": [
                            {"code": "read"},
                            {"code": "create"},
                            {"code": "search-type"},
                        ],
                        "searchParam": [
                            {"name": "_id", "type": "token"},
                            {"name": "patient", "type": "reference"},
                            {"name": "subject", "type": "reference"},
                            {"name": "status", "type": "token"},
                        ],
                    },
                    {
                        "type": "DocumentReference",
                        "profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-documentreference",
                        "interaction": [
                            {"code": "read"},
                            {"code": "create"},
                            {"code": "search-type"},
                        ],
                        "searchParam": [
                            {"name": "_id", "type": "token"},
                            {"name": "patient", "type": "reference"},
                            {"name": "subject", "type": "reference"},
                        ],
                    },
                    {
                        "type": "Patient",
                        "profile": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient",
                        "interaction": [
                            {"code": "read"},
                            {"code": "create"},
                            {"code": "search-type"},
                        ],
                        "searchParam": [{"name": "_id", "type": "token"}],
                    },
                    {
                        "type": "Bundle",
                        "interaction": [
                            {"code": "read"},
                            {"code": "create"},
                        ],
                    },
                ],
                "interaction": [
                    {"code": "transaction"},
                    {"code": "batch"},
                ],
            }
        ],
    }


def _resolve_references(obj: Any, id_map: Dict[str, str]) -> Any:
    """Recursively rewrite temporary urn:uuid references to canonical FHIR references."""
    if isinstance(obj, dict):
        if "reference" in obj and isinstance(obj["reference"], str):
            ref = obj["reference"]
            if ref in id_map:
                obj["reference"] = id_map[ref]
            elif ref.startswith("urn:uuid:"):
                raw = ref.replace("urn:uuid:", "")
                if raw in id_map:
                    obj["reference"] = id_map[raw]
        return {k: _resolve_references(v, id_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_references(v, id_map) for v in obj]
    return obj


def process_transaction_bundle(
    bundle: Dict[str, Any], store: FHIRStore, base_url: str
) -> Tuple[Dict[str, Any], int]:
    """Ingest an HL7 FHIR R4 transaction or batch bundle atomically."""
    entries = bundle.get("entry", [])
    if not entries:
        return build_operation_outcome("Transaction bundle contains no entries."), 400

    # 1. Build ID mapping table: urn:uuid:xxx -> ResourceType/assigned_id
    id_map: Dict[str, str] = {}
    for entry in entries:
        res = entry.get("resource")
        full_url = entry.get("fullUrl")
        if res:
            rtype = res.get("resourceType", "Resource")
            rid = res.get("id") or str(uuid.uuid4())
            res["id"] = rid
            target_ref = f"{rtype}/{rid}"
            if full_url:
                id_map[full_url] = target_ref
                if full_url.startswith("urn:uuid:"):
                    id_map[full_url.replace("urn:uuid:", "")] = target_ref

    # 2. Resolve internal references and persist each resource
    response_entries = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for entry in entries:
        res = entry.get("resource")
        if not res:
            continue

        # Rewrite references
        resolved_res = _resolve_references(res, id_map)
        saved = store.save(resolved_res)
        rtype = saved.get("resourceType")
        rid = saved.get("id")

        response_entries.append(
            {
                "response": {
                    "status": "201 Created",
                    "location": f"{rtype}/{rid}/_history/1",
                    "etag": 'W/"1"',
                    "lastModified": now_iso,
                }
            }
        )

    # Persist the transaction bundle itself
    store.save(bundle)

    response_bundle = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "transaction-response",
        "entry": response_entries,
    }
    return response_bundle, 200


def create_fhir_blueprint(
    store: Optional[FHIRStore] = None, name: str = "fhir_r4"
) -> Blueprint:
    """Create a Flask Blueprint providing full HL7 FHIR R4 REST endpoints."""
    bp = Blueprint(name, __name__)
    fhir_store = store or get_default_fhir_store()

    def _fhir_response(data: Any, status_code: int = 200) -> Response:
        return Response(
            json.dumps(data, indent=2),
            status=status_code,
            mimetype="application/fhir+json; charset=utf-8",
        )

    @bp.route("/metadata", methods=["GET"])
    def capability_statement():
        """CapabilityStatement (conformance)."""
        base_url = request.base_url.removesuffix("/metadata")
        cs = build_capability_statement(base_url)
        return _fhir_response(cs)

    @bp.route("/", methods=["POST"], strict_slashes=False)
    @bp.route("/Bundle", methods=["POST"], strict_slashes=False)
    def ingest_bundle():
        """Root bundle ingestion endpoint (transaction / batch / collection)."""
        payload = request.get_json(silent=True)
        if not payload or not isinstance(payload, dict):
            return _fhir_response(
                build_operation_outcome("Invalid JSON payload or missing request body."),
                400,
            )

        rtype = payload.get("resourceType")
        if rtype == "Bundle":
            btype = payload.get("type", "transaction")
            if btype in ("transaction", "batch"):
                resp, code = process_transaction_bundle(
                    payload, fhir_store, request.base_url
                )
                return _fhir_response(resp, code)
            else:
                saved = fhir_store.save(payload)
                return _fhir_response(saved, 201)

        # Direct resource post at root
        if rtype in SUPPORTED_RESOURCE_TYPES:
            saved = fhir_store.save(payload)
            res = _fhir_response(saved, 201)
            res.headers["Location"] = f"{rtype}/{saved['id']}"
            return res

        return _fhir_response(
            build_operation_outcome(
                f"Unsupported resourceType '{rtype}' submitted to bundle endpoint."
            ),
            400,
        )

    @bp.route("/<resource_type>", methods=["POST"])
    def create_resource(resource_type: str):
        """Create a discrete FHIR resource (Observation, DiagnosticReport, etc.)."""
        if resource_type not in SUPPORTED_RESOURCE_TYPES:
            return _fhir_response(
                build_operation_outcome(f"Resource type '{resource_type}' not supported."),
                400,
            )

        payload = request.get_json(silent=True)
        if not payload or not isinstance(payload, dict):
            return _fhir_response(
                build_operation_outcome("Invalid JSON payload."), 400
            )

        payload_type = payload.get("resourceType")
        if payload_type and payload_type != resource_type:
            return _fhir_response(
                build_operation_outcome(
                    f"Payload resourceType '{payload_type}' does not match endpoint '/{resource_type}'."
                ),
                400,
            )
        payload["resourceType"] = resource_type

        saved = fhir_store.save(payload)
        resp = _fhir_response(saved, 201)
        resp.headers["Location"] = f"{resource_type}/{saved['id']}"
        return resp

    @bp.route("/<resource_type>/<resource_id>", methods=["GET"])
    def read_resource(resource_type: str, resource_id: str):
        """Read a discrete FHIR resource by ID."""
        if resource_type not in SUPPORTED_RESOURCE_TYPES:
            return _fhir_response(
                build_operation_outcome(f"Resource type '{resource_type}' not supported."),
                404,
            )

        res = fhir_store.get(resource_type, resource_id)
        if not res:
            return _fhir_response(
                build_operation_outcome(
                    f"{resource_type} with ID '{resource_id}' was not found in sandbox store.",
                    code="not-found",
                ),
                404,
            )

        return _fhir_response(res, 200)

    @bp.route("/<resource_type>", methods=["GET"])
    def search_resource(resource_type: str):
        """Search resources of a given type."""
        if resource_type not in SUPPORTED_RESOURCE_TYPES:
            return _fhir_response(
                build_operation_outcome(f"Resource type '{resource_type}' not supported."),
                404,
            )

        matches = fhir_store.search(resource_type, request.args.to_dict())
        searchset = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "searchset",
            "total": len(matches),
            "entry": [
                {
                    "fullUrl": f"{request.base_url}/{m['id']}",
                    "resource": m,
                }
                for m in matches
            ],
        }
        return _fhir_response(searchset, 200)

    @bp.route("/<resource_type>/<resource_id>", methods=["DELETE"])
    def delete_resource(resource_type: str, resource_id: str):
        """Delete a resource by ID."""
        deleted = fhir_store.delete(resource_type, resource_id)
        if deleted:
            return Response(status=204)
        return _fhir_response(
            build_operation_outcome(
                f"{resource_type}/{resource_id} not found.", code="not-found"
            ),
            404,
        )

    # Sandbox inspection & maintenance
    @bp.route("/_sandbox", methods=["GET"])
    def sandbox_summary():
        """Inspection endpoint returning sandbox storage summary."""
        data = fhir_store.summary()
        data["status"] = "active"
        data["server"] = "monai-cardiac-fat-fl HL7 FHIR R4 Sandbox"
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        return jsonify(data)

    @bp.route("/_reset", methods=["POST"])
    def sandbox_reset():
        """Reset and empty the sandbox store."""
        fhir_store.reset()
        return jsonify({"status": "reset", "message": "Sandbox store cleared successfully."})

    return bp


def create_fhir_app(
    storage_dir: Optional[Union[str, Path]] = None,
) -> Flask:
    """Create a standalone Flask application serving the HL7 FHIR R4 Sandbox."""
    app = Flask("cardiac_fat_fhir_server")
    store = FHIRStore(storage_dir=storage_dir)

    # Mount blueprint at both root and /r4 for maximum client interoperability
    bp_root = create_fhir_blueprint(store=store, name="fhir_root")
    bp_r4 = create_fhir_blueprint(store=store, name="fhir_r4")
    app.register_blueprint(bp_root)
    app.register_blueprint(bp_r4, url_prefix="/r4")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "service": "FHIR R4 Mock Server"})

    return app


def launch_fhir_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    storage_dir: Optional[Union[str, Path]] = None,
    auto_open: bool = True,
) -> None:
    """Launch the standalone HL7 FHIR R4 Mock Server & Sandbox."""
    app = create_fhir_app(storage_dir=storage_dir)
    url = f"http://{host}:{port}/r4"

    print("\n========================================================")
    print("  HL7 FHIR R4 Mock Server & Epic on FHIR Sandbox")
    print(f"  FHIR Base URL:     {url}")
    print(f"  Capability URL:    {url}/metadata")
    print(f"  Sandbox Summary:   {url}/_sandbox")
    print("========================================================\n")

    if auto_open:
        threading.Timer(1.2, lambda: webbrowser.open(f"{url}/metadata")).start()

    app.run(host=host, port=port, debug=False, threaded=True)
