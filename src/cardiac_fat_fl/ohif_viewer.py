"""OHIF Viewer Launcher and Local Medical Web Server.

Serves participant CT and DICOM-SEG results with CORS headers, opens the
local interactive viewer, and provides dynamic switching across cohort participants.
"""

from __future__ import annotations

import functools
import http.server
import os
from pathlib import Path
import shutil
import socketserver
import urllib.parse
import webbrowser
from typing import List, Optional

from cardiac_fat_fl.ohif_exporter import export_sample_to_ohif


class CohortHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Permissive CORS and dynamic cohort routing request handler for OHIF viewer."""

    cohort_dir: Path = Path(".")
    current_pid: str = "0"
    all_pids: List[str] = []

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200, "OK")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        clean_path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Handle query params like /?patient=13 or /?p=13
        if "patient" in query or "p" in query:
            target_pid = (query.get("patient") or query.get("p"))[0]
            self.send_response(302)
            self.send_header("Location", f"/patient/{target_pid}/")
            self.end_headers()
            return

        # 2. Handle root / -> redirect to current active patient
        if clean_path in ("", "/"):
            self.send_response(302)
            self.send_header("Location", f"/patient/{self.current_pid}/")
            self.end_headers()
            return

        # 3. Handle /patient/<pid>/ or /patient/<pid>/<subpath>
        if clean_path.startswith("/patient/"):
            parts = clean_path.strip("/").split("/")
            if len(parts) >= 2:
                pid = parts[1]
                subpath = "/".join(parts[2:]) if len(parts) > 2 else ""

                patient_folder = self.cohort_dir / pid
                if not patient_folder.is_dir():
                    self.send_error(404, f"Patient {pid} not found in cohort directory.")
                    return

                export_dir = patient_folder / "ohif_export"
                needs_export = False
                if not (export_dir / "index.html").exists():
                    needs_export = True
                else:
                    try:
                        idx_txt = (export_dir / "index.html").read_text(encoding="utf-8", errors="ignore")
                        if (
                            "v3.13 Next" not in idx_txt
                            or "viewportsGrid" not in idx_txt
                            or "Coronal CT" not in idx_txt
                            or "let isSpinning = true;" in idx_txt
                        ):
                            needs_export = True
                    except Exception:
                        needs_export = True

                if needs_export:
                    print(f"Upgrading/Exporting OHIF assets on-the-fly for Patient {pid}...")
                    try:
                        export_sample_to_ohif(
                            sample_dir=patient_folder,
                            available_participants=self.all_pids,
                        )
                    except Exception as e:
                        print(f"Error exporting patient {pid}: {e}")
                        self.send_error(500, f"Error generating OHIF package for patient {pid}: {e}")
                        return

                if not subpath or subpath == "index.html":
                    target_file = export_dir / "index.html"
                else:
                    target_file = export_dir / subpath

                if target_file.is_file():
                    self._serve_file(target_file)
                    return
                elif target_file.is_dir() and (target_file / "index.html").is_file():
                    self._serve_file(target_file / "index.html")
                    return
                else:
                    self.send_error(404, f"File {subpath} not found for patient {pid}.")
                    return

        # 4. Fallback for relative requests from single export root (e.g. /slices/ct_028.png)
        fallback_file = self.cohort_dir / self.current_pid / "ohif_export" / clean_path.lstrip("/")
        if fallback_file.is_file():
            self._serve_file(fallback_file)
            return

        super().do_GET()

    def _serve_file(self, path: Path) -> None:
        """Serve a specific file from disk with proper headers and MIME types."""
        ctype = self.guess_type(str(path))
        if path.suffix == ".dcm":
            ctype = "application/dicom"
        try:
            with open(path, "rb") as f:
                fs = path.stat()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(fs.st_size))
                self.send_header("Last-Modified", self.date_time_string(int(fs.st_mtime)))
                self.end_headers()
                shutil.copyfileobj(f, self.wfile)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")


# Alias for backward compatibility
CORSHTTPRequestHandler = CohortHTTPRequestHandler


def launch_viewer(
    sample_dir: Path | str,
    port: int = 8000,
    open_browser: bool = True,
    export_only: bool = False,
) -> Path:
    """Exports and launches the local OHIF-compatible viewer with cohort navigation.

    Args:
        sample_dir: Path to the participant folder containing CT and eat_pvat.seg.nrrd.
        port: Port for the local web server. Defaults to 8000.
        open_browser: Whether to open the default browser automatically.
        export_only: If True, exports files and skips launching the web server.

    Returns:
        Path to the generated ohif_export directory.
    """
    sample_path = Path(sample_dir).resolve()

    # Determine cohort directory and initial active participant
    if (sample_path / "0").is_dir() or (sample_path / "1").is_dir():
        cohort_dir = sample_path
        pids = [d.name for d in cohort_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        pids.sort(key=lambda x: int(x) if x.isdigit() else str(x))
        sample_path = cohort_dir / (pids[0] if pids else "0")
    else:
        cohort_dir = sample_path.parent

    # Discover all cohort participants fast
    all_pids = [
        d.name for d in cohort_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "ohif_export"
        and ((d / "eat_pvat.seg.nrrd").is_file() or (d / "ohif_export").is_dir() or any(f.name.endswith(".nrrd") for f in d.iterdir()))
    ]
    all_pids.sort(key=lambda x: int(x) if x.isdigit() else str(x))
    if not all_pids:
        all_pids = [sample_path.name]

    export_dir = sample_path / "ohif_export"
    idx_file = export_dir / "index.html"
    if not idx_file.exists() or "v3.13 Next" not in idx_file.read_text(encoding="utf-8", errors="ignore"):
        print(f"Preparing OHIF DICOM study & segmentation package for {sample_path.name}...")
        export_dir = export_sample_to_ohif(sample_path, available_participants=all_pids)
    print(f"OHIF study ready at: {export_dir}")

    if export_only:
        return export_dir

    class CustomCohortHandler(CohortHTTPRequestHandler):
        pass

    CustomCohortHandler.cohort_dir = cohort_dir
    CustomCohortHandler.current_pid = sample_path.name
    CustomCohortHandler.all_pids = all_pids

    # Try binding port, with fallback if occupied
    server: Optional[socketserver.TCPServer] = None
    target_port = port
    for attempt in range(5):
        try:
            server = socketserver.TCPServer(("127.0.0.1", target_port), CustomCohortHandler)
            break
        except OSError:
            target_port += 1

    if server is None:
        print(f"Could not bind to port {port}-{target_port}. Viewer exported at: {export_dir}")
        return export_dir

    local_url = f"http://127.0.0.1:{target_port}/patient/{sample_path.name}/"
    ohif_local_url = "https://viewer.ohif.org/local"

    print("\n" + "=" * 70)
    print("  CARDIAC FAT OHIF VIEWER SERVER ACTIVE (COHORT MODE)")
    print("=" * 70)
    print(f"  Local Interactive Viewer:  {local_url}")
    print(f"  Official OHIF Local App:   {ohif_local_url}")
    print(f"  Active Participant:        {sample_path.name}")
    print(f"  Total Cohort Cases:        {len(all_pids)}")
    print("=" * 70)
    print("  To view in Official OHIF:")
    print("    1. Open https://viewer.ohif.org/local")
    print(f"    2. Drag and drop the '{export_dir.name}' folder into the browser.")
    print("=" * 70 + "\n")

    if open_browser:
        webbrowser.open(local_url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer server.")
    finally:
        server.server_close()

    return export_dir
