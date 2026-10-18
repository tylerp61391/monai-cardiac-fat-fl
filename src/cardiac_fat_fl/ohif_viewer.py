"""OHIF Viewer Launcher and Local Medical Web Server.

Serves participant CT and DICOM-SEG results with CORS headers, opens the
local interactive viewer, and provides 1-click integration with official OHIF Viewer.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

from cardiac_fat_fl.ohif_exporter import export_sample_to_ohif


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with permissive CORS headers for OHIF Viewer."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type, Origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200, "OK")
        self.end_headers()


def launch_viewer(
    sample_dir: Path | str,
    port: int = 8000,
    open_browser: bool = True,
    export_only: bool = False,
) -> Path:
    """Exports and launches the local OHIF-compatible viewer for a participant sample.

    Args:
        sample_dir: Path to the participant folder containing CT and eat_pvat.seg.nrrd.
        port: Port for the local web server. Defaults to 8000.
        open_browser: Whether to open the default browser automatically.
        export_only: If True, exports files and skips launching the web server.

    Returns:
        Path to the generated ohif_export directory.
    """
    sample_path = Path(sample_dir).resolve()
    print(f"Preparing OHIF DICOM study & segmentation package for {sample_path.name}...")
    export_dir = export_sample_to_ohif(sample_path)
    print(f"OHIF study exported to: {export_dir}")

    if export_only:
        return export_dir

    handler = functools.partial(CORSHTTPRequestHandler, directory=str(export_dir))
    
    # Try binding port, with fallback if occupied
    server: Optional[socketserver.TCPServer] = None
    target_port = port
    for attempt in range(5):
        try:
            server = socketserver.TCPServer(("127.0.0.1", target_port), handler)
            break
        except OSError:
            target_port += 1

    if server is None:
        print(f"Could not bind to port {port}-{target_port}. Viewer exported at: {export_dir}")
        return export_dir

    local_url = f"http://127.0.0.1:{target_port}/"
    ohif_local_url = "https://viewer.ohif.org/local"
    
    print("\n" + "=" * 70)
    print("  CARDIAC FAT OHIF VIEWER SERVER ACTIVE")
    print("=" * 70)
    print(f"  Local Interactive Viewer:  {local_url}")
    print(f"  Official OHIF Local App:   {ohif_local_url}")
    print(f"  Export Folder:             {export_dir}")
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

