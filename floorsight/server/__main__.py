"""Command line entry point.

    python -m floorsight.server --source sim|fake|live --sim-dir data/sim/demo_001 \
        --vision fake|real --agent stub|real|demo --port 8000 --speed 1 --loop

Live cameras:  --source live --live "cam1=0,cam2=rtsp://user:pass@host/stream"
Judge token:   set FLOORSIGHT_TOKEN, then open http://host:port/?token=<token>
Retention:     --preservation-hold on (default; or FLOORSIGHT_PRESERVATION_HOLD) keeps history/evidence;
               off deletes them after --retention-h
"""

from __future__ import annotations

import argparse
import logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m floorsight.server", description="Floorsight live floor map server")
    p.add_argument("--source", choices=["sim", "fake", "live", "realclip", "simlive"], default="fake",
                   help="sim: camera mp4s in --sim-dir; fake: simulator truth with blank frames; live: webcams/RTSP; "
                        "simlive: generated cameras with a controllable simulated forklift; "
                        "realclip: privacy-filtered real clips + floor records in --sim-dir (scripts/export_app_replay.py)")
    p.add_argument("--sim-dir", default="data/sim/demo_001", help="simulator output folder (site.json, cam*.mp4, truth)")
    p.add_argument("--site", default=None, help="site.json (default: <sim-dir>/site.json)")
    p.add_argument("--vision", choices=["fake", "real", "replay"], default="fake", help="replay: per-frame floor records of real clips")
    p.add_argument("--agent", choices=["stub", "real", "demo"], default="stub",
                   help="demo = scripted stand-in replaying the simulator's incidents (UI development only)")
    p.add_argument("--live", default=None, help='live cameras, e.g. "cam1=0,cam2=rtsp://host/stream"')
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier for recorded sources")
    p.add_argument("--loop", action="store_true", help="restart recorded footage (as a fresh session) when it ends")
    p.add_argument("--blur", choices=["auto", "off"], default="auto", help="face filter (auto: withhold frames if filtering fails)")
    p.add_argument("--setup-dir", default=None, help="folder with setup_empty_<camera>.png (default: --sim-dir for recorded footage)")
    p.add_argument("--runtime-dir", default="data/runtime", help="history and evidence (auto-deleted only with --preservation-hold off)")
    p.add_argument("--retention-h", type=float, default=24.0, help="with --preservation-hold off: delete history/evidence older than this")
    p.add_argument("--preservation-hold", choices=["on", "off"], default=None,
                   help="owner hold. on (default, or env FLOORSIGHT_PRESERVATION_HOLD=on): nothing is deleted automatically; "
                        "off: history and evidence older than --retention-h are deleted")
    p.add_argument("--log-level", default="info")
    p.add_argument("--opencv-threads", type=int, default=1, help="Threads per OpenCV kernel; cameras already run concurrently")
    p.add_argument("--no-drop", action="store_true", help="Offline evidence replay: process every frame even when slower than real time")
    p.add_argument("--interlock", choices=["on", "off"], default="on",
                   help="forklift interlock: slow/stop a forklift automatically the moment a warning is confirmed")
    p.add_argument("--sim-duration", type=float, default=40.0, help="--source simlive: seconds of the closed-loop scenario")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    import cv2
    cv2.setNumThreads(max(1, args.opencv_threads))

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from floorsight.server.app import create_app
    from floorsight.server.runtime import RuntimeConfig

    app = create_app(RuntimeConfig.from_args(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower(), ws_ping_interval=20, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
