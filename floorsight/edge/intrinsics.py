"""Lens calibration for real cameras with a ChArUco board (OpenCV 5 objdetect API).

1. Print the board at 100 % scale and glue it to something flat:
       python -m floorsight.edge.intrinsics board --out out/charuco_A4.png --paper A4
   Measure one printed square with a ruler and pass that value as --square-mm.
2. Record 15-30 stills (or a slow video) of the board at different angles,
   distances and positions, covering the corners of the image.
3. Calibrate and write K + dist into the site plan's camera entry:
       python -m floorsight.edge.intrinsics calibrate --images "shots/cam1/*.jpg" \
           --site site.json --camera cam1
       python -m floorsight.edge.intrinsics calibrate --video cam1_board.mp4 --every 10 ...

RMS re-projection error is printed; under ~0.5 px is good, over 1 px means
blurry shots, a non-flat board or a wrong --square-mm.
"""

from __future__ import annotations

import argparse
import glob
import json

import cv2
import numpy as np

from floorsight.contracts import CameraInfo, SitePlan
from floorsight.edge.markers import PAPER_MM, _set_png_dpi, get_dictionary, mm_to_px

# Default board: 7 x 5 squares fits A4 landscape with 35 mm squares.
SQUARES_X, SQUARES_Y = 7, 5
SQUARE_MM, MARKER_MM = 35.0, 26.0
BOARD_DICT = "DICT_4X4_50"
MIN_CORNERS = 6


def make_board(
    squares_x: int = SQUARES_X,
    squares_y: int = SQUARES_Y,
    square_mm: float = SQUARE_MM,
    marker_mm: float = MARKER_MM,
    dict_name: str = BOARD_DICT,
):
    """Board object in metres (so tvecs come out in metres)."""
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm / 1000.0, marker_mm / 1000.0, get_dictionary(dict_name))


def board_image(board, square_mm: float = SQUARE_MM, paper: str = "A4", dpi: int = 300) -> np.ndarray:
    """Full page (landscape) with the board at exact physical scale and a label."""
    pw, ph = PAPER_MM[paper]
    page_w, page_h = mm_to_px(ph, dpi), mm_to_px(pw, dpi)  # landscape
    sx, sy = board.getChessboardSize()
    bw, bh = mm_to_px(sx * square_mm, dpi), mm_to_px(sy * square_mm, dpi)
    if bw > page_w - mm_to_px(20, dpi) or bh > page_h - mm_to_px(30, dpi):
        raise ValueError("board does not fit the page; use smaller squares or A3")
    img = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    page = np.full((page_h, page_w), 255, np.uint8)
    x0, y0 = (page_w - bw) // 2, (page_h - bh) // 2
    page[y0 : y0 + bh, x0 : x0 + bw] = img
    label = (
        f"Floorsight ChArUco {sx}x{sy}  square {square_mm:g} mm  marker {board.getMarkerLength() * 1000:g} mm  "
        f"{BOARD_DICT}  - print at 100%, measure a square, keep flat"
    )
    cv2.putText(page, label, (mm_to_px(10, dpi), mm_to_px(8, dpi)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 300 * 0.9, 0, 2, cv2.LINE_AA)
    return page


def detect_views(images, board, min_corners: int = MIN_CORNERS):
    """Yield (object_points, image_points) for every image with enough ChArUco corners."""
    detector = cv2.aruco.CharucoDetector(board)
    for img in images:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners:
            yield None
            continue
        obj, imgp = board.matchImagePoints(ch_corners, ch_ids)
        if obj is None or len(obj) < min_corners:
            yield None
            continue
        yield obj.reshape(-1, 3).astype(np.float32), imgp.reshape(-1, 2).astype(np.float32)


def calibrate(images: list[np.ndarray], board, flags: int = 0) -> dict:
    """Returns {"K", "dist", "rms_px", "views_used", "views_total", "width", "height"}."""
    if not images:
        raise ValueError("no images")
    h, w = images[0].shape[:2]
    objs, imgs = [], []
    for v in detect_views(images, board):
        if v is not None:
            objs.append(v[0])
            imgs.append(v[1])
    if len(objs) < 4:
        raise RuntimeError(f"only {len(objs)} usable board views (need >= 4, ideally 15+)")
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, (w, h), None, None, flags=flags)
    return {
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "rms_px": float(rms),
        "views_used": len(objs),
        "views_total": len(images),
        "width": w,
        "height": h,
    }


def load_images(pattern: str | None = None, video: str | None = None, every: int = 10, limit: int = 60) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    if pattern:
        for p in sorted(glob.glob(pattern)):
            img = cv2.imread(p)
            if img is not None:
                out.append(img)
    if video:
        cap = cv2.VideoCapture(video)
        i = 0
        while len(out) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                out.append(frame)
            i += 1
        cap.release()
    return out


def write_camera_intrinsics(site_path: str, camera_id: str, result: dict, fps: float | None = None, out_path: str | None = None) -> SitePlan:
    """Set K/dist/width/height on the camera (added if it is not in the plan yet)."""
    with open(site_path, encoding="utf-8") as f:
        site = SitePlan.model_validate(json.load(f))
    cams = {c.camera_id: c for c in site.cameras}
    old = cams.get(camera_id)
    cams[camera_id] = CameraInfo(
        camera_id=camera_id,
        width=result["width"],
        height=result["height"],
        fps=fps if fps is not None else (old.fps if old else 10.0),
        K=result["K"],
        dist=result["dist"],
        nominal_position=old.nominal_position if old else None,
    )
    data = site.model_dump(mode="json")
    data["cameras"] = [cams[k].model_dump(mode="json") for k in sorted(cams)]
    site = SitePlan.model_validate(data)
    with open(out_path or site_path, "w", encoding="utf-8") as f:
        json.dump(site.model_dump(mode="json"), f, indent=2)
    return site


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("board", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--squares-x", type=int, default=SQUARES_X)
        p.add_argument("--squares-y", type=int, default=SQUARES_Y)
        p.add_argument("--square-mm", type=float, default=SQUARE_MM)
        p.add_argument("--marker-mm", type=float, default=MARKER_MM)
    b = sub.choices["board"]
    b.add_argument("--paper", choices=list(PAPER_MM), default="A4")
    b.add_argument("--dpi", type=int, default=300)
    b.add_argument("--out", required=True)
    c = sub.choices["calibrate"]
    c.add_argument("--images")
    c.add_argument("--video")
    c.add_argument("--every", type=int, default=10)
    c.add_argument("--site")
    c.add_argument("--camera")
    c.add_argument("--fps", type=float)
    c.add_argument("--out", help="site plan output path (default: overwrite --site)")
    a = ap.parse_args(argv)
    board = make_board(a.squares_x, a.squares_y, a.square_mm, a.marker_mm)
    if a.cmd == "board":
        page = board_image(board, a.square_mm, a.paper, a.dpi)
        cv2.imwrite(a.out, page)
        _set_png_dpi(a.out, a.dpi)
        print(a.out)
        return
    images = load_images(a.images, a.video, a.every)
    res = calibrate(images, board)
    print(f"RMS re-projection error: {res['rms_px']:.3f} px  ({res['views_used']}/{res['views_total']} views used)")
    print("K =", np.array(res["K"]).round(2).tolist())
    print("dist =", np.round(res["dist"], 5).tolist())
    if a.site and a.camera:
        write_camera_intrinsics(a.site, a.camera, res, a.fps, a.out)
        print(f"written to camera {a.camera} in {a.out or a.site}")


if __name__ == "__main__":
    main()
