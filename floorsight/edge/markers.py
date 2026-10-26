"""Printable ArUco floor markers and the site-plan helper that records where they lie.

Print, then measure:

    python -m floorsight.edge.markers sheet --ids 0-11 --size-mm 200 --paper A4 --out out/markers
    # ... lay the markers flat on the floor, measure each centre from the plan origin ...
    python -m floorsight.edge.markers place --site site.json --measured measured.csv --out site.json

`--size-mm` is the edge of the black square (the part the detector measures).
Each marker is printed with a white quiet zone, a dashed cut line, and a label
with the id, dictionary, size and which edge points "up the plan" (+y).
Markers larger than the printable area are tiled across several sheets with
overlap marks so they can be taped back together.

measured.csv columns: marker_id,x_m,y_m[,size_m][,rotation_deg]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np

from floorsight.contracts import FloorMarker, SitePlan

DICT_NAME = "DICT_4X4_50"
PAPER_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}
MARGIN_MM = 10.0  # unprintable border most printers need
LABEL_MM = 14.0  # height reserved under each marker for its label
QUIET_MODULES = 1.0  # white border around the black square, in marker modules (4x4 + 2 border = 6 modules)
QUIET_MAX_MM = 10.0  # ...but never more than this, so a 150 mm marker still fits one A4 page


def quiet_mm(size_mm: float) -> float:
    return min(size_mm / 6.0 * QUIET_MODULES, QUIET_MAX_MM)


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def get_dictionary(name: str = DICT_NAME):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def marker_image(marker_id: int, size_px: int, name: str = DICT_NAME) -> np.ndarray:
    """Black square of exactly size_px, 1-module black border included (OpenCV convention)."""
    return cv2.aruco.generateImageMarker(get_dictionary(name), marker_id, size_px, borderBits=1)


def _dashed_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, dash: int, color=(150, 150, 150), thick: int = 2):
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = int(math.hypot(bx - ax, by - ay))
        for s in range(0, length, 2 * dash):
            e = min(s + dash, length)
            p = (int(ax + (bx - ax) * s / length), int(ay + (by - ay) * s / length))
            q = (int(ax + (bx - ax) * e / length), int(ay + (by - ay) * e / length))
            cv2.line(img, p, q, color, thick)


def _text(img, txt, org, scale, thick=2, color=(0, 0, 0)):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _label(marker_id: int, size_mm: float, name: str) -> str:
    return f"id {marker_id}  {name}  {size_mm:g} mm black square   (top edge of marker faces +y on the plan)"


def marker_sheets(
    ids: list[int],
    size_mm: float,
    paper: str = "A4",
    dpi: int = 300,
    name: str = DICT_NAME,
) -> list[np.ndarray]:
    """Render printable pages (BGR, white background) at `dpi`. Prints at 100 % scale."""
    if paper not in PAPER_MM:
        raise ValueError(f"paper must be one of {list(PAPER_MM)}")
    pw_mm, ph_mm = PAPER_MM[paper]
    page_w, page_h = mm_to_px(pw_mm, dpi), mm_to_px(ph_mm, dpi)
    margin = mm_to_px(MARGIN_MM, dpi)
    label_h = mm_to_px(LABEL_MM, dpi)
    quiet = mm_to_px(quiet_mm(size_mm), dpi)
    side = mm_to_px(size_mm, dpi)
    cell_w = side + 2 * quiet
    cell_h = side + 2 * quiet + label_h
    usable_w, usable_h = page_w - 2 * margin, page_h - 2 * margin
    text_scale = dpi / 300 * 0.9

    def new_page():
        return np.full((page_h, page_w, 3), 255, np.uint8)

    if cell_w > usable_w or cell_h > usable_h:
        return _tiled_sheets(ids, size_mm, paper, dpi, name)

    cols = max(1, usable_w // cell_w)
    rows = max(1, usable_h // cell_h)
    pages = []
    for start in range(0, len(ids), cols * rows):
        page = new_page()
        for k, mid in enumerate(ids[start : start + cols * rows]):
            r, c = divmod(k, cols)
            x0 = margin + c * cell_w
            y0 = margin + r * cell_h
            m = cv2.cvtColor(marker_image(mid, side, name), cv2.COLOR_GRAY2BGR)
            page[y0 + quiet : y0 + quiet + side, x0 + quiet : x0 + quiet + side] = m
            _dashed_rect(page, x0, y0, x0 + cell_w - 1, y0 + side + 2 * quiet - 1, dash=mm_to_px(3, dpi))
            _text(page, _label(mid, size_mm, name), (x0 + 4, y0 + side + 2 * quiet + label_h // 2), text_scale * min(1.0, cell_w / mm_to_px(150, dpi)))
        _text(page, f"Floorsight floor markers - print at 100% scale ({paper}, {dpi} dpi). Check: black square = {size_mm:g} mm.",
              (margin, margin // 2 + 10), text_scale * 0.8)
        pages.append(page)
    return pages


def _tiled_sheets(ids, size_mm, paper, dpi, name) -> list[np.ndarray]:
    """Marker bigger than a page: split it into page-sized tiles with overlap marks."""
    pw_mm, ph_mm = PAPER_MM[paper]
    page_w, page_h = mm_to_px(pw_mm, dpi), mm_to_px(ph_mm, dpi)
    margin = mm_to_px(MARGIN_MM, dpi)
    head = mm_to_px(LABEL_MM, dpi)
    overlap = mm_to_px(10, dpi)
    tw, th = page_w - 2 * margin, page_h - 2 * margin - head
    side = mm_to_px(size_mm, dpi)
    quiet = mm_to_px(quiet_mm(size_mm), dpi)
    full_side = side + 2 * quiet
    step_x, step_y = tw - overlap, th - overlap
    nx = max(1, math.ceil((full_side - overlap) / step_x))
    ny = max(1, math.ceil((full_side - overlap) / step_y))
    pages = []
    for mid in ids:
        full = np.full((full_side, full_side, 3), 255, np.uint8)
        full[quiet : quiet + side, quiet : quiet + side] = cv2.cvtColor(marker_image(mid, side, name), cv2.COLOR_GRAY2BGR)
        for ty in range(ny):
            for tx in range(nx):
                x0, y0 = tx * step_x, ty * step_y
                tile = full[y0 : y0 + th, x0 : x0 + tw]
                page = np.full((page_h, page_w, 3), 255, np.uint8)
                oy = margin + head
                page[oy : oy + tile.shape[0], margin : margin + tile.shape[1]] = tile
                # overlap guides: the strip right/below this line is printed on the next tile too
                if tx < nx - 1:
                    cv2.line(page, (margin + tw - overlap, oy), (margin + tw - overlap, oy + tile.shape[0]), (0, 0, 255), 1)
                if ty < ny - 1:
                    cv2.line(page, (margin, oy + th - overlap), (margin + tile.shape[1], oy + th - overlap), (0, 0, 255), 1)
                _text(page, f"{_label(mid, size_mm, name)}   tile row {ty + 1}/{ny} col {tx + 1}/{nx} (red line = 10 mm overlap)",
                      (margin, margin + head // 2), dpi / 300 * 0.7)
                pages.append(page)
    return pages


def write_sheets(pages: list[np.ndarray], out_dir: str, dpi: int = 300, prefix: str = "markers") -> list[str]:
    """PNG files with the dpi recorded, so 'print at actual size' gives the right scale."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, p in enumerate(pages):
        path = os.path.join(out_dir, f"{prefix}_{i + 1:02d}.png")
        cv2.imwrite(path, p)
        _set_png_dpi(path, dpi)
        paths.append(path)
    return paths


def _set_png_dpi(path: str, dpi: int) -> None:
    """Insert a pHYs chunk (pixels per metre) after IHDR so printers know the physical size."""
    import struct
    import zlib

    with open(path, "rb") as f:
        data = f.read()
    ppm = int(round(dpi / 0.0254))
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    ihdr_end = 8 + 8 + 13 + 4
    if b"pHYs" in data[:200]:
        return
    with open(path, "wb") as f:
        f.write(data[:ihdr_end] + chunk + data[ihdr_end:])


# --------------------------------------------------------------------------- #
# Site plan update
# --------------------------------------------------------------------------- #


def read_measured(path: str) -> list[dict]:
    """CSV (marker_id,x_m,y_m[,size_m][,rotation_deg]) or JSON list of the same keys."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): v.strip() for k, v in r.items() if v is not None and v.strip() != ""})
    return rows


def apply_measured_markers(site: SitePlan, measured: list[dict], default_size_m: float | None = None) -> SitePlan:
    """Update marker centres by id (or add new markers). Positions in metres on the plan."""
    by_id = {m.marker_id: m for m in site.markers}
    for row in measured:
        mid = int(row["marker_id"])
        x, y = float(row["x_m"]), float(row["y_m"])
        if not (0 <= x <= site.width_m and 0 <= y <= site.depth_m):
            raise ValueError(f"marker {mid} at ({x}, {y}) is outside the {site.width_m} x {site.depth_m} m plan")
        old = by_id.get(mid)
        size = row.get("size_m") or (old.size_m if old else default_size_m)
        if size is None:
            raise ValueError(f"marker {mid}: size_m missing and no default given")
        by_id[mid] = FloorMarker(
            marker_id=mid,
            dictionary=(old.dictionary if old else DICT_NAME),
            center=(x, y),
            size_m=float(size),
            rotation_deg=float(row.get("rotation_deg", old.rotation_deg if old else 0.0)),
        )
    data = site.model_dump()
    data["markers"] = [by_id[k].model_dump() for k in sorted(by_id)]
    return SitePlan.model_validate(data)


def update_site_file(site_path: str, measured_path: str, out_path: str | None = None, default_size_m: float | None = None) -> SitePlan:
    with open(site_path, encoding="utf-8") as f:
        site = SitePlan.model_validate(json.load(f))
    site = apply_measured_markers(site, read_measured(measured_path), default_size_m)
    with open(out_path or site_path, "w", encoding="utf-8") as f:
        json.dump(site.model_dump(mode="json"), f, indent=2)
    return site


def parse_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    if any(not 0 <= i < 50 for i in ids):
        raise ValueError("DICT_4X4_50 ids are 0..49")
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet", help="render printable marker sheets")
    s.add_argument("--ids", required=True, help="e.g. 0-11 or 0,3,5")
    s.add_argument("--size-mm", type=float, required=True)
    s.add_argument("--paper", choices=list(PAPER_MM), default="A4")
    s.add_argument("--dpi", type=int, default=300)
    s.add_argument("--out", required=True)
    p = sub.add_parser("place", help="write measured marker centres into a site plan")
    p.add_argument("--site", required=True)
    p.add_argument("--measured", required=True)
    p.add_argument("--default-size-m", type=float)
    p.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "sheet":
        pages = marker_sheets(parse_ids(a.ids), a.size_mm, a.paper, a.dpi)
        for path in write_sheets(pages, a.out, a.dpi):
            print(path)
    else:
        site = update_site_file(a.site, a.measured, a.out, a.default_size_m)
        print(f"{len(site.markers)} markers written to {a.out or a.site}")


if __name__ == "__main__":
    main()
