"""Per-camera foreground detection: moving blobs (MOG2) and static regions (dual background)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Blob:
    """A foreground region in full-resolution pixel coordinates."""

    kind: str  # "moving" | "static"
    bbox: tuple[int, int, int, int]
    foot_px: tuple[float, float]
    area_px: float
    stationary_s: float
    score: float
    bottom_px: np.ndarray | None = None  # (N, 2) lowest pixel of each (sampled) column


class CameraDetector:
    """Runs at `scale` of the input resolution.

    moving: MOG2 foreground (shadows dropped) that also differs from the long-term background
            and has had frame-to-frame motion within the last `motion_hold_s`.
    static: differs from the long-term background, no motion, for longer than `static_min_s`.
            The long-term background is frozen under foreground, so a parked pallet stays visible;
            regions where the background, not the frame, holds the edges are "ghosts" of something
            that left and are healed into the background.
    """

    def __init__(self, width: int, height: int, fps: float, scale: float = 0.5):
        self.scale = scale
        self.fps = fps
        self.size = (int(round(width * scale)), int(round(height * scale)))
        self.min_area = 260 * scale * scale
        self.k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.motion_hold_s = 1.0
        self.static_min_s = 1.0
        self.diff_thresh = 28.0
        self.global_change = False
        self.plate: np.ndarray | None = None
        self.reset()

    def set_plate(self, image: np.ndarray) -> None:
        """Empty-floor setup image: becomes the long-term background (and MOG2 seed) from the next frame."""
        soft = cv2.GaussianBlur(image, (5, 5), 1.0)  # the plate is sharper than compressed video
        self.plate = cv2.resize(soft, self.size, interpolation=cv2.INTER_AREA)
        self.reset()

    def reset(self, drop_plate: bool = False) -> None:
        if drop_plate:
            self.plate = None
        # Shadows are removed by the chroma-aware test on the long-term difference, not by MOG2.
        self.mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
        self.mog.setNMixtures(3)
        self.mog.setBackgroundRatio(0.6)  # only the dominant mode is background: a forklift passing every lap stays foreground
        self.bg: np.ndarray | None = None
        self.prev_gray: list[np.ndarray] = []
        self.last_motion_t: np.ndarray | None = None
        self.static_count: np.ndarray | None = None
        self.moving_mask: np.ndarray | None = None
        self.static_mask: np.ndarray | None = None
        self.diff_mask: np.ndarray | None = None
        self.frames = 0

    def process(self, image: np.ndarray, t: float) -> list[Blob]:
        small = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if self.bg is None:
            self.bg = self._start_background(small)
            self.last_motion_t = np.full((h, w), -1e9, np.float32)
            self.static_count = np.zeros((h, w), np.uint16)
        self.frames += 1
        self.small = small

        fg = self.mog.apply(small)
        fg = cv2.threshold(fg, 200, 1, cv2.THRESH_BINARY)[1]
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k3)
        self.global_change = self.frames > 5 and cv2.countNonZero(fg) > 0.35 * h * w
        if self.global_change:
            return []

        # Frame differences 0.2 s and 1 s apart (the long lag catches uniform interiors), held for motion_hold_s.
        lag = int(round(self.fps))
        motion = None
        for k in (2, lag):
            if len(self.prev_gray) >= k:
                m = cv2.threshold(cv2.absdiff(gray, self.prev_gray[-k]), 14, 1, cv2.THRESH_BINARY)[1]
                motion = m if motion is None else cv2.bitwise_or(motion, m)
        now = np.zeros((h, w), np.uint8)
        if len(self.prev_gray) >= 2:
            now = cv2.dilate(cv2.threshold(cv2.absdiff(gray, self.prev_gray[-2]), 14, 1, cv2.THRESH_BINARY)[1], self.k9)
        if motion is not None:
            self.last_motion_t[cv2.dilate(motion, self.k9) > 0] = t
        self.prev_gray = (self.prev_gray + [gray])[-lag:]
        recent = (t - self.last_motion_t) < self.motion_hold_s
        self.recent = recent

        # Long-term background difference with shadow suppression (darker, same chroma).
        bg8 = cv2.convertScaleAbs(self.bg)
        c = cv2.split(cv2.absdiff(small, bg8))
        diff = cv2.max(cv2.max(c[0], c[1]), c[2])
        bgg = cv2.cvtColor(bg8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ratio = cv2.divide(gray.astype(np.float32), cv2.max(bgg, 1.0))
        dim = cv2.multiply(self.bg, cv2.merge([ratio, ratio, ratio]))
        c = cv2.split(cv2.absdiff(small.astype(np.float32), dim))
        chroma = cv2.max(cv2.max(c[0], c[1]), c[2])
        shadow = (ratio > 0.55) & (ratio < 0.92) & (chroma < 12.0)
        d = ((diff > self.diff_thresh) & ~shadow).astype(np.uint8)
        d = cv2.morphologyEx(d, cv2.MORPH_OPEN, self.k3)
        d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, self.k5)
        dmask = d > 0
        self.diff_mask = dmask
        # Background follows slow lighting changes only where nothing is in front of it. Shadow-like pixels are
        # never learned: dark grey trousers look like a shadow and would otherwise melt into the background.
        upd = ((cv2.bitwise_or(d, fg) == 0) & ~shadow).astype(np.uint8)
        cv2.accumulateWeighted(small, self.bg, 0.03, mask=upd)

        # Static: different from the long-term background with no motion, for static_min_s.
        # Motion passing in front holds the count rather than resetting it.
        cnt = self.static_count
        cnt += (dmask & ~recent).astype(np.uint16)
        cnt *= d
        np.minimum(cnt, 60000, out=cnt)
        min_frames = int(self.static_min_s * self.fps)
        settled = cnt >= min_frames
        static = cv2.morphologyEx(settled.astype(np.uint8), cv2.MORPH_CLOSE, self.k5)

        # Moving: regions of the long-term difference that MOG2 also flags and that are changing right now
        # (0.2 s apart). Judged per region, not per pixel, so the uniform inside of a large vehicle still counts,
        # while something that simply appeared (a set-down pallet) stops counting as moving at once.
        cand = cv2.morphologyEx(d, cv2.MORPH_CLOSE, self.k9)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        area = np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
        frac = np.bincount(lab.ravel(), weights=now.ravel(), minlength=n) / area
        ffrac = np.bincount(lab.ravel(), weights=fg.ravel(), minlength=n) / area
        keep = (frac >= 0.08) & (ffrac >= 0.15) & (stats[:, cv2.CC_STAT_AREA] >= self.min_area)
        keep[0] = False
        # Someone walking past a person standing still merges with them: carve the settled pixels out.
        moving8 = (keep[lab] & ~settled).astype(np.uint8)
        moving8 = cv2.morphologyEx(moving8, cv2.MORPH_OPEN, self.k5)
        self.moving_mask = moving8 > 0
        n, lab, stats, _ = cv2.connectedComponentsWithStats(moving8, connectivity=8)
        blobs = [
            self._blob("moving", *stats[i], lab[stats[i, 1] : stats[i, 1] + stats[i, 3], stats[i, 0] : stats[i, 0] + stats[i, 2]] == i, 0.0)
            for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area
        ]
        static_blobs = self._static_blobs(static, gray, small.astype(np.float32), min_frames)
        self.static_mask = self.static_count >= min_frames
        return blobs + static_blobs

    def _start_background(self, small: np.ndarray) -> np.ndarray:
        """First frame after a reset: the setup plate (gain-matched to today's light) or, failing that, the frame."""
        if self.plate is None:
            return small.astype(np.float32)
        plate = self.plate.astype(np.float32)
        cur = small.astype(np.float32)
        same = np.abs(cur - plate).max(axis=2) < 20
        if same.mean() > 0.2:
            gain = np.median(cur[same] / np.maximum(plate[same], 1.0), axis=0)
            plate = np.clip(plate * gain, 0, 255)
        seed = plate.astype(np.uint8)
        self.mog.apply(seed, learningRate=1.0)
        for _ in range(4):
            self.mog.apply(seed)
        return plate

    def heal(self, bbox: tuple[int, int, int, int]) -> None:
        """Take the current view as background inside a full-resolution box (where nothing moves)."""
        if self.bg is None or self.diff_mask is None:
            return
        x, y, w, h = (int(round(v * self.scale)) for v in bbox)
        H, W = self.diff_mask.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sl = (slice(y0, y1), slice(x0, x1))
        m = self.diff_mask[sl] & ~self.recent[sl]
        self.bg[sl][m] = self.small[sl][m]
        self.static_count[sl][m] = 0

    def _static_blobs(self, static: np.ndarray, gray: np.ndarray, smallf: np.ndarray, min_frames: int) -> list[Blob]:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(static, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area * 0.6:
                continue
            roi = lab[y : y + h, x : x + w] == i
            counts = self.static_count[y : y + h, x : x + w][roi]
            age = float(np.median(counts))
            if age < min_frames * 4 and self._is_ghost(roi, x, y, w, h, gray):
                sl = (slice(y, y + h), slice(x, x + w))
                grow = cv2.dilate(roi.astype(np.uint8), self.k9) > 0
                self.bg[sl][grow] = smallf[sl][grow]
                self.static_count[sl][grow] = 0
                continue
            out.append(self._blob("static", x, y, w, h, area, roi, age / self.fps))
        return out

    def object_edges(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """(frame, background) median edge strength along the outline of the static region inside a full-res box."""
        if self.static_mask is None:
            return 0.0, 0.0
        x, y, w, h = (int(round(v * self.scale)) for v in bbox)
        H, W = self.static_mask.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        roi = self.static_mask[y0:y1, x0:x1]
        if roi.sum() < 4:
            return 0.0, 0.0
        gray = cv2.cvtColor(self.small, cv2.COLOR_BGR2GRAY)
        self._is_ghost(roi, x0, y0, x1 - x0, y1 - y0, gray)
        return self.last_ghost_test

    def _is_ghost(self, roi: np.ndarray, x: int, y: int, w: int, h: int, gray: np.ndarray) -> bool:
        """Edges along the region outline that live in the background rather than the frame -> something left."""
        pad = 3
        y0, y1 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        m[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = roi
        band = (cv2.dilate(m, self.k5) - cv2.erode(m, self.k3)) > 0
        cur = gray[y0:y1, x0:x1].astype(np.float32)
        bg = cv2.cvtColor(self.bg[y0:y1, x0:x1].astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        def edge(img):
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            return float(np.median(cv2.magnitude(gx, gy)[band]))

        # Medians: a floor marker crossing the outline touches a few band pixels, an object's silhouette all of them.
        e_cur, e_bg = edge(cur), edge(bg)
        self.last_ghost_test = (e_cur, e_bg)
        return e_bg > 1.5 * e_cur and e_bg > 20.0  # plain floor texture alone sits around 12

    def _blob(self, kind: str, x: int, y: int, w: int, h: int, area: float, roi: np.ndarray, stationary_s: float) -> Blob:
        s = 1.0 / self.scale
        # Foot: centre of the lowest rows of the region (more robust than the bbox centre for odd shapes).
        rows = np.where(roi.any(axis=1))[0]
        low = roi[max(rows[-1] - 2, 0) : rows[-1] + 1]
        cols = np.where(low.any(axis=0))[0]
        fx = x + (cols.min() + cols.max() + 1) / 2.0 if len(cols) else x + w / 2.0
        fy = y + rows[-1] + 1.0
        fill = area / float(w * h)
        has = roi.any(axis=0)
        lowest = roi.shape[0] - 1 - np.argmax(roi[::-1], axis=0)
        cs = np.where(has)[0]
        if len(cs) > 24:
            cs = cs[np.linspace(0, len(cs) - 1, 24).round().astype(int)]
        bottom = np.column_stack([(x + cs + 0.5) * s, (y + lowest[cs] + 1.0) * s]) if len(cs) else None
        return Blob(
            kind=kind,
            bbox=(int(x * s), int(y * s), int(w * s), int(h * s)),
            foot_px=(fx * s, fy * s),
            area_px=float(area * s * s),
            stationary_s=stationary_s,
            score=float(min(1.0, 0.4 + fill)),
            bottom_px=bottom,
        )

    def fraction(self, kind: str, roi: tuple[int, int, int, int]) -> float:
        """Share of a full-resolution ROI covered by static (or moving/any foreground) evidence."""
        mask = self.static_mask if kind == "static" else self.diff_mask
        if mask is None:
            return 0.0
        x, y, w, h = (int(round(v * self.scale)) for v in roi)
        H, W = mask.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(mask[y0:y1, x0:x1].mean())
