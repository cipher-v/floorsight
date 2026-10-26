"""Floor calibration from ArUco markers: image<->floor homography, camera pose, drift error."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from floorsight.contracts import CameraInfo, SitePlan
from floorsight.edge.markers import DICT_NAME, get_dictionary

# ArUco returns corners TL, TR, BR, BL; on the floor these sit at (dx, dy) * size/2 from the
# centre (verified on rendered footage: < 2 px against the true projection).
CORNER_OFFSETS = np.array([(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])
# Undistorting a pixel is iterative; OpenCV's default 5 rounds leave ~0.05 px (about a centimetre far away) at the corners.
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)


def undistort_px(px: np.ndarray, K: np.ndarray, dist: np.ndarray | None) -> np.ndarray:
    """Image pixels (N, 2) -> where an ideal pinhole camera with the same K sees them.

    The image<->floor homographies hold for ideal pixels only. Without lens distortion the pixels come back
    unchanged, so floor coordinates stay bit-identical to the plain homography.
    """
    px = np.asarray(px, np.float64).reshape(-1, 2)
    if dist is None or not np.any(dist) or not len(px):
        return px
    pts = px.reshape(-1, 1, 2)
    if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4: the variant with stopping criteria has its own name
        return cv2.undistortPointsIter(pts, K, dist, None, K, UNDISTORT_CRITERIA).reshape(-1, 2)
    try:  # OpenCV 5 takes the criteria on undistortPoints itself
        return cv2.undistortPoints(pts, K, dist, R=None, P=K, criteria=UNDISTORT_CRITERIA).reshape(-1, 2)
    except (cv2.error, TypeError):
        return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def site_dictionary(site: SitePlan) -> str:
    """The one ArUco dictionary the plan's floor markers use (the default when there are none)."""
    names = {m.dictionary for m in site.markers} or {DICT_NAME}
    if len(names) > 1:
        raise ValueError(f"floor markers mix ArUco dictionaries {sorted(names)}; one detector reads one dictionary")
    return names.pop()


def marker_floor_corners(site: SitePlan) -> dict[int, np.ndarray]:
    """Floor (x, y) of each marker's four corners, in ArUco corner order."""
    out = {}
    for m in site.markers:
        a = np.radians(m.rotation_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        out[m.marker_id] = np.asarray(m.center, float) + (CORNER_OFFSETS * m.size_m / 2) @ rot.T
    return out


class MarkerDetector:
    """Fast detector for routine drift checks, thorough (upscaled, tuned) detector for calibration."""

    def __init__(self, site: SitePlan):
        dictionary = get_dictionary(site_dictionary(site))  # ValueError on an unknown or mixed dictionary
        fast = cv2.aruco.DetectorParameters()
        fast.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        fine = cv2.aruco.DetectorParameters()
        fine.minMarkerPerimeterRate = 0.01
        fine.adaptiveThreshWinSizeMin = 3
        fine.adaptiveThreshWinSizeMax = 53
        fine.adaptiveThreshWinSizeStep = 6
        fine.polygonalApproxAccuracyRate = 0.06
        fine.minCornerDistanceRate = 0.02
        fine.perspectiveRemovePixelPerCell = 8
        fine.perspectiveRemoveIgnoredMarginPerCell = 0.2
        fine.errorCorrectionRate = 0.6
        fine.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._fast = cv2.aruco.ArucoDetector(dictionary, fast)
        self._fine = cv2.aruco.ArucoDetector(dictionary, fine)
        self.known = set(m.marker_id for m in site.markers)

    def detect(self, image: np.ndarray, fine: bool = False) -> dict[int, np.ndarray]:
        """{marker_id: 4x2 image corners} for markers listed in the site plan."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out = self._run(self._fast, gray, 1.0)
        if fine:
            up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            out.update(self._run(self._fine, up, 2.0))
        return out

    def _run(self, det, gray: np.ndarray, scale: float) -> dict[int, np.ndarray]:
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c[0].astype(np.float64) / scale for c, i in zip(corners, ids.ravel()) if int(i) in self.known}


@dataclass
class Calibration:
    """Pose and floor mapping of one camera, fitted from marker corners."""

    K: np.ndarray
    R: np.ndarray  # world -> camera rotation
    tvec: np.ndarray
    H_floor2img: np.ndarray
    H_img2floor: np.ndarray
    error_m: float
    markers_used: int
    homography_error_m: float
    dist: np.ndarray | None = None  # lens distortion (OpenCV order) the pose was fitted with; None/zero = ideal pinhole

    @property
    def position(self) -> np.ndarray:
        return (-self.R.T @ self.tvec).ravel()

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.column_stack([self.R, self.tvec])

    @property
    def distorted(self) -> bool:
        return self.dist is not None and bool(np.any(self.dist))

    def to_floor(self, px: np.ndarray) -> np.ndarray:
        """Image pixels (N, 2) -> floor metres (N, 2)."""
        px = undistort_px(px, self.K, self.dist).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(px, self.H_img2floor).reshape(-1, 2)

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N, 3) -> pixels (N, 2) and camera depth (N,)."""
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        cam = xyz @ self.R.T + self.tvec.ravel()
        z = cam[:, 2]
        if self.distorted and len(cam):
            # The same lens model the pose was fitted with, so drawn zones and boxes land where the camera sees them.
            ideal = np.column_stack([cam[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None], np.ones(len(cam))])
            return cv2.projectPoints(ideal, np.zeros(3), np.zeros(3), self.K, self.dist)[0].reshape(-1, 2), z
        uv = cam @ self.K.T
        return uv[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None], z

    def plane_to_floor(self, px: np.ndarray, height_m: float) -> np.ndarray:
        """Image pixels -> floor (x, y) of points lying on the horizontal plane z = height_m."""
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.R[:, 2] * height_m + self.tvec.ravel()])
        px = undistort_px(px, self.K, self.dist).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(px, np.linalg.inv(H)).reshape(-1, 2)

    def metres_per_pixel(self, px: tuple[float, float]) -> float:
        a = self.to_floor(np.array([px, (px[0], px[1] + 1.0), (px[0] + 1.0, px[1])]))
        return float(max(np.linalg.norm(a[1] - a[0]), np.linalg.norm(a[2] - a[0])))

    def floor_error(self, obs: dict[int, np.ndarray], floor_corners: dict[int, np.ndarray]) -> float:
        """Median floor distance between observed marker corners mapped to the floor and their true spots."""
        if self.distorted:
            obs = {i: undistort_px(c, self.K, self.dist) for i, c in obs.items()}
        return floor_error(self.H_img2floor, obs, floor_corners)


def floor_error(H_img2floor: np.ndarray, obs: dict[int, np.ndarray], floor_corners: dict[int, np.ndarray]) -> float:
    if not obs:
        return float("inf")
    img = np.concatenate([obs[i] for i in obs]).reshape(-1, 1, 2)
    flo = np.concatenate([floor_corners[i] for i in obs])
    est = cv2.perspectiveTransform(img, H_img2floor).reshape(-1, 2)
    return float(np.median(np.linalg.norm(est - flo, axis=1)))


def fit_calibration(
    obs: dict[int, np.ndarray], floor_corners: dict[int, np.ndarray], cam: CameraInfo, min_markers: int = 2
) -> Calibration | None:
    """MAGSAC homography for inlier selection, then a PnP pose (known K) for the floor mapping.

    With K known the pose has 6 degrees of freedom against the homography's 8, so it extrapolates
    better across the floor when the markers are few and bunched; the homography is kept as a
    cross-check (`homography_error_m`).
    """
    ids = [i for i in obs if i in floor_corners]
    if len(ids) < min_markers:
        return None
    img = np.concatenate([obs[i] for i in ids]).astype(np.float64)
    flo = np.concatenate([floor_corners[i] for i in ids]).astype(np.float64)
    K = np.asarray(cam.K, np.float64)
    dist = np.asarray(cam.dist, np.float64)
    # USAC rejects orientation-reversing models; image y runs down and floor y up, so fit to (x, -y).
    # Homographies hold for ideal pinhole pixels: undistort first (a no-op without lens distortion).
    Hh, inl = cv2.findHomography(undistort_px(img, K, dist), flo * (1.0, -1.0), cv2.USAC_MAGSAC, 0.10)
    if Hh is None:
        return None
    Hh = np.diag([1.0, -1.0, 1.0]) @ Hh
    keep = inl.ravel().astype(bool)
    if keep.sum() < 6:
        return None
    obj = np.column_stack([flo[keep], np.zeros(keep.sum())])
    ok, rvec, tvec = cv2.solvePnP(obj, img[keep], K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img[keep], K, dist, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    if (-R.T @ tvec).ravel()[2] <= 0:  # camera must be above the floor
        return None
    Hf2i = K @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()])
    Hi2f = np.linalg.inv(Hf2i)
    used = {i: obs[i] for i in ids}
    if np.any(dist):  # PnP applied the lens model itself; the homographies need it applied to the pixels
        used = {i: undistort_px(c, K, dist) for i, c in used.items()}
    err_pnp = floor_error(Hi2f, used, floor_corners)
    err_h = floor_error(Hh, used, floor_corners)
    return Calibration(
        K=K,
        R=R,
        tvec=tvec.reshape(3, 1),
        H_floor2img=Hf2i,
        H_img2floor=Hi2f,
        error_m=err_pnp,
        markers_used=len(ids),
        homography_error_m=err_h,
        dist=dist,
    )


def merge_observations(history: list[dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """Per-marker median of corners seen over several frames of a static camera."""
    by_id: dict[int, list[np.ndarray]] = {}
    for obs in history:
        for i, c in obs.items():
            by_id.setdefault(i, []).append(c)
    return {i: np.median(np.stack(cs), axis=0) for i, cs in by_id.items()}
