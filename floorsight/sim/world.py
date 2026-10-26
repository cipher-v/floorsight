"""Geometry and rendering for the simulated warehouse.

Renders what a real ceiling-mounted camera would see: a textured floor with
lane paint and ArUco markers, racks, people, a forklift and pallets, with
shadows, sensor noise and lighting flicker. The vision pipeline never reads
anything from here except the rendered pixels and the public site plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from floorsight.cvsafe import fill_poly

PX_PER_M = 80  # floor texture resolution


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


def look_at(position, target, up=(0.0, 0.0, 1.0)):
    """World->camera rotation and translation (OpenCV axes: x right, y down, z forward)."""
    c = np.asarray(position, float)
    f = np.asarray(target, float) - c
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)  # image "down"
    R = np.stack([r, d, f])  # rows are camera axes in world coordinates
    t = -R @ c
    return R, t


def intrinsics(width, height, hfov_deg):
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], float)


@dataclass
class SimCamera:
    camera_id: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    width: int = 960
    height: int = 540
    hfov_deg: float = 78.0
    fps: float = 10.0
    # Optional "knock": after bump_t the camera is rotated by bump_yaw/bump_pitch degrees.
    bump_t: float | None = None
    bump_yaw_deg: float = 0.0
    bump_pitch_deg: float = 0.0
    K: np.ndarray = field(init=False)

    def __post_init__(self):
        self.K = intrinsics(self.width, self.height, self.hfov_deg)
        self._R0, self._t0 = look_at(self.position, self.target)

    def pose(self, t: float):
        if self.bump_t is None or t < self.bump_t:
            return self._R0, self._t0
        yaw = math.radians(self.bump_yaw_deg)
        pitch = math.radians(self.bump_pitch_deg)
        # Rotate about the camera's own axes (a knock on the mount).
        Ry = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0]
        Rx = cv2.Rodrigues(np.array([pitch, 0.0, 0.0]))[0]
        R = Rx @ Ry @ self._R0
        c = np.asarray(self.position, float)
        return R, -R @ c

    def project(self, pts_world: np.ndarray, t: float):
        R, tv = self.pose(t)
        cam = (R @ pts_world.T).T + tv
        z = cam[:, 2]
        uv = (self.K @ cam.T).T
        uv = uv[:, :2] / np.maximum(z[:, None], 1e-6)
        return uv, z

    def floor_homography(self, t: float):
        """Floor metres (x, y) -> image pixels."""
        R, tv = self.pose(t)
        Rt = np.column_stack([R[:, 0], R[:, 1], tv])
        return self.K @ Rt


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def box_faces(cx, cy, w, d, h, heading=0.0, z0=0.0):
    """Faces of an oriented box: list of (4x3 polygon, outward normal)."""
    c, s = math.cos(heading), math.sin(heading)
    hw, hd = w / 2, d / 2
    local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    base = [(cx + x * c - y * s, cy + x * s + y * c) for x, y in local]
    bot = [(x, y, z0) for x, y in base]
    top = [(x, y, z0 + h) for x, y in base]
    faces = [(np.array(top), np.array([0, 0, 1.0]))]
    for i in range(4):
        j = (i + 1) % 4
        quad = np.array([bot[i], bot[j], top[j], top[i]])
        ex, ey = base[j][0] - base[i][0], base[j][1] - base[i][1]
        n = np.array([ey, -ex, 0.0])
        n /= np.linalg.norm(n) + 1e-9
        faces.append((quad, n))
    return faces


def prism_faces(cx, cy, radius, z0, z1, sides=12):
    ring = [
        (cx + radius * math.cos(2 * math.pi * k / sides), cy + radius * math.sin(2 * math.pi * k / sides))
        for k in range(sides)
    ]
    faces = [(np.array([(x, y, z1) for x, y in ring]), np.array([0, 0, 1.0]))]
    for i in range(sides):
        j = (i + 1) % sides
        quad = np.array([(*ring[i], z0), (*ring[j], z0), (*ring[j], z1), (*ring[i], z1)])
        mx, my = (ring[i][0] + ring[j][0]) / 2 - cx, (ring[i][1] + ring[j][1]) / 2 - cy
        n = np.array([mx, my, 0.0])
        n /= np.linalg.norm(n) + 1e-9
        faces.append((quad, n))
    return faces


LIGHT = np.array([0.35, -0.5, 0.8])
LIGHT /= np.linalg.norm(LIGHT)


def shade(color, normal):
    k = 0.55 + 0.45 * max(0.0, float(np.dot(normal, LIGHT)))
    return tuple(int(min(255, c * k)) for c in color)


@dataclass
class Body:
    """Something drawn in 3D. `faces` is a list of (polygon, normal, colour)."""

    faces: list
    shadow: tuple[float, float, float, float] | None = None  # cx, cy, rx, ry on the floor


def make_person(x, y, shirt, pants=(60, 50, 45), skin=(140, 170, 205)):
    faces = []
    for poly, n in prism_faces(x, y, 0.22, 0.0, 0.85, sides=10):
        faces.append((poly, n, pants))
    for poly, n in prism_faces(x, y, 0.25, 0.85, 1.45, sides=10):
        faces.append((poly, n, shirt))
    for poly, n in prism_faces(x, y, 0.12, 1.45, 1.72, sides=8):
        faces.append((poly, n, skin))
    return Body(faces, shadow=(x + 0.12, y - 0.1, 0.32, 0.26))


def make_forklift(x, y, heading):
    faces = []
    body = (40, 170, 230)  # BGR yellow-orange
    for poly, n in box_faces(x, y, 2.0, 1.1, 1.2, heading):
        faces.append((poly, n, body))
    # Mast and cab frame
    mx, my = x + 1.0 * math.cos(heading), y + 1.0 * math.sin(heading)
    for poly, n in box_faces(mx, my, 0.15, 0.9, 2.1, heading):
        faces.append((poly, n, (40, 40, 40)))
    cx, cy = x - 0.3 * math.cos(heading), y - 0.3 * math.sin(heading)
    for poly, n in box_faces(cx, cy, 0.9, 0.9, 0.9, heading, z0=1.2):
        faces.append((poly, n, (35, 120, 170)))
    return Body(faces, shadow=(x + 0.25, y - 0.2, 1.25, 0.75))


def make_pallet(x, y):
    faces = []
    for poly, n in box_faces(x, y, 1.2, 1.0, 0.15, 0.0):
        faces.append((poly, n, (70, 110, 150)))
    for poly, n in box_faces(x, y, 1.1, 0.95, 0.85, 0.0, z0=0.15):
        faces.append((poly, n, (95, 150, 190)))
    return Body(faces, shadow=(x + 0.2, y - 0.15, 0.75, 0.6))


def subdivide(quad, step=0.5):
    """Split a planar quad into tiles so painter's ordering stays correct near small objects."""
    a, b, c, d = quad
    nu = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
    nv = max(1, int(math.ceil(np.linalg.norm(d - a) / step)))
    tiles = []
    for i in range(nu):
        for j in range(nv):
            u0, u1 = i / nu, (i + 1) / nu
            v0, v1 = j / nv, (j + 1) / nv

            def at(u, v):
                return a + (b - a) * u + (d - a) * v

            tiles.append(np.array([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)]))
    return tiles


def make_rack(polygon, height):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    w, d = max(xs) - min(xs), max(ys) - min(ys)
    faces = []
    for poly, n in box_faces(cx, cy, w, d, height, 0.0):
        for k, tile in enumerate(subdivide(poly, 0.5)):
            shelf = (128, 100, 74) if k % 2 == 0 else (118, 92, 68)
            faces.append((tile, n, shelf))
    return Body(faces)


# --------------------------------------------------------------------------- #
# Floor texture
# --------------------------------------------------------------------------- #


def floor_to_tex(xy, depth_m):
    x, y = xy
    return int(round(x * PX_PER_M)), int(round((depth_m - y) * PX_PER_M))


def build_floor_texture(site: dict, seed: int = 7) -> np.ndarray:
    W, D = site["width_m"], site["depth_m"]
    w, h = int(W * PX_PER_M), int(D * PX_PER_M)
    rng = np.random.default_rng(seed)
    tex = np.full((h, w, 3), 158, np.uint8)
    noise = rng.normal(0, 7, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    tex = np.clip(tex.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    # Concrete slab seams every 2 m
    for gx in np.arange(0, W + 0.01, 2.0):
        u = int(gx * PX_PER_M)
        cv2.line(tex, (u, 0), (u, h), (135, 135, 135), 1)
    for gy in np.arange(0, D + 0.01, 2.0):
        v = int(gy * PX_PER_M)
        cv2.line(tex, (0, v), (w, v), (135, 135, 135), 1)

    def poly_px(poly):
        return np.array([floor_to_tex(p, D) for p in poly], np.int32)

    for z in site["zones"]:
        p = poly_px(z["polygon"])
        if z["kind"] == "lane":
            cv2.polylines(tex, [p], True, (30, 200, 235), 6)
        elif z["kind"] == "exit":
            overlay = tex.copy()
            fill_poly(overlay, [p], (70, 160, 60))
            tex = cv2.addWeighted(overlay, 0.45, tex, 0.55, 0)
            cv2.polylines(tex, [p], True, (40, 140, 40), 5)
            x0, y0 = p.min(axis=0)
            cv2.putText(tex, "EXIT B", (int(x0) + 8, int(y0) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        elif z["kind"] == "restricted":
            overlay = tex.copy()
            fill_poly(overlay, [p], (170, 120, 80))
            tex = cv2.addWeighted(overlay, 0.35, tex, 0.65, 0)
            cv2.polylines(tex, [p], True, (0, 0, 200), 4)
        elif z["kind"] == "area" and "crossing" in z["zone_id"]:
            x0, y0 = p.min(axis=0)
            x1, y1 = p.max(axis=0)
            for k in range(int(x0), int(x1), 40):
                cv2.rectangle(tex, (k, int(y0)), (k + 18, int(y1)), (225, 225, 225), -1)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for m in site["markers"]:
        side = int(m["size_m"] * PX_PER_M)
        img = cv2.aruco.generateImageMarker(aruco_dict, m["marker_id"], side)
        quiet = side // 4
        tile = np.full((side + 2 * quiet, side + 2 * quiet), 255, np.uint8)
        tile[quiet : quiet + side, quiet : quiet + side] = img
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        # Texture rows run against floor y, so flip so the marker reads correctly from above.
        cu, cv_ = floor_to_tex(m["center"], D)
        th, tw = tile.shape[:2]
        y0, x0 = cv_ - th // 2, cu - tw // 2
        tex[y0 : y0 + th, x0 : x0 + tw] = tile
    return tex


def tex_to_floor_matrix(depth_m):
    """Texture pixel (u, v) -> floor metres (x, y), homogeneous."""
    return np.array([[1 / PX_PER_M, 0, 0], [0, -1 / PX_PER_M, depth_m], [0, 0, 1]], float)


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #


class Renderer:
    def __init__(self, site: dict, cameras: list[SimCamera], seed: int = 11):
        self.site = site
        self.cameras = cameras
        self.tex = build_floor_texture(site)
        self.T = tex_to_floor_matrix(site["depth_m"])
        self.rng = np.random.default_rng(seed)
        self.racks = [make_rack(o["polygon"], o["height_m"]) for o in site["obstacles"]]
        self._bg_cache: dict = {}

    def _background(self, cam: SimCamera, t: float):
        key = (cam.camera_id, cam.bump_t is not None and t >= cam.bump_t)
        if key in self._bg_cache:
            return self._bg_cache[key]
        H = cam.floor_homography(t) @ self.T
        img = np.zeros((cam.height, cam.width, 3), np.uint8)
        grad = np.linspace(70, 105, cam.height, dtype=np.float32)[:, None, None]
        img[:] = np.clip(grad * np.array([1.0, 1.0, 1.05]), 0, 255).astype(np.uint8)
        floor = cv2.warpPerspective(self.tex, H, (cam.width, cam.height), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(
            np.full(self.tex.shape[:2], 255, np.uint8), H, (cam.width, cam.height), flags=cv2.INTER_NEAREST
        )
        # Discard anything that maps from behind the camera.
        R, tv = cam.pose(t)
        ys, xs = np.mgrid[0 : cam.height, 0 : cam.width]
        ray = np.linalg.inv(cam.K) @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
        world_ray = R.T @ ray
        mask[(world_ray[2] >= 0).reshape(cam.height, cam.width)] = 0
        img[mask > 0] = floor[mask > 0]
        self._bg_cache[key] = (img, H)
        return img, H

    def _draw_faces(self, img, cam, t, faces):
        R, tv = cam.pose(t)
        centre = -R.T @ tv
        items = []
        for poly, n, color in faces:
            mid = poly.mean(axis=0)
            if np.dot(n, centre - mid) <= 0:
                continue
            uv, z = cam.project(poly, t)
            if (z < 0.2).any():
                continue
            items.append((float(np.linalg.norm(centre - mid)), uv, shade(color, n)))
        return items

    def render(self, cam: SimCamera, t: float, bodies: list[Body], glare=None):
        base, H = self._background(cam, t)
        img = base.copy()
        # Shadows
        shadow_layer = img.copy()
        for b in bodies:
            if b.shadow is None:
                continue
            cx, cy, rx, ry = b.shadow
            ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            pts = np.stack([cx + rx * np.cos(ang), cy + ry * np.sin(ang), np.zeros_like(ang)], 1)
            uv, z = cam.project(pts, t)
            if (z > 0.2).all():
                fill_poly(shadow_layer, [uv], (60, 60, 60))
        img = cv2.addWeighted(shadow_layer, 0.45, img, 0.55, 0)
        # Painter's algorithm over every face in the scene
        items = []
        for b in self.racks + bodies:
            items += self._draw_faces(img, cam, t, b.faces)
        items.sort(key=lambda it: -it[0])
        for _, uv, color in items:
            fill_poly(img, [uv], color)
        if glare is not None:
            gx, gy, gr, strength = glare
            uv, z = cam.project(np.array([[gx, gy, 0.0]]), t)
            if z[0] > 0:
                layer = np.zeros_like(img, np.float32)
                cv2.ellipse(layer, (int(uv[0, 0]), int(uv[0, 1])), (int(gr * 1.6), int(gr)), 0, 0, 360, (255, 255, 255), -1)
                layer = cv2.GaussianBlur(layer, (0, 0), gr / 3)
                img = np.clip(img.astype(np.float32) + layer * strength, 0, 255).astype(np.uint8)
        # Sensor effects: flicker, noise, slight blur
        flicker = 1.0 + 0.025 * math.sin(2 * math.pi * 0.37 * t + hash(cam.camera_id) % 7)
        noisy = img.astype(np.float32) * flicker + self.rng.normal(0, 3.0, img.shape).astype(np.float32)
        img = np.clip(noisy, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(img, (3, 3), 0.6)
