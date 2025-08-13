import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class CameraCalibration:
    cam_id: str
    K: np.ndarray  # 3x3
    dist: np.ndarray  # (k1, k2, p1, p2, k3)
    R: np.ndarray  # world->camera rotation 3x3
    t: np.ndarray  # world->camera translation 3x1
    image_size: Tuple[int, int]  # (w, h)

    def projection_matrix(self) -> np.ndarray:
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ Rt


def _parse_float_list(line: str) -> np.ndarray:
    return np.array([float(x) for x in line.strip().split()], dtype=np.float64)


def read_tka_file(path: Path) -> CameraCalibration:
    text = path.read_text(errors="ignore")
    # normalize newlines and strip CR
    lines = [ln.strip().rstrip("\r") for ln in text.splitlines() if ln.strip()]

    # expect format with %M rotation matrix across 3 lines following '%M'
    R = None
    X = Y = Z = None
    f_mm = None
    k1 = k2 = 0.0
    p1 = p2 = 0.0
    sx = sy = None  # pixel size (mm)
    cx = cy = None  # principal point (pixels)
    w = h = None
    cam_id = path.stem.replace("calib_", "")

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln == "%M":
            # next 3 lines are rows of rotation
            r0 = _parse_float_list(lines[i + 1])
            r1 = _parse_float_list(lines[i + 2])
            r2 = _parse_float_list(lines[i + 3])
            R = np.vstack([r0, r1, r2])
            i += 4
            continue
        if ln.startswith("%X"):
            X = float(ln.split()[1])
        elif ln.startswith("%Y"):
            Y = float(ln.split()[1])
        elif ln.startswith("%Z"):
            Z = float(ln.split()[1])
        elif ln.startswith("%f"):
            f_mm = float(ln.split()[1])
        elif re.match(r"%K2? ", ln):
            parts = ln.split()
            if parts[0] == "%K":
                k1 = float(parts[1])
            elif parts[0] == "%K2":
                k2 = float(parts[1])
        elif ln.startswith("%S"):
            # scale ignored
            pass
        elif ln.startswith("%x"):
            sx = float(ln.split()[1])
        elif ln.startswith("%y"):
            sy = float(ln.split()[1])
        elif ln.startswith("%a"):
            cx = float(ln.split()[1])
        elif ln.startswith("%b"):
            cy = float(ln.split()[1])
        elif ln.startswith("%is"):
            _, w_str, h_str = ln.split()
            w = int(w_str)
            h = int(h_str)
        elif ln.startswith("%c"):
            # decentering given later as zeros; keep default
            pass
        i += 1

    if any(v is None for v in (R, X, Y, Z, f_mm, sx, sy, cx, cy, w, h)):
        raise ValueError(f"Incomplete calibration in {path}")

    # Intrinsics
    fx = f_mm / sx
    fy = f_mm / sy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

    # Extrinsics - lines provide camera center in world coords...
    C = np.array([X, Y, Z], dtype=np.float64)
    # By convention x_cam = R * X_world + t, with t = -R * C
    t = -R @ C

    return CameraCalibration(
        cam_id=cam_id,
        K=K,
        dist=dist,
        R=R,
        t=t.reshape(3),
        image_size=(w, h),
    )


def load_all_calibrations(seq_dir: Path) -> Dict[str, CameraCalibration]:
    """Load calibration files from the sequence directory (same directory as images)."""
    cams: Dict[str, CameraCalibration] = {}
    for cam in ["1A", "1B", "1C", "2A", "2B", "2C"]:
        p = seq_dir / f"calib_{cam}.tka"
        if not p.exists():
            continue
        cams[cam] = read_tka_file(p)
    if not cams:
        raise FileNotFoundError(f"No calib_*.tka files in {seq_dir}")
    return cams


def relative_pose(c1: CameraCalibration, c2: CameraCalibration) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pose of cam2 relative to cam1: x2 = R * x1 + T
    Using world->cam poses: x_i = R_i X + t_i
    R = R2 * R1^T, T = t2 - R * t1
    """
    R = c2.R @ c1.R.T
    T = c2.t - R @ c1.t
    return R, T
