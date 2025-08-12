from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class FrameBundle:
    index: int
    images: Dict[str, np.ndarray]  # key: camera id e.g. '1A'


class SeqDataset:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.calib_dir = self.root / "calib"
        if not self.frames_dir.exists():
            raise FileNotFoundError(self.frames_dir)
        # discover max index by listing one camera pattern
        example = sorted(self.frames_dir.glob("STEREO_1A_*.bmp"))
        if not example:
            raise FileNotFoundError("No frames found for STEREO_1A_*.bmp")
        self.num_frames = len(example)

    def load_bundle(self, idx: int, cams: List[str]) -> FrameBundle:
        images: Dict[str, np.ndarray] = {}
        for cam in cams:
            prefix = "STEREO" if cam.endswith(("A", "B")) else "TEXTURE"
            name = f"{prefix}_{cam}_{idx:03d}.bmp"
            path = self.frames_dir / name
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if prefix == "STEREO" else cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            images[cam] = img
        return FrameBundle(index=idx, images=images)

    def frame_indices(self) -> List[int]:
        return list(range(self.num_frames))
