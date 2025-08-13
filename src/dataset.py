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
        self.frames_dir = self.root  # Images and calibration files are in the same directory
        # Calibration files are expected to be in the same directory as images
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        # discover max index by listing one camera pattern
        example = sorted(self.root.glob("STEREO_1A_*.bmp"))
        if not example:
            raise FileNotFoundError("No frames found for STEREO_1A_*.bmp")
        self.num_frames = len(example)

    def load_bundle(self, idx: int, cams: List[str]) -> FrameBundle:
        images: Dict[str, np.ndarray] = {}
        for cam in cams:
            prefix = "STEREO" if cam.endswith(("A", "B")) else "TEXTURE"
            name = f"{prefix}_{cam}_{idx:03d}.bmp"
            path = self.root / name  # Images are in the root directory
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if prefix == "STEREO" else cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            images[cam] = img
        return FrameBundle(index=idx, images=images)

    def frame_indices(self) -> List[int]:
        return list(range(self.num_frames))
