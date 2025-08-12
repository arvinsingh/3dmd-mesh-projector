#!/usr/bin/env python3
"""
3dMD Mesh Projection Tool - Main Entry Point
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from cli import main

if __name__ == "__main__":
    sys.exit(main())
