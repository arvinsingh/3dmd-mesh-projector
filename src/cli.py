#!/usr/bin/env python3
"""
3D Mesh Projection System
Pipeline for projecting 3dMD ground-truth or reconstructed 3D meshes onto camera images
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

from view_mesh import view_reconstruction
from obj_project import process_frame_projections
from calibration import load_all_calibrations
from dataset import SeqDataset


def list_available_data(seq_dir: Path, ground_truth_dir: Path):
    """List available data for processing."""
    print("\nAvailable Data Summary")
    print("=" * 50)
    
    if seq_dir.exists():
        ds = SeqDataset(seq_dir)
        print(f"Sequence Directory: {seq_dir}")
        print(f"   Number of frames: {ds.num_frames}")
        print(f"   Frame indices: 0-{ds.num_frames-1}")
        
        try:
            calibrations = load_all_calibrations(ds.calib_dir)
            print(f"   Available cameras: {list(calibrations.keys())}")
        except Exception as e:
            print(f"   Calibration error: {e}")
    else:
        print(f"ERROR: Sequence directory not found: {seq_dir}")
    
    if ground_truth_dir.exists():
        obj_files = list(ground_truth_dir.glob("*.obj"))
        print(f"\nGround Truth Directory: {ground_truth_dir}")
        print(f"   Available meshes: {len(obj_files)}")
        if obj_files:
            examples = sorted(obj_files)[:5]
            for obj_file in examples:
                print(f"   - {obj_file.name}")
            if len(obj_files) > 5:
                print(f"   ... and {len(obj_files) - 5} more")
    else:
        print(f"ERROR: Ground truth directory not found: {ground_truth_dir}")


def create_projections(seq_dir: Path, frame_idx: int, ground_truth_dir: Path, 
                      output_dir: Path, cameras: List[str] = None):
    """Create mesh projection overlays."""
    print(f"\nCreating Mesh Projections")
    print(f"Sequence: {seq_dir}")
    print(f"Frame: {frame_idx}")
    print(f"Ground Truth: {ground_truth_dir}")
    print(f"Output: {output_dir}")
    
    try:
        results = process_frame_projections(
            seq_dir, frame_idx, ground_truth_dir, output_dir, cameras
        )
        
        if results:
            print(f"Created {len(results)} projection overlays")
            for cam_id, output_path in results.items():
                print(f"   {cam_id}: {output_path.name}")
        else:
            print("WARNING: No projections were created")
            
    except Exception as e:
        print(f"ERROR: Projection creation failed: {e}")


def view_mesh(mesh_path: Path):
    """View a 3D mesh file."""
    print(f"\nViewing Mesh: {mesh_path}")
    
    if not mesh_path.exists():
        print(f"ERROR: Mesh file not found: {mesh_path}")
        return
    
    try:
        view_reconstruction(mesh_path)
        print("Mesh viewing completed")
    except Exception as e:
        print(f"ERROR: Mesh viewing failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="3D Mesh Projection System for 3dMD Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available data
  python main_entry.py data/sequence1 --list
  
  # Create projection overlays for frame 5
  python main_entry.py data/sequence1 --project --frame 5
  
  # View existing mesh
  python main_entry.py data/sequence1 --view-mesh data/ground-truth/XAN1_000.obj
        """
    )
    
    parser.add_argument(
        "seq_dir", type=Path,
        help="Path to sequence directory (e.g., data/sequence1)"
    )
    
    # action args
    parser.add_argument(
        "--list", action="store_true",
        help="List available data and exit"
    )
    parser.add_argument(
        "--project", action="store_true",
        help="Create mesh projection overlays"
    )
    parser.add_argument(
        "--view-mesh", type=Path,
        help="View specific mesh file"
    )
    
    # config args
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to process (default: 0)"
    )
    parser.add_argument(
        "--ground-truth", type=Path,
        default=Path("data/ground-truth"),
        help="Ground truth directory (default: data/ground-truth)"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs"),
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--cameras", nargs="+",
        help="Camera IDs for projections (default: all available)"
    )
    
    args = parser.parse_args()
    
    # validate seq dir
    if not args.seq_dir.exists():
        print(f"ERROR: Sequence directory not found: {args.seq_dir}")
        return 1
    
    # handle list command
    if args.list:
        list_available_data(args.seq_dir, args.ground_truth)
        return 0
    
    # handle view-mesh command
    if args.view_mesh:
        view_mesh(args.view_mesh)
        return 0
    
    # handle project command
    if args.project:
        print("3D Mesh Projection System")
        print("=" * 40)
        
        create_projections(
            args.seq_dir, args.frame, args.ground_truth,
            args.output / "projections", args.cameras
        )
        
        print(f"\nProcessing complete!")
        print(f"Check outputs in: {args.output}")
        return 0
    
    print("ERROR: No action specified. Use --list, --project, or --view-mesh")
    return 1


if __name__ == "__main__":
    sys.exit(main())
