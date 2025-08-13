#!/usr/bin/env python3
"""
Mesh projection tool for 3D reconstruction visualization.
Projects ground-truth .obj meshes onto camera images to create overlay visualizations.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from typing import Dict, List, Tuple

from calibration import load_all_calibrations, CameraCalibration
from dataset import SeqDataset


def project_vertices(mesh_path: Path, cam: CameraCalibration) -> np.ndarray:
    """
    Project 3D mesh vertices onto 2D image coordinates using camera calibration.
    
    Args:
        mesh_path: Path to .obj mesh file
        cam: Camera calibration parameters
        
    Returns:
        Array of 2D coordinates (u, v) for each vertex
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    
    if len(vertices) == 0:
        return np.empty((0, 2))
    
    # transform world coordinates to camera coordinates
    # X_cam = R * X_world + t
    vertices_cam = (cam.R @ vertices.T + cam.t.reshape(3, 1)).T
    
    # proj to image coordinates using intrinsic matrix
    # apply lens distortion correction
    vertices_2d, _ = cv2.projectPoints(
        vertices_cam.reshape(-1, 1, 3),
        np.zeros(3),  # no additional rotation
        np.zeros(3),  # no additional translation
        cam.K,
        cam.dist
    )
    
    return vertices_2d.reshape(-1, 2)


def project_mesh_wireframe(mesh_path: Path, cam: CameraCalibration, 
                          image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Project 3D mesh wireframe onto 2D image coordinates.
    
    Args:
        mesh_path: Path to .obj mesh file
        cam: Camera calibration parameters
        image_shape: (height, width) of target image
        
    Returns:
        2D wireframe mask image
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(vertices) == 0 or len(triangles) == 0:
        return np.zeros(image_shape, dtype=np.uint8)
    
    # proj vertices
    vertices_2d = project_vertices(mesh_path, cam)
    
    # create wireframe mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for triangle in triangles:
        # get triangle vertices in 2D
        v1, v2, v3 = vertices_2d[triangle].astype(int)
        
        # draw triangle edges if within image bounds
        if (0 <= v1[0] < image_shape[1] and 0 <= v1[1] < image_shape[0] and
            0 <= v2[0] < image_shape[1] and 0 <= v2[1] < image_shape[0]):
            cv2.line(mask, tuple(v1), tuple(v2), 255, 1)
            
        if (0 <= v2[0] < image_shape[1] and 0 <= v2[1] < image_shape[0] and
            0 <= v3[0] < image_shape[1] and 0 <= v3[1] < image_shape[0]):
            cv2.line(mask, tuple(v2), tuple(v3), 255, 1)
            
        if (0 <= v3[0] < image_shape[1] and 0 <= v3[1] < image_shape[0] and
            0 <= v1[0] < image_shape[1] and 0 <= v1[1] < image_shape[0]):
            cv2.line(mask, tuple(v3), tuple(v1), 255, 1)
    
    return mask


def overlay_vertices(image: np.ndarray, uv: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Legacy function for backward compatibility."""
    img = image.copy()
    for p in uv.astype(int):
        u, v = p
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 1, color, -1)
    return img


def create_overlay_image(image: np.ndarray, vertices_2d: np.ndarray, 
                        wireframe: np.ndarray = None, 
                        vertex_color: Tuple[int, int, int] = (0, 255, 0),
                        wireframe_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    Create overlay image with projected mesh.
    
    Args:
        image: Input image (grayscale or color)
        vertices_2d: 2D vertex coordinates
        wireframe: Optional wireframe mask
        vertex_color: Color for vertex points (BGR)
        wireframe_color: Color for wireframe (BGR)
        
    Returns:
        Overlay image with mesh projection
    """
    # handle both grayscale and color img
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        overlay = image.copy()
    else:
        # convert other formats to grayscale then to BGR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Draw wireframe if available
    if wireframe is not None and np.any(wireframe > 0):
        # draw wireframe lines directly on overlay
        wireframe_points = np.where(wireframe > 0)
        for y, x in zip(wireframe_points[0], wireframe_points[1]):
            if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                overlay[y, x] = wireframe_color
    
    # draw vertex points
    for vertex in vertices_2d:
        u, v = int(vertex[0]), int(vertex[1])
        if 0 <= u < overlay.shape[1] and 0 <= v < overlay.shape[0]:
            cv2.circle(overlay, (u, v), 2, vertex_color, -1)
    
    return overlay


def process_frame_projections(seq_dir: Path, frame_idx: int, 
                            mesh_dir: Path, output_dir: Path,
                            cameras: List[str] = None) -> Dict[str, Path]:
    """
    Process all camera projections for a single frame.
    
    Args:
        seq_dir: Sequence directory containing frames and calibration files
        frame_idx: Frame index to process
        mesh_dir: Directory containing mesh (.obj) files
        output_dir: Output directory for overlay images
        cameras: List of camera IDs to process (default: all available)
        
    Returns:
        Dictionary mapping camera ID to output file path
    """
    ds = SeqDataset(seq_dir)
    calibrations = load_all_calibrations(seq_dir)
    
    if cameras is None:
        cameras = list(calibrations.keys())
    
    # find corresponding mesh file - look for various naming patterns
    mesh_patterns = [
        f"*_{frame_idx:03d}.obj",        # pattern: prefix_000.obj  
        f"frame_{frame_idx:03d}.obj",    # pattern: frame_000.obj
        f"{frame_idx:03d}.obj",          # pattern: 000.obj
        f"mesh_{frame_idx:03d}.obj",     # pattern: mesh_000.obj
    ]
    
    mesh_path = None
    for pattern in mesh_patterns:
        mesh_files = list(mesh_dir.glob(pattern))
        if mesh_files:
            mesh_path = mesh_files[0]  # use first matching file
            break
    
    if mesh_path is None:
        raise FileNotFoundError(f"No mesh file found for frame {frame_idx} in {mesh_dir}. Tried patterns: {mesh_patterns}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for cam_id in cameras:
        if cam_id not in calibrations:
            print(f"Warning: No calibration found for camera {cam_id}")
            continue
        
        try:
            bundle = ds.load_bundle(frame_idx, [cam_id])
            image = bundle.images[cam_id]
            
            cam = calibrations[cam_id]
            vertices_2d = project_vertices(mesh_path, cam)
            
            wireframe = project_mesh_wireframe(mesh_path, cam, image.shape)
            
            overlay = create_overlay_image(
                image, vertices_2d, wireframe,
                vertex_color=(0, 255, 0),  # Green vertices
                wireframe_color=(0, 0, 255)  # Red wireframe
            )
            
            output_path = output_dir / f"projection_{cam_id}_frame_{frame_idx:03d}.png"
            cv2.imwrite(str(output_path), overlay)
            results[cam_id] = output_path
            
            print(f"Created projection for camera {cam_id}: {output_path}")
            
        except Exception as e:
            print(f"Failed to process camera {cam_id}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Project meshes onto camera images"
    )
    parser.add_argument(
        "seq_dir", type=Path,
        help="Path to sequence directory containing images and calibration files"
    )
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to process (default: 0)"
    )
    parser.add_argument(
        "--mesh-dir", type=Path, required=True,
        help="Directory containing mesh (.obj) files"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/projections"),
        help="Output directory (default: outputs/projections)"
    )
    parser.add_argument(
        "--cameras", nargs="+",
        help="Camera IDs to process (default: all available)"
    )
    
    args = parser.parse_args()
    
    # validate paths
    if not args.seq_dir.exists():
        print(f"ERROR: Sequence directory not found: {args.seq_dir}")
        return 1
    
    if not args.mesh_dir.exists():
        print(f"ERROR: Mesh directory not found: {args.mesh_dir}")
        return 1
    
    try:
        results = process_frame_projections(
            args.seq_dir, args.frame, args.mesh_dir, 
            args.output, args.cameras
        )
        
        print(f"\nSuccessfully created {len(results)} projection overlays")
        print(f"Output directory: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Error processing projections: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
