#!/usr/bin/env python3
"""
3D mesh viewer for reconstruction results.
"""

import open3d as o3d
import numpy as np
import argparse
from pathlib import Path

def view_reconstruction(obj_file):
    """View 3D reconstruction with enhanced settings."""
    
    mesh = o3d.io.read_triangle_mesh(str(obj_file))
    
    if len(mesh.vertices) == 0:
        print(f"Could not load mesh from {obj_file}")
        return
    
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    has_colors = len(mesh.vertex_colors) > 0
    has_textures = len(mesh.triangle_uvs) > 0
    
    print(f"   Colors: {'Yes' if has_colors else 'No'}")
    print(f"   Textures: {'Yes' if has_textures else 'No'}")
    
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Face Reconstruction", width=1200, height=800)
    
    vis.add_geometry(mesh)
    
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.mesh_show_wireframe = False
    render_option.light_on = True
    
    vis.reset_view_point(True)
    
    print("\nControls:")
    print("   Mouse: Rotate view")
    print("   Mouse wheel: Zoom")
    print("   Shift + mouse: Pan")
    print("   R: Reset view")
    print("   Q/ESC: Quit")
    print("\nViewing mesh... close window when done.")
    
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='View 3D reconstruction results')
    parser.add_argument('mesh_file', help='Path to OBJ mesh file')
    args = parser.parse_args()
    
    mesh_path = Path(args.mesh_file)
    if not mesh_path.exists():
        print(f"File not found: {mesh_path}")
        return 1
    
    view_reconstruction(mesh_path)
    return 0

if __name__ == "__main__":
    exit(main())
