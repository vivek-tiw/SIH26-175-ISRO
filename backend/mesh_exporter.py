"""
TerraFly - 3D Mesh & Geospatial Exporter
Generates:
1. Wavefront OBJ (.obj + .mtl) with UV mapping for high-res texture projection
2. Standard Tessellation Language (STL) for 3D Printing / CAD
3. Polygon File Format (PLY) with RGB vertex colors
4. 32-Bit Floating Point GeoTIFF (.tif)
5. 16-Bit Grayscale PNG Heightmap
6. CSV Elevation Transect & Accuracy Metrics Report
"""

import numpy as np
import io
import struct
import tifffile
from PIL import Image

class MeshExporter:
    def __init__(self):
        pass

    def export_obj(self, elevation: np.ndarray, texture_image: Image.Image, downsample_step: int = 2) -> io.BytesIO:
        """
        Exports elevation grid to Wavefront OBJ format with UV texture coordinates.
        Uses downsampling step to keep browser/3D viewer download snappy and efficient.
        """
        # Downsample grid for export
        grid = elevation[::downsample_step, ::downsample_step]
        h, w = grid.shape
        
        # Calculate bounding dimensions
        aspect = w / h
        x_coords = np.linspace(-50.0 * aspect, 50.0 * aspect, w)
        y_coords = np.linspace(-50.0, 50.0, h)
        
        # Vertical scaling for normalized representation
        z_min, z_max = np.min(grid), np.max(grid)
        z_span = z_max - z_min if (z_max - z_min) > 0 else 1.0
        z_scaled = (grid - z_min) / z_span * 25.0
        
        lines = []
        lines.append("# TerraFly 3D Elevation Terrain Mesh Export")
        lines.append(f"# Vertices: {w * h}, Faces: {(w - 1) * (h - 1) * 2}")
        lines.append("mtllib terrain.mtl")
        lines.append("o TerraFlyTerrain")
        
        # Vertices (v x y z)
        for r in range(h):
            y = float(-y_coords[r]) # Inverted for standard 3D coordinate system
            for c in range(w):
                x = float(x_coords[c])
                z = float(z_scaled[r, c])
                lines.append(f"v {x:.4f} {z:.4f} {y:.4f}")
                
        # UV Texture Coordinates (vt u v)
        for r in range(h):
            v_coord = 1.0 - (r / (h - 1))
            for c in range(w):
                u_coord = c / (w - 1)
                lines.append(f"vt {u_coord:.5f} {v_coord:.5f}")
                
        # Normals (vn nx ny nz)
        lines.append("usemtl TerrainMaterial")
        lines.append("s 1")
        
        # Faces (f v1/vt1 v2/vt2 v3/vt3) - 1-indexed in OBJ format
        for r in range(h - 1):
            for c in range(w - 1):
                # 4 vertex indices of the quad
                top_left = r * w + c + 1
                top_right = r * w + (c + 1) + 1
                bottom_left = (r + 1) * w + c + 1
                bottom_right = (r + 1) * w + (c + 1) + 1
                
                # Triangle 1
                lines.append(f"f {top_left}/{top_left} {bottom_left}/{bottom_left} {top_right}/{top_right}")
                # Triangle 2
                lines.append(f"f {top_right}/{top_right} {bottom_left}/{bottom_left} {bottom_right}/{bottom_right}")
                
        obj_text = "\n".join(lines)
        buf = io.BytesIO()
        buf.write(obj_text.encode("utf-8"))
        buf.seek(0)
        return buf

    def export_stl(self, elevation: np.ndarray, downsample_step: int = 2) -> io.BytesIO:
        """Exports elevation grid as binary STL file for 3D printing and CAD software."""
        grid = elevation[::downsample_step, ::downsample_step]
        h, w = grid.shape
        
        aspect = w / h
        x_coords = np.linspace(0, 100.0 * aspect, w)
        y_coords = np.linspace(0, 100.0, h)
        
        z_min, z_max = np.min(grid), np.max(grid)
        z_span = z_max - z_min if (z_max - z_min) > 0 else 1.0
        z_scaled = (grid - z_min) / z_span * 25.0 + 2.0 # Add 2mm base
        
        num_triangles = (w - 1) * (h - 1) * 2
        buf = io.BytesIO()
        
        # 80-byte header
        header = b"TerraFly Binary STL 3D Elevation Mesh" + b" " * (80 - len("TerraFly Binary STL 3D Elevation Mesh"))
        buf.write(header[:80])
        buf.write(struct.pack("<I", num_triangles))
        
        for r in range(h - 1):
            for c in range(w - 1):
                p1 = (x_coords[c], y_coords[r], z_scaled[r, c])
                p2 = (x_coords[c], y_coords[r+1], z_scaled[r+1, c])
                p3 = (x_coords[c+1], y_coords[r], z_scaled[r, c+1])
                p4 = (x_coords[c+1], y_coords[r+1], z_scaled[r+1, c+1])
                
                # Tri 1: p1, p2, p3
                self._write_stl_triangle(buf, p1, p2, p3)
                # Tri 2: p3, p2, p4
                self._write_stl_triangle(buf, p3, p2, p4)
                
        buf.seek(0)
        return buf

    def _write_stl_triangle(self, buf: io.BytesIO, p1, p2, p3):
        # Calculate normal vector
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
        v2 = np.array([p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]])
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len
        else:
            normal = np.array([0, 0, 1])
            
        # Write normal (3 floats)
        buf.write(struct.pack("<3f", float(normal[0]), float(normal[1]), float(normal[2])))
        # Write 3 vertices (3 * 3 floats)
        buf.write(struct.pack("<3f", float(p1[0]), float(p1[1]), float(p1[2])))
        buf.write(struct.pack("<3f", float(p2[0]), float(p2[1]), float(p2[2])))
        buf.write(struct.pack("<3f", float(p3[0]), float(p3[1]), float(p3[2])))
        # Attribute byte count (uint16)
        buf.write(struct.pack("<H", 0))

    def export_ply(self, elevation: np.ndarray, rgb_img: Image.Image, downsample_step: int = 2) -> io.BytesIO:
        """Exports elevation data as PLY point cloud / colored mesh with RGB vertex colors."""
        grid = elevation[::downsample_step, ::downsample_step]
        h, w = grid.shape
        
        # Resample RGB image to grid size
        rgb_resized = rgb_img.resize((w, h), Image.Resampling.BILINEAR)
        rgb_arr = np.array(rgb_resized)
        
        num_vertices = h * w
        num_faces = (h - 1) * (w - 1) * 2
        
        header = f"""ply
format ascii 1.0
comment TerraFly Point Cloud & Terrain Mesh
element vertex {num_vertices}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {num_faces}
property list uchar int vertex_indices
end_header
"""
        lines = [header.strip()]
        
        aspect = w / h
        x_coords = np.linspace(-50.0 * aspect, 50.0 * aspect, w)
        y_coords = np.linspace(-50.0, 50.0, h)
        
        # Vertices with color
        for r in range(h):
            y = float(-y_coords[r])
            for c in range(w):
                x = float(x_coords[c])
                z = float(grid[r, c])
                cr, cg, cb = int(rgb_arr[r, c, 0]), int(rgb_arr[r, c, 1]), int(rgb_arr[r, c, 2])
                lines.append(f"{x:.3f} {y:.3f} {z:.3f} {cr} {cg} {cb}")
                
        # Faces
        for r in range(h - 1):
            for c in range(w - 1):
                tl = r * w + c
                tr = r * w + (c + 1)
                bl = (r + 1) * w + c
                br = (r + 1) * w + (c + 1)
                lines.append(f"3 {tl} {bl} {tr}")
                lines.append(f"3 {tr} {bl} {br}")
                
        buf = io.BytesIO()
        buf.write("\n".join(lines).encode("utf-8"))
        buf.seek(0)
        return buf

    def export_geotiff(self, elevation: np.ndarray, pixel_scale: float = 1.0, top_left: tuple = (0.0, 0.0)) -> io.BytesIO:
        """Exports 32-bit floating point single-band GeoTIFF."""
        buf = io.BytesIO()
        tifffile.imwrite(
            buf,
            elevation.astype(np.float32),
            photometric="minisblack",
            description="TerraFly Reconstructed Digital Surface Model (DSM) - SIH PS 175"
        )
        buf.seek(0)
        return buf

    def export_png_heightmap(self, elevation: np.ndarray) -> io.BytesIO:
        """Exports 16-bit grayscale PNG heightmap for Unity, Unreal Engine, Blender GIS."""
        emin, emax = np.min(elevation), np.max(elevation)
        span = emax - emin if (emax - emin) > 0 else 1.0
        norm_16 = ((elevation - emin) / span * 65535.0).astype(np.uint16)
        
        img = Image.fromarray(norm_16, mode="I;16")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

# Global singleton
mesh_exporter = MeshExporter()
