"""
TerraFly - 3D Elevation Utilities
Provides colormapping, hillshading, slope/aspect calculation, contouring, and profile slicing.
"""

import numpy as np
from PIL import Image
import io
import base64
from scipy.ndimage import gaussian_filter, sobel

def normalize_array(arr: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Normalizes array to [0, 1] range."""
    if vmin is None:
        vmin = np.nanmin(arr)
    if vmax is None:
        vmax = np.nanmax(arr)
    if vmax - vmin == 0:
        return np.zeros_like(arr, dtype=np.float32)
    norm = (arr - vmin) / (vmax - vmin)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)

def compute_hillshade(elevation: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    """
    Calculates 2D shaded relief (hillshade) from elevation matrix.
    Standard cartographic hillshade algorithm (Horn 1981).
    """
    zenith_rad = np.radians(90.0 - altitude_deg)
    azimuth_rad = np.radians((360.0 - azimuth_deg + 90.0) % 360.0)
    
    # Compute gradients using Sobel operators
    dx = sobel(elevation, axis=1) / 8.0 * z_factor
    dy = sobel(elevation, axis=0) / 8.0 * z_factor
    
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(dy, -dx)
    aspect_rad[aspect_rad < 0] += 2 * np.pi
    
    # Hillshade equation
    shaded = np.sin(zenith_rad) * np.cos(slope_rad) + np.cos(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    shaded = np.clip(shaded, 0.0, 1.0)
    return (shaded * 255).astype(np.uint8)

def compute_slope_aspect(elevation: np.ndarray, cell_size: float = 1.0):
    """Computes slope (in degrees) and aspect (in degrees)."""
    dx = sobel(elevation, axis=1) / (8.0 * cell_size)
    dy = sobel(elevation, axis=0) / (8.0 * cell_size)
    
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    aspect_rad = np.arctan2(dy, -dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (450.0 - aspect_deg) % 360.0 # Standard North = 0 deg
    
    return slope_deg, aspect_deg

# Built-in Colormaps (Turbo, Viridis, Terrain, Magma, Rainbow)
COLORMAPS = {
    "turbo": [
        (0.00, [48, 18, 59]),
        (0.12, [70, 134, 251]),
        (0.25, [27, 229, 181]),
        (0.38, [132, 254, 76]),
        (0.50, [228, 237, 52]),
        (0.62, [254, 155, 45]),
        (0.75, [239, 66, 23]),
        (0.88, [186, 12, 13]),
        (1.00, [122, 4, 3])
    ],
    "terrain": [
        (0.00, [51, 102, 204]),   # Deep water
        (0.15, [102, 178, 255]), # Shallow water
        (0.20, [238, 214, 175]), # Sandy beach
        (0.35, [100, 160, 80]),  # Lowland green
        (0.55, [160, 140, 80]),  # Highlands / plateau
        (0.75, [140, 100, 70]),  # Mountain rocky brown
        (0.90, [190, 190, 190]), # Alpine rock/grey
        (1.00, [255, 255, 255])  # Snow peak
    ],
    "viridis": [
        (0.00, [68, 1, 84]),
        (0.25, [59, 82, 139]),
        (0.50, [33, 145, 140]),
        (0.75, [94, 201, 98]),
        (1.00, [253, 231, 37])
    ],
    "magma": [
        (0.00, [0, 0, 4]),
        (0.25, [81, 18, 124]),
        (0.50, [182, 54, 121]),
        (0.75, [251, 136, 97]),
        (1.00, [252, 253, 191])
    ],
    "rainbow": [
        (0.00, [0, 0, 255]),
        (0.20, [0, 255, 255]),
        (0.40, [0, 255, 0]),
        (0.60, [255, 255, 0]),
        (0.80, [255, 128, 0]),
        (1.00, [255, 0, 0])
    ]
}

def apply_colormap(normalized_data: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """Applies a high-contrast colormap to [0, 1] normalized array, returns RGB uint8 image."""
    cmap = COLORMAPS.get(cmap_name.lower(), COLORMAPS["turbo"])
    h, w = normalized_data.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    positions = [p for p, _ in cmap]
    colors = [np.array(c, dtype=np.float32) for _, c in cmap]
    
    flat_data = normalized_data.flatten()
    r = np.interp(flat_data, positions, [c[0] for c in colors])
    g = np.interp(flat_data, positions, [c[1] for c in colors])
    b = np.interp(flat_data, positions, [c[2] for c in colors])
    
    rgb[..., 0] = r.reshape((h, w)).astype(np.uint8)
    rgb[..., 1] = g.reshape((h, w)).astype(np.uint8)
    rgb[..., 2] = b.reshape((h, w)).astype(np.uint8)
    return rgb

def array_to_base64_image(arr: np.ndarray, format: str = "PNG") -> str:
    """Converts a numpy RGB or Grayscale image into a data:image/png;base64 string."""
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    elif arr.ndim == 3 and arr.shape[2] == 4:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    else:
        raise ValueError("Unsupported array shape for image conversion")
        
    buf = io.BytesIO()
    img.save(buf, format=format, optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"

def extract_cross_section_profile(elevation: np.ndarray, p1: tuple, p2: tuple, num_samples: int = 150) -> dict:
    """
    Extracts elevation profile slice between (x1, y1) and (x2, y2) in pixel or normalized coords.
    Returns array of {distance_ratio, elevation, x, y}.
    """
    h, w = elevation.shape
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    
    # Handle normalized coordinates [0, 1]
    if x1 <= 1.0 and x2 <= 1.0 and y1 <= 1.0 and y2 <= 1.0:
        px1, py1 = int(round(x1 * (w - 1))), int(round(y1 * (h - 1)))
        px2, py2 = int(round(x2 * (w - 1))), int(round(y2 * (h - 1)))
    else:
        px1, py1 = int(round(np.clip(x1, 0, w - 1))), int(round(np.clip(y1, 0, h - 1)))
        px2, py2 = int(round(np.clip(x2, 0, w - 1))), int(round(np.clip(y2, 0, h - 1)))
        
    xs = np.linspace(px1, px2, num_samples)
    ys = np.linspace(py1, py2, num_samples)
    
    profile = []
    total_dist = float(np.sqrt((px2 - px1)**2 + (py2 - py1)**2))
    
    for i in range(num_samples):
        cx, cy = int(round(float(xs[i]))), int(round(float(ys[i])))
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        z = float(elevation[cy, cx])
        dist_pct = float(i / (num_samples - 1))
        profile.append({
            "step": int(i),
            "dist_pct": round(float(dist_pct * 100), 1),
            "dist_px": round(float(dist_pct * total_dist), 1),
            "x": int(cx),
            "y": int(cy),
            "elevation": round(float(z), 2)
        })
        
    return {
        "start": {"x": int(px1), "y": int(py1)},
        "end": {"x": int(px2), "y": int(py2)},
        "length_px": round(float(total_dist), 1),
        "min_elevation": round(float(np.min([p["elevation"] for p in profile])), 2),
        "max_elevation": round(float(np.max([p["elevation"] for p in profile])), 2),
        "profile": profile
    }
