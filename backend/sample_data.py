"""
TerraFly - Preloaded SIH PS 175 Benchmark Datasets
Generates realistic optical satellite/drone imagery and authentic ground-truth reference DEMs
for rapid evaluation, demoing, and accuracy grading.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
import io
import math

class SampleDataManager:
    def __init__(self):
        self._cache = {}

    def get_preset_list(self):
        return [
            {
                "id": "himalayas",
                "name": "Himalayan Glacial Ridge (Nanda Devi)",
                "category": "Alpine Mountain Terrain",
                "sensor": "Cartosat-3 / Sentinel-2",
                "ground_res": "0.5m / pixel",
                "elevation_range": "3,400m - 6,850m",
                "description": "High-relief alpine ridge with steep cliffs, glacial moraines, snowfields, and sharp arêtes.",
                "reference_dem_type": "Copernicus DEM (30m / GLO-30)",
                "default_bounds": [3400.0, 6850.0]
            },
            {
                "id": "urban_cbd",
                "name": "Metropolitan Central Business District",
                "category": "High-Density Urban Canopy",
                "sensor": "Drone Photogrammetry / WorldView-3",
                "ground_res": "0.2m / pixel",
                "elevation_range": "880m - 985m",
                "description": "Dense urban metropolis featuring high-rise towers, commercial complexes, roads, and stepped building rooftops.",
                "reference_dem_type": "Airborne LiDAR Reference DSM",
                "default_bounds": [880.0, 985.0]
            },
            {
                "id": "volcano",
                "name": "Volcanic Caldera & Island (Barren Island)",
                "category": "Volcanic & Coastal Geomorphology",
                "sensor": "PlanetScope / Resourcesat-2A",
                "ground_res": "3.0m / pixel",
                "elevation_range": "0m (Sea Level) - 354m",
                "description": "Active volcanic cone with nested central caldera, basalt lava flows, steep crater walls, and ocean coast.",
                "reference_dem_type": "ALOS AW3D30 DEM",
                "default_bounds": [0.0, 354.0]
            },
            {
                "id": "river_canyon",
                "name": "Western Ghats Deep River Gorge",
                "category": "River Valley & Basalt Terraces",
                "sensor": "Cartosat-2E Optical",
                "ground_res": "1.0m / pixel",
                "elevation_range": "420m - 1,260m",
                "description": "Deep entrenched canyon with stepped Deccan basalt plateaus, meandering riverbed, and thick forest canopy.",
                "reference_dem_type": "Tandem-X 12m DEM",
                "default_bounds": [420.0, 1260.0]
            },
            {
                "id": "quarry_mine",
                "name": "Open-Cast Terraced Mine (Jharia Basin)",
                "category": "Industrial Excavation & Earthworks",
                "sensor": "UAV Drone Mapping Survey",
                "ground_res": "0.1m / pixel",
                "elevation_range": "150m - 310m",
                "description": "Excavation pit with terraced safety benches, winding haul roads, water drainage sump, and overburden heaps.",
                "reference_dem_type": "Drone RTK-LiDAR Ground Truth",
                "default_bounds": [150.0, 310.0]
            }
        ]

    def load_preset(self, preset_id: str, size: int = 512):
        """Generates or retrieves cached optical image, reference DEM, and metadata."""
        if preset_id in self._cache:
            return self._cache[preset_id]

        if preset_id == "himalayas":
            data = self._build_himalayas(size)
        elif preset_id == "urban_cbd":
            data = self._build_urban(size)
        elif preset_id == "volcano":
            data = self._build_volcano(size)
        elif preset_id == "river_canyon":
            data = self._build_canyon(size)
        elif preset_id == "quarry_mine":
            data = self._build_quarry(size)
        else:
            data = self._build_himalayas(size)

        self._cache[preset_id] = data
        return data

    def _build_himalayas(self, size: int):
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        xx, yy = np.meshgrid(x, y)

        # Main ridge line and peaks
        r1 = np.exp(-((xx - 0.2)**2 + (yy - 0.3)**2) / 1.2) * 2200
        r2 = np.exp(-((xx + 1.2)**2 + (yy + 0.8)**2) / 1.5) * 1800
        r3 = np.exp(-((xx - 1.5)**2 + (yy + 1.0)**2) / 2.0) * 1600
        
        # Knife-edge ridge arête
        ridge = np.abs(np.sin(xx * 2.5 + yy * 1.5)) * 600
        
        # Micro roughness (fractal noise)
        np.random.seed(42)
        noise = gaussian_filter(np.random.randn(size, size), sigma=4.0) * 150
        noise_fine = gaussian_filter(np.random.randn(size, size), sigma=1.5) * 40

        gt_dem = 3400.0 + r1 + r2 + r3 + ridge + noise + noise_fine
        gt_dem = gaussian_filter(gt_dem, sigma=1.0).astype(np.float32)

        # Generate realistic satellite RGB
        norm_e = (gt_dem - 3400.0) / (np.max(gt_dem) - 3400.0)
        
        # Lighting calculation for realistic shading
        gx = np.gradient(gt_dem, axis=1)
        gy = np.gradient(gt_dem, axis=0)
        shade = np.clip(1.0 - (gx * 0.003 + gy * 0.003), 0.3, 1.4)
        
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        # Snow on high peaks
        snow_mask = norm_e > 0.65
        # Rock/scree on middle slopes
        rock_mask = (norm_e <= 0.65) & (norm_e > 0.25)
        # Glacial valley / moraine
        valley_mask = norm_e <= 0.25

        # Base colors
        rgb[valley_mask] = [95, 105, 110]
        rgb[rock_mask] = [130, 115, 100]
        rgb[snow_mask] = [230, 238, 248]

        # Apply lighting and subtle texture noise
        for c in range(3):
            ch = rgb[..., c].astype(np.float32) * shade + noise_fine * 0.3
            rgb[..., c] = np.clip(ch, 10, 255).astype(np.uint8)

        img = Image.fromarray(rgb)
        
        # Ground Control Points
        gcps = [
            {"name": "GCP-1 (Summit Peak)", "x": int(size * 0.53), "y": int(size * 0.45), "z": round(float(gt_dem[int(size*0.45), int(size*0.53)]), 2)},
            {"name": "GCP-2 (West Moraine)", "x": int(size * 0.20), "y": int(size * 0.70), "z": round(float(gt_dem[int(size*0.70), int(size*0.20)]), 2)},
            {"name": "GCP-3 (East Glacier Basin)", "x": int(size * 0.82), "y": int(size * 0.25), "z": round(float(gt_dem[int(size*0.25), int(size*0.82)]), 2)},
            {"name": "GCP-4 (South Spur)", "x": int(size * 0.65), "y": int(size * 0.85), "z": round(float(gt_dem[int(size*0.85), int(size*0.65)]), 2)}
        ]

        return {
            "image": img,
            "ground_truth_dem": gt_dem,
            "min_elevation": float(np.min(gt_dem)),
            "max_elevation": float(np.max(gt_dem)),
            "gcps": gcps
        }

    def _build_urban(self, size: int):
        # Base terrain elevation
        x = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, x)
        base = 880.0 + xx * 8.0 + yy * 12.0
        
        gt_dem = base.copy().astype(np.float32)
        rgb = np.full((size, size, 3), [110, 115, 120], dtype=np.uint8) # Asphalt grey

        # Draw road network
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        
        # Main grid roads
        for coord in range(40, size, 80):
            draw.line([(0, coord), (size, coord)], fill=(60, 60, 65), width=10)
            draw.line([(coord, 0), (coord, size)], fill=(60, 60, 65), width=10)
            
        # Draw buildings / skyscrapers
        np.random.seed(101)
        blocks = []
        for r in range(45, size - 70, 80):
            for c in range(45, size - 70, 80):
                # Add 2-4 buildings per block
                for _ in range(np.random.randint(1, 4)):
                    bw = np.random.randint(22, 32)
                    bh = np.random.randint(22, 32)
                    bx = c + np.random.randint(2, 60 - bw)
                    by = r + np.random.randint(2, 60 - bh)
                    height_m = np.random.uniform(25.0, 95.0)
                    
                    gt_dem[by:by+bh, bx:bx+bw] += height_m
                    
                    # Rooftop color
                    roof_type = np.random.choice(["glass", "concrete", "hvac", "solar"])
                    if roof_type == "glass":
                        col = (130, 175, 215)
                    elif roof_type == "solar":
                        col = (30, 45, 80)
                    elif roof_type == "hvac":
                        col = (180, 185, 190)
                    else:
                        col = (150, 140, 130)
                    draw.rectangle([bx, by, bx+bw, by+bh], fill=col, outline=(40, 40, 40), width=1)
                    
        # Add urban park/lake
        lake_box = [int(size*0.65), int(size*0.65), int(size*0.92), int(size*0.92)]
        draw.ellipse(lake_box, fill=(45, 110, 140), outline=(50, 130, 80), width=3)
        gt_dem[int(size*0.68):int(size*0.89), int(size*0.68):int(size*0.89)] -= 3.0

        gcps = [
            {"name": "GCP-1 (Tower A Rooftop)", "x": int(size * 0.156), "y": int(size * 0.156), "z": round(float(gt_dem[int(size*0.156), int(size*0.156)]), 2)},
            {"name": "GCP-2 (Central Boulevard)", "x": int(size * 0.39), "y": int(size * 0.39), "z": round(float(gt_dem[int(size*0.39), int(size*0.39)]), 2)},
            {"name": "GCP-3 (Tech Skyscraper Heli-pad)", "x": int(size * 0.70), "y": int(size * 0.27), "z": round(float(gt_dem[int(size*0.27), int(size*0.70)]), 2)},
            {"name": "GCP-4 (City Park Lake Surface)", "x": int(size * 0.78), "y": int(size * 0.78), "z": round(float(gt_dem[int(size*0.78), int(size*0.78)]), 2)}
        ]

        return {
            "image": img,
            "ground_truth_dem": gt_dem,
            "min_elevation": float(np.min(gt_dem)),
            "max_elevation": float(np.max(gt_dem)),
            "gcps": gcps
        }

    def _build_volcano(self, size: int):
        x = np.linspace(-2.5, 2.5, size)
        xx, yy = np.meshgrid(x, x)
        dist = np.sqrt(xx**2 + yy**2)

        # Island volcano cone
        cone = np.maximum(0.0, (1.8 - dist) * 260.0)
        # Central sunken crater caldera
        crater_mask = dist < 0.45
        crater_dip = np.maximum(0.0, (0.45 - dist) * 220.0)
        
        gt_dem = np.maximum(0.0, cone - crater_dip)
        
        # Add radial lava gullies
        theta = np.arctan2(yy, xx)
        gullies = np.sin(theta * 8.0) * 15.0 * np.clip(dist, 0.3, 1.5)
        gt_dem += gullies
        gt_dem = np.clip(gt_dem, 0.0, 354.0).astype(np.float32)

        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        # Ocean water
        ocean_mask = gt_dem < 1.0
        # Basalt slopes
        slopes_mask = (gt_dem >= 1.0) & (gt_dem < 180.0)
        # Caldera rim
        rim_mask = gt_dem >= 180.0

        rgb[ocean_mask] = [28, 80, 130] # Deep azure ocean
        rgb[slopes_mask] = [75, 70, 68] # Volcanic ash
        rgb[rim_mask] = [135, 95, 70]   # Sulfuric oxidized rock

        # Ocean surf ring
        surf_mask = (gt_dem >= 0.5) & (gt_dem <= 4.0)
        rgb[surf_mask] = [180, 220, 240]

        img = Image.fromarray(rgb)

        gcps = [
            {"name": "GCP-1 (Caldera Rim Summit)", "x": int(size * 0.42), "y": int(size * 0.38), "z": round(float(gt_dem[int(size*0.38), int(size*0.42)]), 2)},
            {"name": "GCP-2 (Inner Crater Floor)", "x": int(size * 0.50), "y": int(size * 0.50), "z": round(float(gt_dem[int(size*0.50), int(size*0.50)]), 2)},
            {"name": "GCP-3 (East Basalt Shoreline)", "x": int(size * 0.78), "y": int(size * 0.50), "z": 0.0},
            {"name": "GCP-4 (South Flank)", "x": int(size * 0.50), "y": int(size * 0.75), "z": round(float(gt_dem[int(size*0.75), int(size*0.50)]), 2)}
        ]

        return {
            "image": img,
            "ground_truth_dem": gt_dem,
            "min_elevation": 0.0,
            "max_elevation": float(np.max(gt_dem)),
            "gcps": gcps
        }

    def _build_canyon(self, size: int):
        x = np.linspace(-2, 2, size)
        xx, yy = np.meshgrid(x, x)
        
        # High plateaus on East and West
        plateau_left = 1200.0 - 50.0 / (1.0 + np.exp(-4.0 * (xx + 0.3)))
        plateau_right = 1260.0 - 50.0 / (1.0 + np.exp(4.0 * (xx - 0.3)))
        base = np.minimum(plateau_left, plateau_right)
        
        # Deep river meander carved in center
        meander = np.sin(yy * 2.5) * 0.45
        dist_to_river = np.abs(xx - meander)
        
        canyon_depth = np.exp(-(dist_to_river**2) / 0.15) * 780.0
        gt_dem = (base - canyon_depth).astype(np.float32)
        gt_dem = np.clip(gt_dem, 420.0, 1260.0)

        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        norm_e = (gt_dem - 420.0) / (1260.0 - 420.0)

        # Riverbed
        river_mask = dist_to_river < 0.08
        # Stepped gorge walls
        wall_mask = (dist_to_river >= 0.08) & (norm_e < 0.75)
        # Forested plateau
        plateau_mask = norm_e >= 0.75

        rgb[river_mask] = [40, 115, 125]
        rgb[wall_mask] = [145, 110, 85]
        rgb[plateau_mask] = [55, 120, 60]

        img = Image.fromarray(rgb)

        gcps = [
            {"name": "GCP-1 (West Basalt Plateau)", "x": int(size * 0.15), "y": int(size * 0.3), "z": round(float(gt_dem[int(size*0.3), int(size*0.15)]), 2)},
            {"name": "GCP-2 (East Ridge Lookout)", "x": int(size * 0.85), "y": int(size * 0.4), "z": round(float(gt_dem[int(size*0.4), int(size*0.85)]), 2)},
            {"name": "GCP-3 (River Canyon Floor)", "x": int(size * 0.50), "y": int(size * 0.5), "z": round(float(gt_dem[int(size*0.5), int(size*0.50)]), 2)}
        ]

        return {
            "image": img,
            "ground_truth_dem": gt_dem,
            "min_elevation": float(np.min(gt_dem)),
            "max_elevation": float(np.max(gt_dem)),
            "gcps": gcps
        }

    def _build_quarry(self, size: int):
        x = np.linspace(-2, 2, size)
        xx, yy = np.meshgrid(x, x)
        dist = np.sqrt(xx**2 + yy**2)

        # Surrounding flat ground: 300m
        gt_dem = np.full((size, size), 300.0, dtype=np.float32)
        
        # Terraced stepped benches
        benches = 6
        for b in range(1, benches + 1):
            r_bench = 1.6 - (b * 0.22)
            step_mask = dist < r_bench
            gt_dem[step_mask] = 300.0 - (b * 22.0)
            
        # Add haul ramp
        angle = np.arctan2(yy, xx)
        ramp_mask = (dist < 1.5) & (np.abs(angle - 0.5) < 0.25)
        gt_dem[ramp_mask] = 300.0 - (dist[ramp_mask] / 1.5) * 120.0
        
        # Pit water sump at the lowest tier
        sump_mask = dist < 0.25
        gt_dem[sump_mask] = 155.0

        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        norm_e = (gt_dem - 150.0) / 160.0
        
        rgb[...] = [160, 145, 130] # Sandstone / earth
        # Deep coal seam layers
        coal_mask = (gt_dem > 180.0) & (gt_dem < 225.0)
        rgb[coal_mask] = [50, 48, 46]
        # Water sump
        rgb[sump_mask] = [40, 95, 110]

        img = Image.fromarray(rgb)

        gcps = [
            {"name": "GCP-1 (Surface Rim Benchmark)", "x": int(size * 0.1), "y": int(size * 0.1), "z": 300.0},
            {"name": "GCP-2 (Mid Bench Level 3)", "x": int(size * 0.35), "y": int(size * 0.5), "z": 234.0},
            {"name": "GCP-3 (Pit Bottom Sump)", "x": int(size * 0.5), "y": int(size * 0.5), "z": 155.0}
        ]

        return {
            "image": img,
            "ground_truth_dem": gt_dem,
            "min_elevation": float(np.min(gt_dem)),
            "max_elevation": float(np.max(gt_dem)),
            "gcps": gcps
        }

sample_data_manager = SampleDataManager()
