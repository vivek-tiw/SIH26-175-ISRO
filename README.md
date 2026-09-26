# TerraFly 🚁
### 3D Elevation Mapping from Single-View Optical Satellite & Drone Imagery
**Smart India Hackathon (SIH) — Problem Statement 175 (PS 175)**

---

## 🌟 Overview
**TerraFly** is a full-stack photogrammetric intelligence tool that reconstructs high-fidelity **3D Digital Surface Models (DSM)** from a single RGB 2D optical image (satellite, aerial, or drone).

### Key Highlights:
1. **Relative Digital Surface Model (rDSM)**: Extracted automatically for any standard optical imagery (PNG, JPG, WebP) using **Depth Anything V2** foundation models and high-speed Shape-from-Shading photogrammetry.
2. **Absolute-Height Metric DSM (mDSM)**: Converts relative disparity into metric elevation (meters Above Sea Level) when GeoTIFF metadata, Ground Control Points (GCPs), or a reference DEM is provided ($H_{metric}(x,y) = \alpha \cdot \text{rDSM}(x,y) + \beta$).
3. **Interactive 3D WebGL Terrain Explorer**: Built with Three.js, featuring real-time hypsometric colormaps (Turbo, Viridis, Terrain), shaded relief (Horn 1981 Hillshading with dynamic solar azimuth/altitude control), water flood simulation, and an interactive **TerraFly First-Person Drone Flight Simulator**!
4. **Geodetic Accuracy Benchmark Engine**: Evaluates reconstructed 3D elevations against ground-truth DEMs (Copernicus DEM GLO-30, Tandem-X, LiDAR) reporting:
   - **RMSE** (Root Mean Square Error)
   - **MAE** (Mean Absolute Error)
   - **NMAD** (Normalized Median Absolute Deviation — Geodetic Standard)
   - **LE90 & LE95** (Linear Error at 90% and 95% Confidence)
   - **Pearson Correlation ($r$) & SSIM**
   - **Survey Grade Certification Badge**
5. **Interactive 3D Cross-Section Slicing Tool**: Click two points on the 3D terrain to extract an elevation profile transect with side-by-side predicted vs reference DEM comparison.
6. **Multi-Format Export Hub**:
   - 3D Mesh: **Wavefront OBJ** (.obj + .mtl with UV texture projection)
   - 3D Printing / CAD: **Binary STL** (.stl)
   - Point Cloud: **PLY** (.ply with RGB vertex colors)
   - GIS Raster: **32-Bit Floating Point GeoTIFF** (.tif)
   - Game Engine: **16-Bit Grayscale Heightmap PNG**
   - Accuracy Audit: **CSV Verification Log**

---

## 🛠️ Tech Stack
- **Backend**: Python 3.13, FastAPI, Uvicorn, NumPy, SciPy, Pillow, Tifffile, PyTorch (MPS/CUDA Accelerated), HuggingFace Transformers
- **AI Models**: Depth Anything V2 (`Depth-Anything-V2-Small-hf`), MiDaS DPT Large, High-Fidelity Photogrammetric Computer Vision Fallback
- **Frontend**: HTML5, Vanilla CSS3 (Deep Space Glassmorphism Design System), Three.js (WebGL), Canvas 2D, JetBrains Mono & Outfit typography

---

## 🚀 Quickstart Guide

### 1. Run Server
```bash
python3 run.py
```
The server will start on: **`http://localhost:8000`**

### 2. Built-in Benchmark Presets (1-Click Demo)
- 🏔️ **Himalayan Glacial Ridge (Nanda Devi)** (Elevation 3,400m - 6,850m)
- 🏙️ **High-Density CBD (Bengaluru)** (Elevation 880m - 985m)
- 🌋 **Volcanic Caldera (Barren Island)** (Elevation 0m - 354m)
- 🏞️ **Western Ghats Deep River Gorge** (Elevation 420m - 1,260m)
- ⛏️ **Open-Cast Terraced Mine (Jharia)** (Elevation 150m - 310m)

---

## 🎮 3D Navigation Controls
- **Orbit Mode**: Left-click + Drag to rotate, Right-click to pan, Scroll to zoom.
- **Top-Down Ortho Mode**: Instant planimetric orthophoto projection.
- **TerraFly Drone Mode**:
  - `W` / `S` : Thrust Forward / Backward
  - `A` / `D` : Yaw Left / Right
  - `Space` / `Shift` : Ascend / Descend
  - `Esc` : Exit Drone Mode
- **Slice Transect Tool**: Click 'Slice Transect', then click two points on the 3D surface to generate real-time elevation profile slice.

---

## 📊 SIH PS 175 Geodetic Formulas Implemented
- $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (Z_{\text{pred}, i} - Z_{\text{ref}, i})^2}$
- $\text{NMAD} = 1.4826 \cdot \text{median}\left(|\Delta h_i - \text{median}(\Delta h)|\right)$
- $\text{LE90} = 90\text{th percentile of } |\Delta h|$
- $\text{Hillshade} = \sin(\text{Alt}) \cdot \cos(\text{Slope}) + \cos(\text{Alt}) \cdot \sin(\text{Slope}) \cdot \cos(\text{Az} - \text{Aspect})$
