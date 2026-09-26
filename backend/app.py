"""
TerraFly - Main FastAPI Application Server
Smart India Hackathon - PS 175: 3D Elevation Mapping from Single-View Optical Imagery
"""

import os
import uuid
import json
import logging
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .depth_engine import depth_engine
from .elevation_engine import elevation_engine
from .mesh_exporter import mesh_exporter
from .sample_data import sample_data_manager
from .utils import (
    compute_hillshade,
    compute_slope_aspect,
    apply_colormap,
    array_to_base64_image,
    normalize_array,
    extract_cross_section_profile
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("terrafly.server")

app = FastAPI(
    title="TerraFly - 3D Elevation Mapping Engine",
    description="Single-View Optical Satellite & Drone Imagery to 3D Digital Surface Model Reconstruction",
    version="1.0.0"
)

# Enable CORS for local dev and embedded browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage for active reconstructions
SESSIONS: Dict[str, Dict[str, Any]] = {}

class ProfileRequest(BaseModel):
    session_id: str
    p1: List[float] # [x1, y1] (0-1 normalized or pixel coords)
    p2: List[float] # [x2, y2]
    num_samples: Optional[int] = 120

class CalibrateRequest(BaseModel):
    session_id: str
    min_elevation: float
    max_elevation: float
    gcps: Optional[List[Dict[str, float]]] = None

@app.get("/api/status")
async def get_system_status():
    return {
        "status": "online",
        "system": "TerraFly Photogrammetric Engine",
        "hackathon": "Smart India Hackathon (PS 175)",
        "hardware_device": depth_engine.device,
        "is_gpu_mps_accelerated": depth_engine.device in ["mps", "cuda"],
        "active_model": depth_engine.model_name or "depth-anything-v2 (lazy-ready)",
        "cached_sessions": len(SESSIONS)
    }

@app.get("/api/presets")
async def list_presets():
    return sample_data_manager.get_preset_list()

@app.post("/api/presets/load/{preset_id}")
async def load_preset(preset_id: str, model_choice: str = "depth-anything-v2"):
    try:
        preset_info = next((p for p in sample_data_manager.get_preset_list() if p["id"] == preset_id), None)
        if not preset_info:
            raise HTTPException(status_code=404, detail="Preset not found")

        data = sample_data_manager.load_preset(preset_id, size=512)
        rgb_img = data["image"]
        ref_dem = data["ground_truth_dem"]
        gcps = data.get("gcps", [])
        
        # 1. Predict Relative DSM (rDSM)
        rdsm = depth_engine.predict_rdsm(rgb_img, model_type=model_choice)
        
        # 2. Calibrate to Absolute Metric DSM (mDSM) using Ground Truth bounds / GCPs
        mdsm, calib_stats = elevation_engine.calibrate_absolute_dsm(
            rdsm=rdsm,
            min_elev=preset_info["default_bounds"][0],
            max_elev=preset_info["default_bounds"][1],
            reference_dem=ref_dem,
            gcps=gcps
        )
        
        # 3. Compute Accuracy Metrics vs Reference DEM
        accuracy_metrics, diff_grid = elevation_engine.evaluate_accuracy(mdsm, ref_dem)
        
        # 4. Generate Visualizations
        session_id = str(uuid.uuid4())
        
        # Generate base64 textures
        norm_mdsm = normalize_array(mdsm)
        turbo_img = apply_colormap(norm_mdsm, "turbo")
        terrain_img = apply_colormap(norm_mdsm, "terrain")
        viridis_img = apply_colormap(norm_mdsm, "viridis")
        hillshade_arr = compute_hillshade(mdsm, azimuth_deg=315.0, altitude_deg=45.0, z_factor=2.0)
        
        # Reference & Difference visual maps
        ref_norm = normalize_array(ref_dem)
        ref_colormap = apply_colormap(ref_norm, "terrain")
        
        diff_abs = np.abs(diff_grid)
        diff_norm = normalize_array(diff_abs, vmin=0.0, vmax=max(5.0, float(np.percentile(diff_abs, 98))))
        diff_colormap = apply_colormap(diff_norm, "magma")

        # Slope & Aspect
        slope_deg, aspect_deg = compute_slope_aspect(mdsm)

        # Compute Elevation & Slope Histograms for Right Sidebar
        elev_counts, elev_bins = np.histogram(mdsm, bins=35)
        elev_hist = [
            {"bin": round(float(elev_bins[i]), 1), "count": int(elev_counts[i])}
            for i in range(len(elev_counts))
        ]

        slope_counts, slope_bins = np.histogram(slope_deg, bins=35, range=(0, 90))
        slope_hist = [
            {"bin": round(float(slope_bins[i]), 1), "count": int(slope_counts[i])}
            for i in range(len(slope_counts))
        ]

        # Dataset Geo-coordinates
        geo_coord_map = {
            "himalayas": {"lat": "30.3753°N", "lon": "79.9702°E", "location": "Nanda Devi Basin, Uttarakhand", "crs": "Cartosat-3 (EPSG:4326)"},
            "urban_cbd": {"lat": "26.9124°N", "lon": "75.7873°E", "location": "Sector 04 Urban Canopy", "crs": "GeoTIFF - EPSG:4326"},
            "volcano": {"lat": "12.2783°N", "lon": "93.8583°E", "location": "Barren Island Caldera, Andaman Sea", "crs": "Resourcesat-2A (EPSG:32646)"},
            "river_canyon": {"lat": "17.4042°N", "lon": "73.7431°E", "location": "Koyna Valley Gorge, Maharashtra", "crs": "Cartosat-2E (EPSG:32643)"},
            "quarry_mine": {"lat": "23.7423°N", "lon": "86.4150°E", "location": "Jharia Basin Open-Cast Mine", "crs": "UAV RTK (EPSG:32645)"}
        }
        geo_info = geo_coord_map.get(preset_id, {"lat": "26.9124°N", "lon": "75.7873°E", "location": "Sector 04", "crs": "GeoTIFF - EPSG:4326"})

        # Store session in memory
        SESSIONS[session_id] = {
            "session_id": session_id,
            "preset_id": preset_id,
            "name": preset_info["name"],
            "rgb_img": rgb_img,
            "rdsm": rdsm,
            "mdsm": mdsm,
            "ref_dem": ref_dem,
            "diff_grid": diff_grid,
            "calib_stats": calib_stats,
            "accuracy_metrics": accuracy_metrics,
            "gcps": gcps,
            "bounds": preset_info["default_bounds"],
            "grid_shape": list(mdsm.shape),
            "geo_info": geo_info
        }

        # Subsample elevation grid for ultra-fast Three.js 3D plane mesh geometry (e.g., 256x256 or 128x128)
        step = 2 # 512 -> 256
        mesh_grid = mdsm[::step, ::step]
        grid_flat = mesh_grid.flatten().round(2).tolist()

        return {
            "session_id": session_id,
            "preset": preset_info,
            "is_geotiff": True,
            "is_calibrated": True,
            "geo_info": geo_info,
            "grid_dimensions": {"width": mdsm.shape[1], "height": mdsm.shape[0]},
            "mesh_grid_dimensions": {"width": mesh_grid.shape[1], "height": mesh_grid.shape[0]},
            "elevation_stats": {
                "min": round(float(np.min(mdsm)), 2),
                "max": round(float(np.max(mdsm)), 2),
                "mean": round(float(np.mean(mdsm)), 2),
                "unit": "meters"
            },
            "calibration_stats": calib_stats,
            "accuracy_metrics": accuracy_metrics,
            "gcps": gcps,
            "elevation_histogram": elev_hist,
            "slope_histogram": slope_hist,
            "elevation_grid_sample": grid_flat,
            "textures": {
                "rgb": array_to_base64_image(np.array(rgb_img)),
                "rdsm": array_to_base64_image((rdsm * 255).astype(np.uint8)),
                "turbo": array_to_base64_image(turbo_img),
                "terrain": array_to_base64_image(terrain_img),
                "viridis": array_to_base64_image(viridis_img),
                "hillshade": array_to_base64_image(hillshade_arr),
                "reference_dem": array_to_base64_image(ref_colormap),
                "diff_error": array_to_base64_image(diff_colormap)
            }
        }
    except Exception as e:
        logger.error(f"Failed to load preset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reconstruct")
async def reconstruct_image(
    file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None),
    model_choice: str = Form("depth-anything-v2"),
    calibration_mode: str = Form("auto"), # 'relative_only', 'auto', 'bounds', 'gcp'
    min_elev: Optional[float] = Form(None),
    max_elev: Optional[float] = Form(None),
    gcps_json: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        filename = file.filename or "upload.png"
        
        # Check GeoTIFF metadata
        geo_meta = elevation_engine.extract_geotiff_metadata(contents)
        
        # Load image via PIL
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            # If multi-band GeoTIFF, use tifffile
            with tifffile.TiffFile(io.BytesIO(contents)) as tif:
                arr = tif.pages[0].asarray()
                if arr.ndim == 2:
                    norm = normalize_array(arr)
                    image = Image.fromarray((norm * 255).astype(np.uint8)).convert("RGB")
                elif arr.ndim == 3 and arr.shape[2] >= 3:
                    image = Image.fromarray(arr[..., :3].astype(np.uint8))
                else:
                    norm = normalize_array(arr[0])
                    image = Image.fromarray((norm * 255).astype(np.uint8)).convert("RGB")

        # Resize to max 512 for responsive real-time 3D processing
        max_dim = 512
        orig_w, orig_h = image.size
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        w, h = image.size

        # 1. Relative DSM
        rdsm = depth_engine.predict_rdsm(image, model_type=model_choice)

        # Parse reference file if supplied
        ref_dem = None
        if reference_file is not None:
            ref_bytes = await reference_file.read()
            try:
                with tifffile.TiffFile(io.BytesIO(ref_bytes)) as tif:
                    ref_dem = tif.pages[0].asarray().astype(np.float32)
            except Exception:
                try:
                    ref_pil = Image.open(io.BytesIO(ref_bytes)).convert("L")
                    ref_dem = np.array(ref_pil, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Could not parse reference DEM: {e}")

        # Parse GCPs if provided
        gcps = []
        if gcps_json:
            try:
                gcps = json.loads(gcps_json)
            except Exception:
                pass

        # Calibration determination
        is_calibrated = False
        default_min = min_elev if min_elev is not None else 100.0
        default_max = max_elev if max_elev is not None else 500.0

        if geo_meta.get("is_dem_raster") and geo_meta.get("min_elevation") is not None:
            default_min = geo_meta["min_elevation"]
            default_max = geo_meta["max_elevation"]
            is_calibrated = True
        elif min_elev is not None and max_elev is not None:
            is_calibrated = True
        elif ref_dem is not None or (gcps and len(gcps) >= 2):
            is_calibrated = True

        if is_calibrated:
            mdsm, calib_stats = elevation_engine.calibrate_absolute_dsm(
                rdsm=rdsm,
                min_elev=default_min,
                max_elev=default_max,
                reference_dem=ref_dem,
                gcps=gcps
            )
        else:
            # Relative only (0 - 100 relative height units)
            mdsm = (rdsm * 100.0).astype(np.float32)
            calib_stats = {
                "method": "relative_rdsm_uncalibrated",
                "scale_factor": 100.0,
                "shift_meters": 0.0,
                "min_elevation_m": 0.0,
                "max_elevation_m": 100.0,
                "mean_elevation_m": round(float(np.mean(mdsm)), 2),
                "unit": "relative units [0-100]"
            }

        # Accuracy Evaluation
        accuracy_metrics = None
        diff_grid = None
        diff_colormap_b64 = None
        ref_colormap_b64 = None

        if ref_dem is not None:
            acc_res = elevation_engine.evaluate_accuracy(mdsm, ref_dem)
            if isinstance(acc_res, tuple):
                accuracy_metrics, diff_grid = acc_res
                diff_abs = np.abs(diff_grid)
                diff_norm = normalize_array(diff_abs, vmin=0.0, vmax=max(5.0, float(np.percentile(diff_abs, 98))))
                diff_colormap_b64 = array_to_base64_image(apply_colormap(diff_norm, "magma"))
                ref_colormap_b64 = array_to_base64_image(apply_colormap(normalize_array(ref_dem), "terrain"))

        session_id = str(uuid.uuid4())

        # Generate base64 textures
        norm_mdsm = normalize_array(mdsm)
        turbo_img = apply_colormap(norm_mdsm, "turbo")
        terrain_img = apply_colormap(norm_mdsm, "terrain")
        viridis_img = apply_colormap(norm_mdsm, "viridis")
        hillshade_arr = compute_hillshade(mdsm, azimuth_deg=315.0, altitude_deg=45.0, z_factor=2.0)

        # Slope & Aspect
        slope_deg, aspect_deg = compute_slope_aspect(mdsm)

        # Compute Elevation & Slope Histograms for Right Sidebar
        elev_counts, elev_bins = np.histogram(mdsm, bins=35)
        elev_hist = [
            {"bin": round(float(elev_bins[i]), 1), "count": int(elev_counts[i])}
            for i in range(len(elev_counts))
        ]

        slope_counts, slope_bins = np.histogram(slope_deg, bins=35, range=(0, 90))
        slope_hist = [
            {"bin": round(float(slope_bins[i]), 1), "count": int(slope_counts[i])}
            for i in range(len(slope_counts))
        ]

        geo_info = {
            "lat": "26.9124°N",
            "lon": "75.7873°E",
            "location": filename,
            "crs": "GeoTIFF - EPSG:4326" if geo_meta.get("is_geotiff") else "Optical RGB Raster"
        }

        # Save session
        SESSIONS[session_id] = {
            "session_id": session_id,
            "filename": filename,
            "rgb_img": image,
            "rdsm": rdsm,
            "mdsm": mdsm,
            "ref_dem": ref_dem,
            "diff_grid": diff_grid,
            "calib_stats": calib_stats,
            "accuracy_metrics": accuracy_metrics,
            "gcps": gcps,
            "bounds": [default_min, default_max],
            "geo_metadata": geo_meta,
            "grid_shape": [h, w],
            "geo_info": geo_info
        }

        # Downsample for 3D mesh transfer
        step = 2 if max(w, h) >= 400 else 1
        mesh_grid = mdsm[::step, ::step]

        return {
            "session_id": session_id,
            "filename": filename,
            "is_geotiff": geo_meta.get("is_geotiff", False),
            "is_calibrated": is_calibrated,
            "geo_metadata": geo_meta,
            "geo_info": geo_info,
            "grid_dimensions": {"width": w, "height": h},
            "mesh_grid_dimensions": {"width": mesh_grid.shape[1], "height": mesh_grid.shape[0]},
            "elevation_stats": {
                "min": round(float(np.min(mdsm)), 2),
                "max": round(float(np.max(mdsm)), 2),
                "mean": round(float(np.mean(mdsm)), 2),
                "unit": "meters" if is_calibrated else "relative [0-100]"
            },
            "calibration_stats": calib_stats,
            "accuracy_metrics": accuracy_metrics,
            "gcps": gcps,
            "elevation_histogram": elev_hist,
            "slope_histogram": slope_hist,
            "elevation_grid_sample": mesh_grid.flatten().round(2).tolist(),
            "textures": {
                "rgb": array_to_base64_image(np.array(image)),
                "rdsm": array_to_base64_image((rdsm * 255).astype(np.uint8)),
                "turbo": array_to_base64_image(turbo_img),
                "terrain": array_to_base64_image(terrain_img),
                "viridis": array_to_base64_image(viridis_img),
                "hillshade": array_to_base64_image(hillshade_arr),
                "reference_dem": ref_colormap_b64,
                "diff_error": diff_colormap_b64
            }
        }
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/profile")
async def get_elevation_profile(req: ProfileRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    mdsm = session["mdsm"]
    ref_dem = session.get("ref_dem")
    
    pred_profile = extract_cross_section_profile(mdsm, tuple(req.p1), tuple(req.p2), req.num_samples)
    
    ref_profile = None
    if ref_dem is not None:
        ref_profile = extract_cross_section_profile(ref_dem, tuple(req.p1), tuple(req.p2), req.num_samples)
        
    return {
        "predicted_profile": pred_profile,
        "reference_profile": ref_profile,
        "unit": session["calib_stats"].get("unit", "meters")
    }

@app.post("/api/calibrate")
async def recalibrate_session(req: CalibrateRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    rdsm = session["rdsm"]
    ref_dem = session.get("ref_dem")
    
    mdsm, calib_stats = elevation_engine.calibrate_absolute_dsm(
        rdsm=rdsm,
        min_elev=req.min_elevation,
        max_elev=req.max_elevation,
        reference_dem=ref_dem,
        gcps=req.gcps
    )
    
    session["mdsm"] = mdsm
    session["calib_stats"] = calib_stats
    session["gcps"] = req.gcps or session.get("gcps", [])
    
    accuracy_metrics = None
    diff_colormap_b64 = None
    if ref_dem is not None:
        acc_res = elevation_engine.evaluate_accuracy(mdsm, ref_dem)
        if isinstance(acc_res, tuple):
            accuracy_metrics, diff_grid = acc_res
            session["accuracy_metrics"] = accuracy_metrics
            session["diff_grid"] = diff_grid
            diff_abs = np.abs(diff_grid)
            diff_norm = normalize_array(diff_abs, vmin=0.0, vmax=max(5.0, float(np.percentile(diff_abs, 98))))
            diff_colormap_b64 = array_to_base64_image(apply_colormap(diff_norm, "magma"))

    norm_mdsm = normalize_array(mdsm)
    step = 2 if max(mdsm.shape) >= 400 else 1
    mesh_grid = mdsm[::step, ::step]

    return {
        "session_id": req.session_id,
        "is_calibrated": True,
        "elevation_stats": {
            "min": round(float(np.min(mdsm)), 2),
            "max": round(float(np.max(mdsm)), 2),
            "mean": round(float(np.mean(mdsm)), 2),
            "unit": "meters"
        },
        "calibration_stats": calib_stats,
        "accuracy_metrics": accuracy_metrics,
        "elevation_grid_sample": mesh_grid.flatten().round(2).tolist(),
        "textures": {
            "turbo": array_to_base64_image(apply_colormap(norm_mdsm, "turbo")),
            "terrain": array_to_base64_image(apply_colormap(norm_mdsm, "terrain")),
            "viridis": array_to_base64_image(apply_colormap(norm_mdsm, "viridis")),
            "hillshade": array_to_base64_image(compute_hillshade(mdsm, 315.0, 45.0, 2.0)),
            "diff_error": diff_colormap_b64
        }
    }

@app.get("/api/export/{export_type}/{session_id}")
async def export_data(export_type: str, session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    mdsm = session["mdsm"]
    rgb_img = session["rgb_img"]
    
    if export_type == "obj":
        buf = mesh_exporter.export_obj(mdsm, rgb_img)
        return StreamingResponse(
            buf,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=terrafly_model_{session_id[:8]}.obj"}
        )
    elif export_type == "stl":
        buf = mesh_exporter.export_stl(mdsm)
        return StreamingResponse(
            buf,
            media_type="application/sla",
            headers={"Content-Disposition": f"attachment; filename=terrafly_mesh_{session_id[:8]}.stl"}
        )
    elif export_type == "ply":
        buf = mesh_exporter.export_ply(mdsm, rgb_img)
        return StreamingResponse(
            buf,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=terrafly_pointcloud_{session_id[:8]}.ply"}
        )
    elif export_type == "geotiff":
        buf = mesh_exporter.export_geotiff(mdsm)
        return StreamingResponse(
            buf,
            media_type="image/tiff",
            headers={"Content-Disposition": f"attachment; filename=terrafly_dsm_{session_id[:8]}.tif"}
        )
    elif export_type == "heightmap_png":
        buf = mesh_exporter.export_png_heightmap(mdsm)
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=terrafly_heightmap_16bit_{session_id[:8]}.png"}
        )
    elif export_type == "metrics_csv":
        metrics = session.get("accuracy_metrics") or {}
        calib = session.get("calib_stats") or {}
        
        lines = [
            "TerraFly Elevation Reconstruction & Accuracy Report",
            f"Session ID,{session_id}",
            f"Dataset / File,{session.get('name', session.get('filename', 'Custom'))}",
            f"Calibration Method,{calib.get('method', 'N/A')}",
            f"Elevation Range (Min / Max),{calib.get('min_elevation_m', 'N/A')}m to {calib.get('max_elevation_m', 'N/A')}m",
            "",
            "--- Geodetic Accuracy Metrics (vs Reference DEM) ---",
            f"RMSE (Root Mean Square Error),{metrics.get('rmse', 'N/A')} m",
            f"MAE (Mean Absolute Error),{metrics.get('mae', 'N/A')} m",
            f"NMAD (Normalized Median Absolute Deviation),{metrics.get('nmad', 'N/A')} m",
            f"LE90 (Linear Error 90% Confidence),{metrics.get('le90', 'N/A')} m",
            f"LE95 (Linear Error 95% Confidence),{metrics.get('le95', 'N/A')} m",
            f"Mean Bias Error,{metrics.get('mean_error_bias', 'N/A')} m",
            f"Pearson Correlation (r),{metrics.get('pearson_r', 'N/A')}",
            f"SSIM (Structural Similarity),{metrics.get('ssim', 'N/A')}",
            f"Photogrammetric Grade,{metrics.get('grade', 'N/A')}"
        ]
        buf = io.BytesIO("\n".join(lines).encode("utf-8"))
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=terrafly_metrics_{session_id[:8]}.csv"}
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid export type")

# Serve frontend static assets
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
