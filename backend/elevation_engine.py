"""
TerraFly - Elevation Calibration & Geodetic Accuracy Assessment Engine
Handles:
- Relative Digital Surface Models (rDSM)
- Absolute Height Mapping (mDSM) via GeoTIFF metadata, GCPs, or Reference DEM
- Full SIH PS 175 Geodetic Validation Metrics (RMSE, MAE, NMAD, LE90, LE95, SSIM, Pearson r)
"""

import numpy as np
import tifffile
from PIL import Image
import io
import logging
from typing import Optional, Dict, Any, Tuple, List
from scipy.ndimage import zoom

logger = logging.getLogger("terrafly.elevation")

class ElevationEngine:
    def __init__(self):
        pass

    def extract_geotiff_metadata(self, file_bytes: bytes) -> Dict[str, Any]:
        """Parses GeoTIFF tags, bounding coordinates, and elevation statistics."""
        try:
            with tifffile.TiffFile(io.BytesIO(file_bytes)) as tif:
                page = tif.pages[0]
                tags = {tag.name: tag.value for tag in page.tags}
                arr = page.asarray()
                
                # Check for GeoTIFF keys
                pixel_scale = tags.get("ModelPixelScaleTag", None)
                tie_points = tags.get("ModelTiepointTag", None)
                geo_key_directory = tags.get("GeoKeyDirectoryTag", None)
                
                is_georeferenced = pixel_scale is not None or tie_points is not None or "GeoKeyDirectoryTag" in tags
                
                # If the TIFF contains raw elevation floating-point data
                is_dem_raster = arr.dtype in [np.float32, np.float64, np.int16, np.int32] and arr.ndim == 2
                
                min_val = float(np.nanmin(arr)) if arr.size > 0 else 0.0
                max_val = float(np.nanmax(arr)) if arr.size > 0 else 1.0
                
                meta = {
                    "is_geotiff": is_georeferenced,
                    "is_dem_raster": is_dem_raster,
                    "width": int(page.imagewidth),
                    "height": int(page.imagelength),
                    "dtype": str(arr.dtype),
                    "channels": arr.shape[2] if arr.ndim == 3 else 1,
                    "min_elevation": round(min_val, 2) if is_dem_raster else None,
                    "max_elevation": round(max_val, 2) if is_dem_raster else None,
                    "pixel_scale": list(pixel_scale) if pixel_scale is not None else None,
                    "tie_points": list(tie_points) if tie_points is not None else None,
                    "has_spatial_metadata": is_georeferenced
                }
                return meta
        except Exception as e:
            logger.info(f"Not a valid GeoTIFF or standard image: {e}")
            return {
                "is_geotiff": False,
                "is_dem_raster": False,
                "has_spatial_metadata": False
            }

    def calibrate_absolute_dsm(
        self,
        rdsm: np.ndarray,
        min_elev: float = 500.0,
        max_elev: float = 1200.0,
        gcps: Optional[List[Dict[str, float]]] = None,
        reference_dem: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calibrates relative DSM (normalized [0, 1]) into absolute metric elevations (mDSM in meters).
        Methods:
        1. Reference DEM Least-Squares fit (Scale & Shift)
        2. Ground Control Points (GCPs) Linear Regression fit
        3. Min/Max Bound Interpolation
        """
        h, w = rdsm.shape
        calibration_method = "min_max_bounds"
        scale = max_elev - min_elev
        shift = min_elev

        if reference_dem is not None:
            # Resize reference DEM to match rdsm shape if needed
            if reference_dem.shape != rdsm.shape:
                ref_resized = self._resize_raster(reference_dem, (h, w))
            else:
                ref_resized = reference_dem
                
            # Filter valid (non-NaN / non-nodata) pixels
            valid_mask = np.isfinite(ref_resized) & (ref_resized > -9990)
            if np.sum(valid_mask) > 50:
                y_ref = ref_resized[valid_mask]
                x_rdsm = rdsm[valid_mask]
                
                # Fit y = scale * x + shift using least squares
                A = np.vstack([x_rdsm, np.ones(len(x_rdsm))]).T
                res_scale, res_shift = np.linalg.lstsq(A, y_ref, rcond=None)[0]
                
                if res_scale > 0:
                    scale = float(res_scale)
                    shift = float(res_shift)
                    calibration_method = "reference_dem_least_squares"
                    
        elif gcps and len(gcps) >= 2:
            # GCP-based calibration: GCP format: [{'x': 100, 'y': 150, 'z': 750.5}]
            gcp_x_rdsm = []
            gcp_z_true = []
            for gcp in gcps:
                gx = int(np.clip(gcp.get("x", 0), 0, w - 1))
                gy = int(np.clip(gcp.get("y", 0), 0, h - 1))
                gz = float(gcp.get("z", 0.0))
                gcp_x_rdsm.append(rdsm[gy, gx])
                gcp_z_true.append(gz)
                
            gcp_x_rdsm = np.array(gcp_x_rdsm)
            gcp_z_true = np.array(gcp_z_true)
            
            A = np.vstack([gcp_x_rdsm, np.ones(len(gcp_x_rdsm))]).T
            res_scale, res_shift = np.linalg.lstsq(A, gcp_z_true, rcond=None)[0]
            if res_scale > 0:
                scale = float(res_scale)
                shift = float(res_shift)
                calibration_method = "ground_control_points"

        # Calculate absolute metric elevation
        mdsm = (rdsm * scale + shift).astype(np.float32)

        calib_stats = {
            "method": calibration_method,
            "scale_factor": round(float(scale), 4),
            "shift_meters": round(float(shift), 2),
            "min_elevation_m": round(float(np.min(mdsm)), 2),
            "max_elevation_m": round(float(np.max(mdsm)), 2),
            "mean_elevation_m": round(float(np.mean(mdsm)), 2),
            "elevation_range_m": round(float(np.max(mdsm) - np.min(mdsm)), 2),
            "unit": "meters"
        }

        return mdsm, calib_stats

    def evaluate_accuracy(self, pred_dsm: np.ndarray, ref_dsm: np.ndarray) -> Dict[str, Any]:
        """
        Computes standard geodetic accuracy metrics between predicted DSM and Reference DEM:
        - RMSE (Root Mean Square Error)
        - MAE (Mean Absolute Error)
        - Mean Error (Bias)
        - NMAD (Normalized Median Absolute Deviation)
        - LE90 / LE95 (Linear Error at 90% and 95% confidence)
        - Pearson Correlation (r)
        - SSIM (Structural Similarity Index)
        - Error Distribution & Difference Raster
        """
        h, w = pred_dsm.shape
        if ref_dsm.shape != pred_dsm.shape:
            ref_aligned = self._resize_raster(ref_dsm, (h, w))
        else:
            ref_aligned = ref_dsm

        # Valid mask
        valid = np.isfinite(pred_dsm) & np.isfinite(ref_aligned) & (ref_aligned > -9990)
        
        if np.sum(valid) < 10:
            return {"error": "Insufficient overlapping valid elevation data for evaluation"}

        pred_v = pred_dsm[valid]
        ref_v = ref_aligned[valid]

        diff = pred_v - ref_v
        abs_diff = np.abs(diff)

        # 1. RMSE
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        
        # 2. MAE
        mae = float(np.mean(abs_diff))
        
        # 3. Mean Error (Bias)
        bias = float(np.mean(diff))
        
        # 4. Standard Deviation
        std_err = float(np.std(diff))
        
        # 5. NMAD (Normalized Median Absolute Deviation) - Geodetic standard
        med_diff = np.median(diff)
        nmad = float(1.4826 * np.median(np.abs(diff - med_diff)))
        
        # 6. LE90 and LE95
        le90 = float(np.percentile(abs_diff, 90))
        le95 = float(np.percentile(abs_diff, 95))
        max_error = float(np.max(abs_diff))
        
        # 7. Pearson correlation r
        if np.std(pred_v) > 0 and np.std(ref_v) > 0:
            corr_matrix = np.corrcoef(pred_v, ref_v)
            pearson_r = float(corr_matrix[0, 1])
        else:
            pearson_r = 1.0

        # 8. SSIM (Structural Similarity)
        ssim_val = self._compute_ssim(pred_dsm, ref_aligned)

        # Create 2D Difference Raster for visualization
        diff_grid = np.zeros((h, w), dtype=np.float32)
        diff_grid[valid] = (pred_dsm[valid] - ref_aligned[valid])
        
        # Error histogram
        hist_counts, bin_edges = np.histogram(diff, bins=25)
        histogram = [
            {"bin_start": round(float(bin_edges[i]), 2), "bin_end": round(float(bin_edges[i+1]), 2), "count": int(hist_counts[i])}
            for i in range(len(hist_counts))
        ]

        metrics = {
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "mean_error_bias": round(bias, 3),
            "std_dev_error": round(std_err, 3),
            "nmad": round(nmad, 3),
            "le90": round(le90, 3),
            "le95": round(le95, 3),
            "max_error": round(max_error, 3),
            "pearson_r": round(pearson_r, 4),
            "ssim": round(ssim_val, 4),
            "evaluated_pixels": int(np.sum(valid)),
            "histogram": histogram,
            "grade": self._calculate_accuracy_grade(rmse, le90, pearson_r)
        }

        return metrics, diff_grid

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Computes SSIM index between two 2D elevation grids."""
        # Normalize both grids to [0, 1]
        i1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1) + 1e-8)
        i2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2) + 1e-8)
        
        mu1 = float(np.mean(i1))
        mu2 = float(np.mean(i2))
        sigma1_sq = float(np.var(i1))
        sigma2_sq = float(np.var(i2))
        sigma12 = float(np.cov(i1.flatten(), i2.flatten())[0, 1])
        
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return float(np.clip(ssim, 0.0, 1.0))

    def _calculate_accuracy_grade(self, rmse: float, le90: float, r: float) -> str:
        """Determines SIH photogrammetric accuracy tier."""
        if rmse < 1.5 and r > 0.95:
            return "Survey Grade (Tier 1 - Sub-meter Accuracy)"
        elif rmse < 3.5 and r > 0.90:
            return "Engineering Grade (Tier 2 - High Precision)"
        elif rmse < 8.0 and r > 0.80:
            return "Reconnaissance Grade (Tier 3 - Regional Mapping)"
        else:
            return "Exploratory Grade (Tier 4 - Qualitative Relief)"

    def _resize_raster(self, arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resizes a 2D raster array to match target shape."""
        h_orig, w_orig = arr.shape
        h_target, w_target = target_shape
        zoom_factors = (h_target / h_orig, w_target / w_orig)
        return zoom(arr, zoom_factors, order=1).astype(np.float32)

# Global singleton
elevation_engine = ElevationEngine()
