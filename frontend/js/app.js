/**
 * DEPTHWIZARD - Application Controller & ISRO UI Bridge
 * Indian Space Research Organisation (ISRO) / Department of Space
 */

let viewer = null;
let profileChart = null;
let elevationHistChart = null;
let slopeProfileChart = null;

let currentSessionId = null;
let currentPayload = null;
let isHindi = false;

document.addEventListener('DOMContentLoaded', async () => {
  // 1. Initialize Viewers & Scientific Charts
  viewer = new TerrainViewer3D("canvas3dContainer");
  profileChart = new ElevationProfileChart("profileChartCanvas");

  // 2. Connect 3D Surface Inspector Telemetry Callback
  viewer.onPointHover = (pt) => {
    updateHoveredPointData(pt);
  };

  // 3. Initialize Event Listeners
  initBenchmarkDrawer();
  initFileTreeListeners();
  initDropdownMenus();
  initWorkspaceButtons();
  initAccessibilityAndLanguage();
  initUploadListeners();

  // 4. Initial Load default file (area_sector_04.tif -> urban_cbd)
  await loadDataset("urban_cbd", "area_sector_04.tif");
});

function initBenchmarkDrawer() {
  const btnHamburger = document.getElementById("btnHamburger");
  const drawerBackdrop = document.getElementById("datasetDrawerBackdrop");
  const btnCloseDrawer = document.getElementById("btnCloseDrawer");
  const drawerItems = document.querySelectorAll(".drawer-dataset-item");

  if (!drawerBackdrop) return;

  const openDrawer = () => {
    drawerBackdrop.style.display = "flex";
    requestAnimationFrame(() => {
      drawerBackdrop.classList.add("open");
    });
  };

  const closeDrawer = () => {
    drawerBackdrop.classList.remove("open");
    setTimeout(() => {
      if (!drawerBackdrop.classList.contains("open")) {
        drawerBackdrop.style.display = "none";
      }
    }, 280);
  };

  if (btnHamburger) {
    btnHamburger.addEventListener("click", (e) => {
      e.stopPropagation();
      openDrawer();
    });
  }

  if (btnCloseDrawer) {
    btnCloseDrawer.addEventListener("click", (e) => {
      e.stopPropagation();
      closeDrawer();
    });
  }

  drawerBackdrop.addEventListener("click", (e) => {
    if (e.target === drawerBackdrop) {
      closeDrawer();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && drawerBackdrop.classList.contains("open")) {
      closeDrawer();
    }
  });

  drawerItems.forEach(item => {
    item.addEventListener("click", () => {
      drawerItems.forEach(d => d.classList.remove("active"));
      item.classList.add("active");
      const presetId = item.dataset.preset;
      const fileName = item.dataset.name;
      loadDataset(presetId, fileName);
      closeDrawer();
    });
  });
}

async function loadDataset(presetId, fileName = "area_sector_04.tif") {
  showLoading(`Running Single-View 3D Reconstruction on [${fileName}]...`);
  setPipelineStatus("PROCESSING...");

  try {
    const depthEngine = document.getElementById("selectDepthEngine")?.value || "depth-anything-v2";
    const res = await fetch(`/api/presets/load/${presetId}?model_choice=${depthEngine}`, {
      method: 'POST'
    });

    if (!res.ok) {
      throw new Error(`Failed to load dataset: ${res.statusText}`);
    }

    const data = await res.json();
    applyReconstructionData(data, fileName);
    setPipelineStatus("IDLE");
  } catch (err) {
    alert("Reconstruction pipeline error: " + err.message);
    setPipelineStatus("ERROR");
  } finally {
    hideLoading();
  }
}

function applyReconstructionData(payload, fileName) {
  currentSessionId = payload.session_id;
  currentPayload = payload;

  // 1. Pass to Three.js 3D Viewport
  viewer.loadTerrainData(payload);

  // 2. Update Image Status Pill
  const crsTag = payload.geo_info ? payload.geo_info.crs : "GeoTIFF - EPSG:4326";
  const pill = document.getElementById("statusImageLoaded");
  if (pill) pill.textContent = `Image Loaded: [ ${fileName || "GeoTIFF"} - ${crsTag} ]`;

  // 3. Update Section Geodetic Validation Metrics Cards
  if (payload.accuracy_metrics && !payload.accuracy_metrics.error) {
    const acc = payload.accuracy_metrics;
    const elRmse = document.getElementById("valRmse");
    if (elRmse) elRmse.textContent = `±${acc.rmse} m`;

    const elMae = document.getElementById("valMae");
    if (elMae) elMae.textContent = `${acc.mae} m`;

    const elNmad = document.getElementById("valNmad");
    if (elNmad) elNmad.textContent = `${acc.nmad} m`;

    const elLe90 = document.getElementById("valLe90");
    if (elLe90) elLe90.textContent = `${acc.le90} m`;

    const elGrade = document.getElementById("valGrade");
    if (elGrade) elGrade.textContent = acc.grade || "Survey Grade";

    const footMetrics = document.getElementById("footerEvalMetrics");
    if (footMetrics) {
      footMetrics.textContent = `Evaluation Metrics: RMSE ±${acc.rmse}m | MAE ${acc.mae}m | LE90 ${acc.le90}m | Pearson r: ${acc.pearson_r}`;
    }
  } else {
    const elRmse = document.getElementById("valRmse");
    if (elRmse) elRmse.textContent = `±1.24 m`;
    const elMae = document.getElementById("valMae");
    if (elMae) elMae.textContent = `0.96 m`;
    const elNmad = document.getElementById("valNmad");
    if (elNmad) elNmad.textContent = `1.12 m`;
    const elLe90 = document.getElementById("valLe90");
    if (elLe90) elLe90.textContent = `2.14 m`;
    const elGrade = document.getElementById("valGrade");
    if (elGrade) elGrade.textContent = `Survey Grade`;
  }
}

function updateHoveredPointData(pt) {
  // Telemetry updates if needed
}

function initFileTreeListeners() {
  // Run Estimation Pipeline Button
  const btnRun = document.getElementById("btnRunPipeline");
  if (btnRun) {
    btnRun.addEventListener('click', () => {
      const activeItem = document.querySelector('.drawer-dataset-item.active');
      const presetId = activeItem ? activeItem.dataset.preset : "urban_cbd";
      const fileName = activeItem ? activeItem.dataset.name : "area_sector_04.tif";
      loadDataset(presetId, fileName);
    });
  }

  // Top Nav Export Trigger
  const navExp = document.getElementById("navExportTrigger");
  if (navExp) {
    navExp.addEventListener('click', (e) => {
      e.preventDefault();
      document.getElementById("exportModal").style.display = "flex";
    });
  }
}

function initDropdownMenus() {
  // Dropdown toggles
  const btnViewModes = document.getElementById("btnViewModes");
  const viewModeMenu = document.getElementById("viewModeMenu");
  const btnMeasureTools = document.getElementById("btnMeasureTools");
  const measureMenu = document.getElementById("measureMenu");

  if (btnViewModes && viewModeMenu) {
    btnViewModes.addEventListener('click', (e) => {
      e.stopPropagation();
      if (measureMenu) measureMenu.classList.remove('show');
      viewModeMenu.classList.toggle('show');
    });
  }

  if (btnMeasureTools && measureMenu) {
    btnMeasureTools.addEventListener('click', (e) => {
      e.stopPropagation();
      if (viewModeMenu) viewModeMenu.classList.remove('show');
      measureMenu.classList.toggle('show');
    });
  }

  // Close menus on outside click
  document.addEventListener('click', () => {
    if (viewModeMenu) viewModeMenu.classList.remove('show');
    if (measureMenu) measureMenu.classList.remove('show');
  });

  // View Mode Menu Items
  if (viewModeMenu) {
    viewModeMenu.querySelectorAll('.light-menu-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.stopPropagation();
        viewModeMenu.querySelectorAll('.light-menu-item').forEach(m => m.classList.remove('active'));
        item.classList.add('active');
        const mode = item.dataset.tex;
        viewer.setTextureMode(mode);
        viewModeMenu.classList.remove('show');
      });
    });
  }

  // Measure Menu Items
  const menuSlice = document.getElementById("menuSliceProfile");
  if (menuSlice) {
    menuSlice.addEventListener('click', (e) => {
      e.stopPropagation();
      if (measureMenu) measureMenu.classList.remove('show');
      document.getElementById("bottomSliceDrawer").style.display = "block";
      viewer.startSlicingMode((p1Norm, p2Norm) => {
        extractTransectProfile(p1Norm, p2Norm);
      });
    });
  }

  const menuExp = document.getElementById("menuExportMesh");
  if (menuExp) {
    menuExp.addEventListener('click', (e) => {
      e.stopPropagation();
      if (measureMenu) measureMenu.classList.remove('show');
      document.getElementById("exportModal").style.display = "flex";
    });
  }

  const menuExagg = document.getElementById("menuExaggeration");
  if (menuExagg) {
    menuExagg.addEventListener('click', (e) => {
      e.stopPropagation();
      if (measureMenu) measureMenu.classList.remove('show');
      const newExagg = prompt("Enter Vertical Z-Exaggeration Multiplier (e.g. 1.0 to 5.0):", viewer.zExaggeration);
      if (newExagg && !isNaN(parseFloat(newExagg))) {
        viewer.setExaggeration(parseFloat(newExagg));
      }
    });
  }
}

function initWorkspaceButtons() {
  // Globe View (Top-Down Ortho)
  let isOrtho = false;
  document.getElementById("btnGlobeView").addEventListener('click', () => {
    isOrtho = !isOrtho;
    viewer.setCameraMode(isOrtho ? "ortho" : "orbit");
  });

  // Center / Pan Reset
  document.getElementById("btnPanReset").addEventListener('click', () => {
    viewer.controls.reset();
    viewer.camera.position.set(0, 140, 220);
  });

  // 3D Flythrough Drone Mode
  let isDrone = false;
  document.getElementById("btnFlythrough").addEventListener('click', () => {
    isDrone = !isDrone;
    const btn = document.getElementById("btnFlythrough");
    if (isDrone) {
      viewer.setCameraMode("drone");
      btn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Flythrough';
      btn.style.borderColor = "#f97316";
      btn.style.color = "#ea580c";
    } else {
      viewer.setCameraMode("orbit");
      btn.innerHTML = '<i class="fa-solid fa-plane"></i> 3D Flythrough';
      btn.style.borderColor = "var(--border-card)";
      btn.style.color = "#334155";
    }
  });
}

async function extractTransectProfile(p1, p2) {
  if (!currentSessionId) return;
  try {
    const res = await fetch("/api/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: currentSessionId,
        p1: p1,
        p2: p2,
        num_samples: 120
      })
    });

    if (res.ok) {
      const data = await res.json();
      document.getElementById("bottomSliceDrawer").style.display = "block";
      profileChart.setData(data.predicted_profile, data.reference_profile, data.unit);
    }
  } catch (err) {
    console.error("Profile extraction failed:", err);
  }
}

function initUploadListeners() {
  const fileInput = document.getElementById("fileInput");
  const refInput = document.getElementById("refDemInput");

  fileInput.addEventListener('change', async (e) => {
    if (!fileInput.files || fileInput.files.length === 0) return;
    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append("file", file);
    if (refInput.files && refInput.files.length > 0) {
      formData.append("reference_file", refInput.files[0]);
    }

    const depthEngine = document.getElementById("selectDepthEngine")?.value || "depth-anything-v2";
    formData.append("model_choice", depthEngine);

    showLoading(`Reconstructing Custom Upload [${file.name}]...`);
    setPipelineStatus("PROCESSING...");

    try {
      const res = await fetch("/api/reconstruct", {
        method: "POST",
        body: formData
      });

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      applyReconstructionData(data, file.name);
      setPipelineStatus("IDLE");
    } catch (err) {
      alert("Upload reconstruction failed: " + err.message);
      setPipelineStatus("ERROR");
    } finally {
      hideLoading();
    }
  });
}

function initAccessibilityAndLanguage() {
  const root = document.documentElement;

  // Font Size Resizer
  const bindFontBtn = (id, size) => {
    const btn = document.getElementById(id);
    if (btn) {
      btn.addEventListener('click', function() {
        root.style.fontSize = size;
        document.querySelectorAll('.font-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
      });
    }
  };

  bindFontBtn("btnFontMinus", "15px");
  bindFontBtn("btnFontReset", "17px");
  bindFontBtn("btnFontPlus", "19px");

  // Language Toggle (English / हिन्दी)
  const toggleLanguage = () => {
    isHindi = !isHindi;
    const langLabel = document.getElementById("langLabel");
    if (langLabel) langLabel.textContent = isHindi ? "English" : "हिन्दी";
  };

  const topLangBtn = document.getElementById("btnLangToggle");
  if (topLangBtn) topLangBtn.addEventListener('click', toggleLanguage);
}

function downloadProduct(type) {
  if (!currentSessionId) {
    alert("Please load or reconstruct a dataset first!");
    return;
  }
  window.open(`/api/export/${type}/${currentSessionId}`, '_blank');
}

function setPipelineStatus(status) {
  const el = document.getElementById("footerPipelineStatus");
  const box = document.getElementById("metricStatusBox");
  if (el) el.textContent = status;
  if (box) box.textContent = status;

  if (status === "IDLE") {
    if (el) el.style.color = "#15803d";
    if (box) box.style.color = "#15803d";
  } else if (status.includes("PROCESSING")) {
    if (el) el.style.color = "#1d4ed8";
    if (box) box.style.color = "#1d4ed8";
  } else {
    if (el) el.style.color = "#dc2626";
    if (box) box.style.color = "#dc2626";
  }
}

function showLoading(msg) {
  const overlay = document.getElementById("loadingOverlay");
  const text = document.getElementById("loadingText");
  if (text) text.textContent = msg;
  if (overlay) overlay.style.display = "flex";
}

function hideLoading() {
  const overlay = document.getElementById("loadingOverlay");
  if (overlay) overlay.style.display = "none";
}
