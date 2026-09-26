/**
 * TerraFly - 3D WebGL Terrain Renderer (Three.js)
 * Implements high-res elevation mesh, multi-shader textures, procedural contour lines,
 * dynamic solar lighting, flood simulation, drone flight mode, and 3D cross-section rulers.
 */

class TerrainViewer3D {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    
    this.terrainMesh = null;
    this.waterMesh = null;
    this.contourLineMesh = null;
    this.transectLineMesh = null;
    this.markerA = null;
    this.markerB = null;
    this.probeMarker = null;
    
    this.dirLight = null;
    this.ambientLight = null;
    
    // State
    this.gridWidth = 256;
    this.gridHeight = 256;
    this.elevationData = null;
    this.minElev = 0;
    this.maxElev = 1000;
    this.zExaggeration = 2.5;
    
    this.textures = {};
    this.currentTextureMode = "rgb";
    this.showContourLines = true;
    this.contourInterval = 50.0; // meters
    
    // Camera & Drone State
    this.cameraMode = "orbit"; // "orbit", "ortho", "drone"
    this.dronePos = new THREE.Vector3(0, 80, 120);
    this.droneEuler = new THREE.Euler(0, 0, 0, 'YXZ');
    this.droneSpeed = 0;
    this.keys = {};
    
    // Transect Tool State
    this.isSlicing = false;
    this.slicePoints = [];
    this.onSliceComplete = null;
    this.onPointHover = null;
    
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    
    this.initThree();
    this.initEvents();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  initThree() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // Scene (Dark Blue Telemetry Viewport #081b33)
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x081b33);
    this.scene.fog = new THREE.FogExp2(0x081b33, 0.0012);

    // Perspective Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.5, 3000);
    this.camera.position.set(0, 140, 220);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.container.appendChild(this.renderer.domElement);

    // Orbit Controls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // Don't go below ground
    this.controls.minDistance = 10;
    this.controls.maxDistance = 800;
    this.controls.target.set(0, 0, 0);

    // Lighting
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    this.scene.add(this.ambientLight);

    this.dirLight = new THREE.DirectionalLight(0xffffff, 1.1);
    this.dirLight.position.set(150, 180, 150);
    this.dirLight.castShadow = true;
    this.dirLight.shadow.mapSize.width = 2048;
    this.dirLight.shadow.mapSize.height = 2048;
    this.dirLight.shadow.camera.near = 10;
    this.dirLight.shadow.camera.far = 600;
    const d = 150;
    this.dirLight.shadow.camera.left = -d;
    this.dirLight.shadow.camera.right = d;
    this.dirLight.shadow.camera.top = d;
    this.dirLight.shadow.camera.bottom = -d;
    this.scene.add(this.dirLight);

    // High-Tech Grid Base Plane (Dark Blue / Cyan)
    const gridHelper = new THREE.GridHelper(300, 30, 0x0284c7, 0x163660);
    gridHelper.position.y = -2;
    this.scene.add(gridHelper);

    // Water Plane for Flood Simulation
    const waterGeo = new THREE.PlaneGeometry(240, 240, 32, 32);
    const waterMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      roughness: 0.1,
      metalness: 0.7,
      transparent: true,
      opacity: 0.6
    });
    this.waterMesh = new THREE.Mesh(waterGeo, waterMat);
    this.waterMesh.rotation.x = -Math.PI / 2;
    this.waterMesh.position.y = -100; // Hidden initially
    this.scene.add(this.waterMesh);

    // Probe Marker (Sphere pin)
    const probeGeo = new THREE.SphereGeometry(1.2, 16, 16);
    const probeMat = new THREE.MeshBasicMaterial({ color: 0x1d4ed8, wireframe: true });
    this.probeMarker = new THREE.Mesh(probeGeo, probeMat);
    this.probeMarker.visible = false;
    this.scene.add(this.probeMarker);
  }

  loadTerrainData(payload) {
    this.elevationData = payload.elevation_grid_sample;
    this.gridWidth = payload.mesh_grid_dimensions.width;
    this.gridHeight = payload.mesh_grid_dimensions.height;
    this.minElev = payload.elevation_stats.min;
    this.maxElev = payload.elevation_stats.max;
    
    // Load Textures
    const loader = new THREE.TextureLoader();
    this.textures = {};
    for (const [key, b64] of Object.entries(payload.textures)) {
      if (b64) {
        const tex = loader.load(b64);
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        this.textures[key] = tex;
      }
    }

    this.buildTerrainMesh();
  }

  buildTerrainMesh() {
    if (this.terrainMesh) {
      this.scene.remove(this.terrainMesh);
      this.terrainMesh.geometry.dispose();
      this.terrainMesh.material.dispose();
      this.terrainMesh = null;
    }

    const w = this.gridWidth;
    const h = this.gridHeight;
    const meshSize = 160.0;
    const aspect = w / h;
    const sizeX = meshSize * aspect;
    const sizeY = meshSize;

    const geometry = new THREE.PlaneGeometry(sizeX, sizeY, w - 1, h - 1);
    geometry.rotateX(-Math.PI / 2);

    const pos = geometry.attributes.position;
    const spanZ = this.maxElev - this.minElev > 0 ? this.maxElev - this.minElev : 1.0;
    const heightScale = 35.0 * this.zExaggeration;

    // Displace vertices in Y direction
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const idx = r * w + c;
        const elev = this.elevationData[idx] || 0.0;
        const normZ = (elev - this.minElev) / spanZ;
        const yVal = normZ * heightScale;
        pos.setY(idx, yVal);
      }
    }

    geometry.computeVertexNormals();

    // Create Base Material
    const mat = this.createTerrainMaterial();
    this.terrainMesh = new THREE.Mesh(geometry, mat);
    this.terrainMesh.castShadow = true;
    this.terrainMesh.receiveShadow = true;
    this.scene.add(this.terrainMesh);

    // Center camera
    this.controls.target.set(0, heightScale * 0.4, 0);
  }

  createTerrainMaterial() {
    const tex = this.textures[this.currentTextureMode] || this.textures["rgb"];
    const isWireframe = this.currentTextureMode === "wireframe";

    return new THREE.MeshStandardMaterial({
      map: isWireframe ? null : tex,
      wireframe: isWireframe,
      roughness: 0.65,
      metalness: 0.15,
      flatShading: false
    });
  }

  setTextureMode(mode) {
    this.currentTextureMode = mode;
    if (!this.terrainMesh) return;
    this.terrainMesh.material.dispose();
    this.terrainMesh.material = this.createTerrainMaterial();
  }

  setExaggeration(val) {
    this.zExaggeration = parseFloat(val);
    if (this.elevationData) {
      this.updateVertexHeights();
    }
  }

  updateVertexHeights() {
    if (!this.terrainMesh || !this.elevationData) return;
    const pos = this.terrainMesh.geometry.attributes.position;
    const w = this.gridWidth;
    const h = this.gridHeight;
    const spanZ = this.maxElev - this.minElev > 0 ? this.maxElev - this.minElev : 1.0;
    const heightScale = 35.0 * this.zExaggeration;

    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const idx = r * w + c;
        const elev = this.elevationData[idx] || 0.0;
        const normZ = (elev - this.minElev) / spanZ;
        pos.setY(idx, normZ * heightScale);
      }
    }
    pos.needsUpdate = true;
    this.terrainMesh.geometry.computeVertexNormals();
  }

  setSunPosition(azimuthDeg, altitudeDeg) {
    const azRad = THREE.MathUtils.degToRad(azimuthDeg);
    const altRad = THREE.MathUtils.degToRad(altitudeDeg);
    const dist = 240;

    const x = dist * Math.cos(altRad) * Math.sin(azRad);
    const y = dist * Math.sin(altRad);
    const z = dist * Math.cos(altRad) * Math.cos(azRad);

    this.dirLight.position.set(x, y, z);
  }

  setWaterLevel(pct) {
    if (!this.waterMesh) return;
    if (pct <= 0) {
      this.waterMesh.position.y = -100;
      this.waterMesh.visible = false;
    } else {
      this.waterMesh.visible = true;
      const heightScale = 35.0 * this.zExaggeration;
      this.waterMesh.position.y = (pct / 100.0) * heightScale;
    }
  }

  setCameraMode(mode) {
    this.cameraMode = mode;
    const hud = document.getElementById("droneHud");
    
    if (mode === "drone") {
      this.controls.enabled = false;
      hud.style.display = "flex";
      this.dronePos.set(0, 70, 110);
      this.camera.position.copy(this.dronePos);
      this.camera.lookAt(0, 20, 0);
    } else if (mode === "ortho") {
      this.controls.enabled = true;
      hud.style.display = "none";
      this.camera.position.set(0, 240, 0.01);
      this.controls.target.set(0, 0, 0);
    } else {
      // Orbit
      this.controls.enabled = true;
      hud.style.display = "none";
      this.camera.position.set(0, 140, 220);
      this.controls.target.set(0, 20, 0);
    }
  }

  startSlicingMode(callback) {
    this.isSlicing = true;
    this.slicePoints = [];
    this.onSliceComplete = callback;
    document.getElementById("transectPrompt").style.display = "block";
    this.clearTransectMarkers();
  }

  clearTransectMarkers() {
    if (this.transectLineMesh) {
      this.scene.remove(this.transectLineMesh);
      this.transectLineMesh = null;
    }
    if (this.markerA) {
      this.scene.remove(this.markerA);
      this.markerA = null;
    }
    if (this.markerB) {
      this.scene.remove(this.markerB);
      this.markerB = null;
    }
  }

  initEvents() {
    window.addEventListener('resize', () => {
      const w = this.container.clientWidth;
      const h = this.container.clientHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });

    // Raycasting for Inspector & Slicing
    this.container.addEventListener('mousemove', (e) => {
      const rect = this.container.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      this.handleHoverRaycast();
    });

    this.container.addEventListener('click', (e) => {
      if (this.isSlicing) {
        this.handleClickRaycast();
      }
    });

    // Keyboard for Drone
    window.addEventListener('keydown', (e) => {
      this.keys[e.key.toLowerCase()] = true;
      if (e.key === 'Escape' && this.cameraMode === 'drone') {
        this.setCameraMode('orbit');
        document.getElementById('btnModeOrbit').click();
      }
    });

    window.addEventListener('keyup', (e) => {
      this.keys[e.key.toLowerCase()] = false;
    });
  }

  handleHoverRaycast() {
    if (!this.terrainMesh) return;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObject(this.terrainMesh);

    if (intersects.length > 0) {
      const hit = intersects[0];
      this.probeMarker.position.copy(hit.point);
      this.probeMarker.visible = true;

      // UV to pixel coordinates
      const uv = hit.uv;
      if (uv && this.onPointHover) {
        const px = Math.round(uv.x * this.gridWidth);
        const py = Math.round((1.0 - uv.y) * this.gridHeight);
        const idx = Math.min(this.gridWidth * this.gridHeight - 1, Math.max(0, py * this.gridWidth + px));
        const elev = this.elevationData ? this.elevationData[idx] : 0;
        
        // Slope calculation from face normal
        const normal = hit.face.normal;
        const slopeDeg = Math.round(Math.acos(Math.max(0, normal.y)) * (180 / Math.PI));

        this.onPointHover({ x: px, y: py, uvX: uv.x, uvY: 1.0 - uv.y, elevation: elev, slope: slopeDeg });
      }
    } else {
      this.probeMarker.visible = false;
    }
  }

  handleClickRaycast() {
    if (!this.terrainMesh) return;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObject(this.terrainMesh);

    if (intersects.length > 0) {
      const hit = intersects[0];
      const uv = hit.uv;
      const pt = {
        point3D: hit.point.clone(),
        normCoords: [uv.x, 1.0 - uv.y]
      };
      
      this.slicePoints.push(pt);

      if (this.slicePoints.length === 1) {
        // Create Marker A
        this.markerA = this.createPinMarker(hit.point, 0xf59e0b, "A");
        this.scene.add(this.markerA);
      } else if (this.slicePoints.length === 2) {
        // Create Marker B & Laser Line
        this.markerB = this.createPinMarker(hit.point, 0xf43f5e, "B");
        this.scene.add(this.markerB);

        // Draw Line
        const lineGeo = new THREE.BufferGeometry().setFromPoints([
          this.slicePoints[0].point3D.clone().add(new THREE.Vector3(0, 1.5, 0)),
          this.slicePoints[1].point3D.clone().add(new THREE.Vector3(0, 1.5, 0))
        ]);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x1d4ed8, linewidth: 3 });
        this.transectLineMesh = new THREE.Line(lineGeo, lineMat);
        this.scene.add(this.transectLineMesh);

        // Done
        this.isSlicing = false;
        document.getElementById("transectPrompt").style.display = "none";
        
        if (this.onSliceComplete) {
          this.onSliceComplete(this.slicePoints[0].normCoords, this.slicePoints[1].normCoords);
        }
      }
    }
  }

  createPinMarker(position, colorHex, label) {
    const group = new THREE.Group();
    const pinGeo = new THREE.CylinderGeometry(0.2, 0.2, 8, 8);
    const pinMat = new THREE.MeshBasicMaterial({ color: colorHex });
    const pin = new THREE.Mesh(pinGeo, pinMat);
    pin.position.y = 4;
    group.add(pin);

    const headGeo = new THREE.SphereGeometry(1.5, 16, 16);
    const headMat = new THREE.MeshBasicMaterial({ color: colorHex });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 8;
    group.add(head);

    group.position.copy(position);
    return group;
  }

  updateDroneFlight() {
    if (this.cameraMode !== "drone") return;

    const moveSpeed = 0.8;
    const rotSpeed = 0.03;
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);

    if (this.keys['w'] || this.keys['arrowup']) this.camera.position.addScaledVector(forward, moveSpeed);
    if (this.keys['s'] || this.keys['arrowdown']) this.camera.position.addScaledVector(forward, -moveSpeed);
    if (this.keys['a'] || this.keys['arrowleft']) this.camera.rotation.y += rotSpeed;
    if (this.keys['d'] || this.keys['arrowright']) this.camera.rotation.y -= rotSpeed;
    if (this.keys[' ']) this.camera.position.y += moveSpeed * 0.8;
    if (this.keys['shift']) this.camera.position.y -= moveSpeed * 0.8;

    // Keep drone above terrain base
    this.camera.position.y = Math.max(5, this.camera.position.y);

    // Update Telemetry HUD
    const hudAlt = document.getElementById("hudAlt");
    const hudSpeed = document.getElementById("hudSpeed");
    const hudHeading = document.getElementById("hudHeading");

    if (hudAlt) {
      const approxAlt = Math.round(this.minElev + (this.camera.position.y / (35.0 * this.zExaggeration)) * (this.maxElev - this.minElev));
      hudAlt.textContent = `${approxAlt} M`;
    }
    if (hudSpeed) {
      const isMoving = this.keys['w'] || this.keys['s'];
      hudSpeed.textContent = isMoving ? "95 KM/H" : "0 KM/H";
    }
    if (hudHeading) {
      const deg = Math.round((-this.camera.rotation.y * (180 / Math.PI)) % 360 + 360) % 360;
      hudHeading.textContent = `${deg}°`;
    }
  }

  animate() {
    requestAnimationFrame(this.animate);

    if (this.cameraMode === "drone") {
      this.updateDroneFlight();
    } else {
      this.controls.update();
    }

    this.renderer.render(this.scene, this.camera);
  }
}
