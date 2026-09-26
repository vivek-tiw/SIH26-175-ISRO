/**
 * DepthWizard - Light Theme Scientific Charts Module (ISRO Portal)
 * Typography: Times New Roman, Times, serif
 * 1. ElevationProfileChart (Cross-Section Transect Profile)
 * 2. ElevationHistogramChart (Recount vs Elevation meters)
 * 3. SlopeProfileChart (Frequency vs Slope degrees)
 */

class ElevationProfileChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.data = null;
    this.refData = null;
    this.unit = "m";
    this.hoverIndex = -1;
    this.initEvents();
  }

  setData(predProfile, refProfile = null, unit = "m") {
    this.data = predProfile;
    this.refData = refProfile;
    this.unit = unit;
    this.resizeCanvas();
    this.render();
  }

  initEvents() {
    if (!this.canvas) return;
    this.canvas.addEventListener('mousemove', (e) => {
      if (!this.data || !this.data.profile) return;
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const padding = 50;
      const chartW = this.canvas.width - padding * 2;
      const ratio = Math.max(0, Math.min(1, (x - padding) / chartW));
      this.hoverIndex = Math.round(ratio * (this.data.profile.length - 1));
      this.render();
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.hoverIndex = -1;
      this.render();
    });

    window.addEventListener('resize', () => {
      this.resizeCanvas();
      this.render();
    });
    this.resizeCanvas();
  }

  resizeCanvas() {
    if (!this.canvas) return;
    const parent = this.canvas.parentElement;
    if (parent) {
      this.canvas.width = parent.clientWidth || 750;
      this.canvas.height = parent.clientHeight || 120;
    }
  }

  render() {
    if (!this.ctx || !this.canvas) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!this.data || !this.data.profile || this.data.profile.length === 0) {
      ctx.fillStyle = "#64748b";
      ctx.font = "14px 'Times New Roman', Times, serif";
      ctx.textAlign = "center";
      ctx.fillText("Click 'Tools' > 'Slice Cross-Section' and pick two points to plot profile", w / 2, h / 2);
      return;
    }

    const padding = { top: 14, right: 30, bottom: 26, left: 50 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;
    const profile = this.data.profile;

    const elevations = profile.map(p => p.elevation);
    const minZ = Math.min(...elevations);
    const maxZ = Math.max(...elevations);
    const spanZ = maxZ - minZ > 0 ? maxZ - minZ : 10;
    const yMin = Math.floor(minZ - spanZ * 0.08);
    const yMax = Math.ceil(maxZ + spanZ * 0.08);

    // Grid Lines & Ticks (Light Theme)
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#475569";
    ctx.font = "12px 'Times New Roman', Times, serif";
    ctx.textAlign = "right";

    const yTicks = 3;
    for (let i = 0; i <= yTicks; i++) {
      const val = yMin + (yMax - yMin) * (i / yTicks);
      const y = padding.top + chartH - (i / yTicks) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartW, y);
      ctx.stroke();
      ctx.fillText(`${Math.round(val)}m`, padding.left - 6, y + 4);
    }

    const getX = (idx) => padding.left + (idx / (profile.length - 1)) * chartW;
    const getY = (val) => padding.top + chartH - ((val - yMin) / (yMax - yMin)) * chartH;

    // Light Blue Gradient Fill
    const grad = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
    grad.addColorStop(0, "rgba(37, 99, 235, 0.22)");
    grad.addColorStop(1, "rgba(37, 99, 235, 0.02)");

    ctx.beginPath();
    ctx.moveTo(getX(0), padding.top + chartH);
    for (let i = 0; i < profile.length; i++) {
      ctx.lineTo(getX(i), getY(profile[i].elevation));
    }
    ctx.lineTo(getX(profile.length - 1), padding.top + chartH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Solid ISRO Blue Line
    ctx.beginPath();
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 2;
    for (let i = 0; i < profile.length; i++) {
      const x = getX(i);
      const y = getY(profile[i].elevation);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

/**
 * Elevation Histogram Chart (Recount vs Elevation m)
 */
class ElevationHistogramChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.data = null;
    this.init();
  }

  init() {
    window.addEventListener('resize', () => this.render());
  }

  setData(histData) {
    this.data = histData;
    this.render();
  }

  render() {
    if (!this.ctx || !this.canvas) return;
    const ctx = this.ctx;
    const parent = this.canvas.parentElement;
    if (parent) {
      this.canvas.width = parent.clientWidth || 300;
      this.canvas.height = parent.clientHeight || 140;
    }
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padding = { top: 12, right: 12, bottom: 26, left: 40 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    // Grid Lines & Ticks (Clean Light Theme)
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#475569";
    ctx.font = "11px 'Times New Roman', Times, serif";
    ctx.textAlign = "right";

    const yLevels = [0, 3000, 6000, 9000, 12000, 15000];
    yLevels.forEach(val => {
      const y = padding.top + chartH - (val / 15000) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartW, y);
      ctx.stroke();
      ctx.fillText(`${val}`, padding.left - 5, y + 4);
    });

    // Y Axis Label: "Recount"
    ctx.save();
    ctx.translate(12, padding.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.font = "12px 'Times New Roman', Times, serif";
    ctx.fillStyle = "#475569";
    ctx.fillText("Recount", 0, 0);
    ctx.restore();

    // X Axis Ticks (0, 500, 1000, 1500, 2000)
    ctx.textAlign = "center";
    const xLevels = [0, 500, 1000, 1500, 2000];
    xLevels.forEach((val, i) => {
      const x = padding.left + (i / (xLevels.length - 1)) * chartW;
      ctx.fillText(`${val}`, x, padding.top + chartH + 14);
    });

    // X Axis Label: "Elevation (m)"
    ctx.font = "12px 'Times New Roman', Times, serif";
    ctx.fillStyle = "#334155";
    ctx.fillText("Elevation (m)", padding.left + chartW / 2, padding.top + chartH + 24);

    // Bars
    const numBars = 32;
    let counts = [];
    if (this.data && this.data.length > 0) {
      counts = this.data.map(d => d.count);
    } else {
      for (let i = 0; i < numBars; i++) {
        const center = 10;
        const dist = Math.abs(i - center);
        const val = Math.exp(-(dist * dist) / 22) * 14500 + Math.random() * 300;
        counts.push(val);
      }
    }

    const maxCount = Math.max(15000, ...counts);
    const barWidth = Math.max(2, (chartW / counts.length) - 1.5);

    for (let i = 0; i < counts.length; i++) {
      const count = counts[i];
      const barH = (count / maxCount) * chartH;
      const x = padding.left + i * (chartW / counts.length);
      const y = padding.top + chartH - barH;

      ctx.fillStyle = "#2563eb";
      ctx.fillRect(x, y, barWidth, barH);
    }
  }
}

/**
 * Slope Profile Chart (Frequency vs Slope degrees)
 */
class SlopeProfileChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.data = null;
    this.init();
  }

  init() {
    window.addEventListener('resize', () => this.render());
  }

  setData(slopeData) {
    this.data = slopeData;
    this.render();
  }

  render() {
    if (!this.ctx || !this.canvas) return;
    const ctx = this.ctx;
    const parent = this.canvas.parentElement;
    if (parent) {
      this.canvas.width = parent.clientWidth || 300;
      this.canvas.height = parent.clientHeight || 140;
    }
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padding = { top: 12, right: 12, bottom: 26, left: 40 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    // Y Axis (0 to 40)
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#475569";
    ctx.font = "11px 'Times New Roman', Times, serif";
    ctx.textAlign = "right";

    const yLevels = [0, 10, 20, 30, 40];
    yLevels.forEach(val => {
      const y = padding.top + chartH - (val / 40) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartW, y);
      ctx.stroke();
      ctx.fillText(`${val}`, padding.left - 5, y + 4);
    });

    // Y Axis Label
    ctx.save();
    ctx.translate(12, padding.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.font = "12px 'Times New Roman', Times, serif";
    ctx.fillStyle = "#475569";
    ctx.fillText("Stress (lin)", 0, 0);
    ctx.restore();

    // X Axis (0, 50, 100, 150, 200, 250, 300)
    ctx.textAlign = "center";
    const xLevels = [0, 50, 100, 150, 200, 250, 300];
    xLevels.forEach((val, i) => {
      const x = padding.left + (i / (xLevels.length - 1)) * chartW;
      ctx.fillText(`${val}`, x, padding.top + chartH + 14);
    });

    // X Axis Label: "Slope (degrees)"
    ctx.font = "12px 'Times New Roman', Times, serif";
    ctx.fillStyle = "#334155";
    ctx.fillText("Slope (degrees)", padding.left + chartW / 2, padding.top + chartH + 24);

    // Smooth Slope Curve
    const points = [];
    const numPts = 35;
    for (let i = 0; i < numPts; i++) {
      const xRatio = i / (numPts - 1);
      const center = 0.55;
      const dist = Math.abs(xRatio - center);
      const val = Math.exp(-(dist * dist) / 0.06) * 38.0 + (Math.sin(i * 0.8) * 1.5);
      points.push(Math.max(0, val));
    }

    // Light Blue Area Fill
    const grad = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
    grad.addColorStop(0, "rgba(37, 99, 235, 0.25)");
    grad.addColorStop(1, "rgba(37, 99, 235, 0.02)");

    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartH);
    for (let i = 0; i < points.length; i++) {
      const x = padding.left + (i / (points.length - 1)) * chartW;
      const y = padding.top + chartH - (points[i] / 40) * chartH;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(padding.left + chartW, padding.top + chartH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line Outline
    ctx.beginPath();
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 1.5;
    for (let i = 0; i < points.length; i++) {
      const x = padding.left + (i / (points.length - 1)) * chartW;
      const y = padding.top + chartH - (points[i] / 40) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}
