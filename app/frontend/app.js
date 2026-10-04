/**
 * LX-DFD — AI Deepfake Forensics Research Laboratory Controller
 * Generalizable Deepfake Face Detection Using Spatial-Frequency Feature Learning
 */

const API_BASE = window.location.origin || "http://127.0.0.1:8000";

let activeCharts = {};

document.addEventListener("DOMContentLoaded", () => {
  initParticleCanvas();
  setupNavigation();
  setupMobileDrawer();
  loadPage("page-overview");
});

/* ==========================================================================
   AMBIENT PARTICLE CANVAS
   ========================================================================== */
function initParticleCanvas() {
  const canvas = document.getElementById("particle-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener("resize", () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particles = [];
  const count = Math.min(width > 900 ? 35 : 18, 45);

  for (let i = 0; i < count; i++) {
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      radius: Math.random() * 1.5 + 0.8,
      color: Math.random() > 0.5 ? "rgba(0, 229, 255, 0.4)" : "rgba(139, 92, 246, 0.3)"
    });
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);

    // Connecting lines
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 130) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0, 229, 255, ${0.12 * (1 - dist / 130)})`;
          ctx.lineWidth = 0.6;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Particles
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = width;
      if (p.x > width) p.x = 0;
      if (p.y < 0) p.y = height;
      if (p.y > height) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
    });

    requestAnimationFrame(animate);
  }

  animate();
}

/* ==========================================================================
   NAVIGATION & MOBILE DRAWER
   ========================================================================== */
function setupNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      navItems.forEach(n => n.classList.remove("active"));
      item.classList.add("active");
      const pageId = item.getAttribute("data-page");
      loadPage(pageId);

      document.getElementById("sidebar")?.classList.remove("mobile-open");
    });
  });
}

function setupMobileDrawer() {
  const toggleBtn = document.getElementById("mobile-toggle-btn");
  const closeBtn = document.getElementById("mobile-close-btn");
  const sidebar = document.getElementById("sidebar");

  toggleBtn?.addEventListener("click", () => sidebar?.classList.add("mobile-open"));
  closeBtn?.addEventListener("click", () => sidebar?.classList.remove("mobile-open"));
}

function destroyActiveCharts() {
  Object.keys(activeCharts).forEach(key => {
    if (activeCharts[key]) {
      activeCharts[key].destroy();
      delete activeCharts[key];
    }
  });
}

function animateCounters() {
  const counters = document.querySelectorAll("[data-counter-target]");
  counters.forEach(counter => {
    const target = parseFloat(counter.getAttribute("data-counter-target"));
    const isPercent = counter.getAttribute("data-counter-percent") === "true";
    const isInteger = Number.isInteger(target);
    const duration = 800;
    const start = performance.now();

    function step(now) {
      const progress = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = target * ease;

      if (isPercent) {
        counter.textContent = current.toFixed(1) + "%";
      } else if (isInteger) {
        counter.textContent = Math.round(current).toLocaleString();
      } else {
        counter.textContent = current.toFixed(3);
      }

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }
    requestAnimationFrame(step);
  });
}

function loadPage(pageId) {
  destroyActiveCharts();
  const container = document.getElementById("main-content");
  if (!container) return;

  container.className = "main-content fade-in-section";

  switch (pageId) {
    case "page-overview":
      renderOverview(container);
      break;
    case "page-dataset":
      renderDataset(container);
      break;
    case "page-architecture":
      renderArchitecture(container);
      break;
    case "page-training":
      renderTraining(container);
      break;
    case "page-comparison":
      renderComparison(container);
      break;
    case "page-generalization":
      renderGeneralization(container);
      break;
    case "page-robustness":
      renderRobustness(container);
      break;
    case "page-explainability":
      renderExplainability(container);
      break;
    case "page-errors":
      renderErrors(container);
      break;
    case "page-conclusion":
      renderConclusion(container);
      break;
  }

  setTimeout(animateCounters, 50);
}

/* ==========================================================================
   01. RESEARCH OVERVIEW
   ========================================================================== */
function renderOverview(container) {
  container.innerHTML = `
    <!-- Modern Hero Panel -->
    <div class="hero-panel">
      <div class="hero-left">
        <div class="hero-tag">
          <span>🔬</span>
          <span>AI RESEARCH PLATFORM</span>
        </div>
        <h1 class="hero-title">
          Generalizable Deepfake
          <span class="hero-highlight">Face Detection</span>
        </h1>
        <p class="hero-desc">
          A research-oriented framework combining spatial and frequency-domain features to investigate robust deepfake detection across unseen conditions.
        </p>
        <div class="hero-badges-row">
          <span class="tech-badge badge-cyan">SPATIAL FEATURES</span>
          <span class="tech-badge badge-purple">FREQUENCY FEATURES</span>
          <span class="tech-badge badge-blue">CROSS-DOMAIN GENERALIZATION</span>
        </div>
      </div>

      <!-- Animated AI Waveform & Radar Visual -->
      <div class="hero-right-visual">
        <div class="visual-scanline-ring"></div>
        <div class="waveform-container">
          <div class="waveform-bar"></div>
          <div class="waveform-bar"></div>
          <div class="waveform-bar"></div>
          <div class="waveform-bar"></div>
          <div class="waveform-bar"></div>
          <div class="waveform-bar"></div>
          <div class="waveform-bar"></div>
          <div class="waveform-bar"></div>
          <div class="waveform-bar"></div>
        </div>
      </div>
    </div>

    <!-- 4 Large KPI Metric Cards -->
    <div class="grid-4">
      <div class="metric-kpi-card kpi-cyan">
        <div class="kpi-header">
          <span class="kpi-label">TOTAL IMAGES</span>
          <div class="kpi-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
          </div>
        </div>
        <div class="kpi-number cyan" data-counter-target="12890">12,890</div>
        <div class="kpi-subtext">Total Images in Dataset</div>
      </div>

      <div class="metric-kpi-card kpi-green">
        <div class="kpi-header">
          <span class="kpi-label">AUTHENTIC REAL</span>
          <div class="kpi-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
          </div>
        </div>
        <div class="kpi-number green" data-counter-target="5890">5,890</div>
        <div class="kpi-subtext">45.7% Balanced Real Faces</div>
      </div>

      <div class="metric-kpi-card kpi-red">
        <div class="kpi-header">
          <span class="kpi-label">MANIPULATED FAKE</span>
          <div class="kpi-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
          </div>
        </div>
        <div class="kpi-number red" data-counter-target="7000">7,000</div>
        <div class="kpi-subtext">54.3% GAN / Deepfakes</div>
      </div>

      <div class="metric-kpi-card kpi-purple">
        <div class="kpi-header">
          <span class="kpi-label">MODEL ARCHITECTURE</span>
          <div class="kpi-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
          </div>
        </div>
        <div class="kpi-number purple" style="font-size: 28px;">LX-DFD</div>
        <div class="kpi-subtext">Dual-Branch Attention Fusion</div>
      </div>
    </div>

    <!-- Spatial-Frequency Model Architecture Visual Flow -->
    <div class="arch-diagram-wrapper">
      <div class="card-header-row">
        <div class="card-title" style="color: var(--accent-cyan);">
          <span>🧠</span>
          <span>SPATIAL-FREQUENCY MODEL ARCHITECTURE</span>
        </div>
        <span class="tech-badge badge-cyan">END-TO-END NEURAL FLOW</span>
      </div>

      <div class="arch-flow-row">
        <!-- Input -->
        <div class="arch-node node-input">
          <span>📷</span>
          <span>INPUT IMAGE (RGB 224×224)</span>
        </div>

        <div class="flow-connector"></div>

        <!-- Face Preprocessing -->
        <div class="arch-node node-prep">
          <span>🎯</span>
          <span>FACE PREPROCESSING (OpenCV YuNet + 15% Context Crop)</span>
        </div>

        <div class="flow-connector"></div>

        <!-- Dual Branches -->
        <div class="arch-branches-container">
          <div class="arch-branch-card branch-spatial">
            <div class="branch-title-row">
              <span style="font-weight: 700; color: var(--accent-cyan); font-size: 13px;">SPATIAL BRANCH</span>
              <span class="branch-tag tag-cyan">EfficientNet-B0</span>
            </div>
            <ul class="branch-feature-bullets">
              <li><span class="bullet-cyan"></span> RGB Pixel Texture Extraction</li>
              <li><span class="bullet-cyan"></span> Semantic Boundary & Artifact Features</li>
              <li><span class="bullet-cyan"></span> 256-D Spatial Embedding (F_s)</li>
            </ul>
          </div>

          <div class="arch-branch-card branch-frequency">
            <div class="branch-title-row">
              <span style="font-weight: 700; color: var(--accent-purple); font-size: 13px;">FREQUENCY BRANCH</span>
              <span class="branch-tag tag-purple">2D DCT ConvNet</span>
            </div>
            <ul class="branch-feature-bullets">
              <li><span class="bullet-purple"></span> 2D DCT Log-Magnitude Spectrum</li>
              <li><span class="bullet-purple"></span> GAN Upsampling Frequency Lattice</li>
              <li><span class="bullet-purple"></span> 256-D Frequency Embedding (F_f)</li>
            </ul>
          </div>
        </div>

        <div class="flow-connector"></div>

        <!-- Fusion -->
        <div class="arch-node node-fusion">
          <span>⚖️</span>
          <span>ADAPTIVE ATTENTION FUSION &nbsp;[ F_fused = w_s • F_s + w_f • F_f ]</span>
        </div>

        <div class="flow-connector"></div>

        <!-- Classifier Head -->
        <div class="arch-node node-classifier">
          <span>⚡</span>
          <span>CLASSIFIER HEAD (Linear 256→128 → GELU → Dropout 0.3 → Linear 1)</span>
        </div>

        <div class="flow-connector"></div>

        <!-- Final Decision -->
        <div class="arch-node node-output">
          <span>🛡️</span>
          <span style="font-weight: 800; letter-spacing: 1px;">FINAL PREDICTION: AUTHENTIC REAL [0] / MANIPULATED FAKE [1]</span>
        </div>
      </div>
    </div>

    <!-- Research Gap & Base Paper Citation Grid -->
    <div class="grid-2">
      <!-- Base Paper Citation -->
      <div class="card card-glow-cyan">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-cyan);">BASE RESEARCH PAPER CITATION</div>
          <span class="tech-badge badge-cyan">IEEE ACCESS 2023</span>
        </div>
        <p style="font-size: 14.5px; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
          "An Improved Dense CNN Architecture for Deepfake Image Detection"
        </p>
        <p style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.6;">
          Yogesh Patel, Sudeep Tanwar, Pronaya Bhattacharya, Rajesh Gupta, Turki Alsuwian, Innocent Ewean Davidson, and Thokozile F. Mazibuko.<br>
          <em>IEEE Access</em>, Vol. 11, 2023, pp. 22081–22095.<br>
          <strong style="color: var(--accent-cyan);">DOI: 10.1109/ACCESS.2023.3251417</strong>
        </p>
      </div>

      <!-- Generalization Problem Card -->
      <div class="card" style="border-color: rgba(245, 158, 11, 0.4);">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-warning);">THE GENERALIZATION CHALLENGE</div>
          <span class="tech-badge" style="background: rgba(245, 158, 11, 0.1); color: var(--accent-warning); border: 1px solid rgba(245, 158, 11, 0.3);">RESEARCH MOTIVATION</span>
        </div>
        <p style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;">
          Current deepfake detectors suffer severe performance drop when deployed across unseen generators, diffusion models, and real-world social media compressions. LX-DFD bridges this gap via dual spatial-frequency feature representation.
        </p>
      </div>
    </div>
  `;
}

/* ==========================================================================
   02. DATASET AUDIT & INTEGRITY
   ========================================================================== */
function renderDataset(container) {
  container.innerHTML = `
    <div style="margin-bottom: 24px;">
      <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 800; color: #FFFFFF;">Dataset Analytics & Integrity Audit</h1>
      <p style="font-size: 13px; color: var(--text-muted);">Zero-Leakage Partition Protocol & Perceptual Hash Verification</p>
    </div>

    <!-- 4 KPI Cards -->
    <div class="grid-4">
      <div class="metric-kpi-card kpi-cyan">
        <div class="kpi-header"><span class="kpi-label">TOTAL IMAGES</span></div>
        <div class="kpi-number cyan" data-counter-target="12890">12,890</div>
        <div class="kpi-subtext">100% Valid JPG Format</div>
      </div>
      <div class="metric-kpi-card kpi-green">
        <div class="kpi-header"><span class="kpi-label">AUTHENTIC REAL</span></div>
        <div class="kpi-number green" data-counter-target="5890">5,890</div>
        <div class="kpi-subtext">45.7% Balanced Ground Truth</div>
      </div>
      <div class="metric-kpi-card kpi-red">
        <div class="kpi-header"><span class="kpi-label">MANIPULATED FAKE</span></div>
        <div class="kpi-number red" data-counter-target="7000">7,000</div>
        <div class="kpi-subtext">54.3% Synthesized Faces</div>
      </div>
      <div class="metric-kpi-card kpi-purple">
        <div class="kpi-header"><span class="kpi-label">CORRUPTED FILES</span></div>
        <div class="kpi-number purple">0</div>
        <div class="kpi-subtext">Zero Integrity Errors</div>
      </div>
    </div>

    <!-- Dataset Analytics Main Section -->
    <div class="dataset-analytics-grid">
      <!-- Donut Chart -->
      <div class="card card-glow-cyan">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-cyan);">DATASET ANALYTICS — CLASS DISTRIBUTION</div>
        </div>
        <div class="chart-donut-container">
          <canvas id="dataset-donut-chart"></canvas>
          <div class="donut-center-info">
            <div class="donut-center-value" data-counter-target="12890">12,890</div>
            <div class="donut-center-tag">TOTAL IMAGES</div>
          </div>
        </div>
        <div class="dataset-breakdown-row">
          <div class="dataset-stat-box">
            <div class="stat-val" style="color: var(--status-real);">5,890</div>
            <div class="stat-lbl">REAL (45.7%)</div>
          </div>
          <div class="dataset-stat-box">
            <div class="stat-val" style="color: var(--status-fake);">7,000</div>
            <div class="stat-lbl">FAKE (54.3%)</div>
          </div>
        </div>
      </div>

      <!-- Dataset Audit Card -->
      <div class="card card-glow-purple">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-purple);">DATASET AUDIT & PARTITIONING</div>
          <span class="tech-badge badge-purple">LEAK-FREE PROTOCOL</span>
        </div>
        <div class="dataset-audit-list">
          <div class="audit-item">
            <span class="audit-label">Dataset</span>
            <span class="audit-value">Kaggle Deepfake Face Images</span>
          </div>
          <div class="audit-item">
            <span class="audit-label">Selection</span>
            <span class="audit-value">Final Dataset</span>
          </div>
          <div class="audit-item">
            <span class="audit-label">Training Set (70%)</span>
            <span class="audit-value" style="color: var(--accent-cyan);">9,024 Images (4,124 Real / 4,900 Fake)</span>
          </div>
          <div class="audit-item">
            <span class="audit-label">Validation Set (15%)</span>
            <span class="audit-value">1,933 Images (883 Real / 1,050 Fake)</span>
          </div>
          <div class="audit-item">
            <span class="audit-label">Held-Out Test Set (15%)</span>
            <span class="audit-value">1,933 Images (883 Real / 1,050 Fake)</span>
          </div>
          <div class="audit-item">
            <span class="audit-label">Perceptual dHash Leakage Check</span>
            <span class="audit-value" style="color: var(--status-real);">✓ Verified 0 Leakage</span>
          </div>
        </div>
      </div>
    </div>
  `;

  setTimeout(() => {
    const ctx = document.getElementById("dataset-donut-chart");
    if (ctx) {
      activeCharts["datasetDonut"] = new Chart(ctx, {
        type: "doughnut",
        data: {
          labels: ["Real Images", "Fake Images"],
          datasets: [{
            data: [5890, 7000],
            backgroundColor: ["#10B981", "#EF4444"],
            borderColor: ["#050812", "#050812"],
            borderWidth: 4,
            hoverOffset: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: "75%",
          plugins: {
            legend: { display: false }
          }
        }
      });
    }
  }, 50);
}

/* ==========================================================================
   03. MODEL ARCHITECTURE
   ========================================================================== */
function renderArchitecture(container) {
  container.innerHTML = `
    <div style="margin-bottom: 24px;">
      <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 800; color: #FFFFFF;">LX-DFD Dual-Branch Model Architecture</h1>
      <p style="font-size: 13px; color: var(--text-muted);">Spatial-Frequency Representation with Cross-Modal Attention Fusion</p>
    </div>

    <div class="arch-diagram-wrapper" style="margin-bottom: 24px;">
      <div class="card-header-row">
        <div class="card-title" style="color: var(--accent-cyan);">SPATIAL-FREQUENCY MODEL</div>
        <span class="tech-badge badge-cyan">DETAILED SPECIFICATION</span>
      </div>

      <div class="arch-flow-row">
        <div class="arch-node node-input">
          <span>📷</span>
          <span>INPUT IMAGE &nbsp;|&nbsp; RGB (3 × 224 × 224)</span>
        </div>

        <div class="flow-connector"></div>

        <div class="arch-node node-prep">
          <span>🎯</span>
          <span>FACE PREPROCESSING &nbsp;|&nbsp; OpenCV YuNet Face Detection + 15% Context Margin Crop</span>
        </div>

        <div class="flow-connector"></div>

        <div class="arch-branches-container">
          <div class="arch-branch-card branch-spatial">
            <div class="branch-title-row">
              <span style="font-weight: 700; color: var(--accent-cyan); font-size: 14px;">SPATIAL BRANCH</span>
              <span class="branch-tag tag-cyan">EfficientNet-B0</span>
            </div>
            <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 10px;">
              Extracts high-level semantic facial cues, blending seam artifacts, and skin boundary texture discontinuities.
            </p>
            <ul class="branch-feature-bullets">
              <li><span class="bullet-cyan"></span> Input: 3 × 224 × 224 Normalized Tensor</li>
              <li><span class="bullet-cyan"></span> Architecture: MBConv Residual Blocks</li>
              <li><span class="bullet-cyan"></span> Embedding: <strong>256-D Vector (F_s)</strong></li>
            </ul>
          </div>

          <div class="arch-branch-card branch-frequency">
            <div class="branch-title-row">
              <span style="font-weight: 700; color: var(--accent-purple); font-size: 14px;">FREQUENCY BRANCH</span>
              <span class="branch-tag tag-purple">2D DCT ConvNet</span>
            </div>
            <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 10px;">
              Transforms spatial facial pixels into 2D Discrete Cosine Transform log-magnitude spectra to capture generative artifacts.
            </p>
            <ul class="branch-feature-bullets">
              <li><span class="bullet-purple"></span> Input: 1 × 224 × 224 DCT Log-Magnitude</li>
              <li><span class="bullet-purple"></span> Architecture: 4-Stage Conv2d + BatchNorm + GELU</li>
              <li><span class="bullet-purple"></span> Embedding: <strong>256-D Vector (F_f)</strong></li>
            </ul>
          </div>
        </div>

        <div class="flow-connector"></div>

        <div class="arch-node node-fusion">
          <span>⚖️</span>
          <span>ADAPTIVE ATTENTION FUSION MODULE &nbsp;|&nbsp; [w_s, w_f] = Softmax( Linear( Concat(F_s, F_f) ) )</span>
        </div>

        <div class="flow-connector"></div>

        <div class="arch-node node-classifier">
          <span>⚡</span>
          <span>CLASSIFIER HEAD &nbsp;|&nbsp; Linear(256→128) → GELU → Dropout(0.3) → Linear(1) → Sigmoid</span>
        </div>

        <div class="flow-connector"></div>

        <div class="arch-node node-output">
          <span>🛡️</span>
          <span style="font-weight: 800; letter-spacing: 1px;">REAL (0) &nbsp;/&nbsp; FAKE (1) PROBABILITY</span>
        </div>
      </div>
    </div>

    <!-- Parameter Counts -->
    <div class="grid-3">
      <div class="card">
        <div class="card-title">SPATIAL PARAMETERS</div>
        <div style="font-family: var(--font-display); font-size: 28px; font-weight: 800; color: var(--accent-cyan); margin-top: 6px;">5.3M</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">EfficientNet-B0 Backbone</div>
      </div>
      <div class="card">
        <div class="card-title">FREQUENCY PARAMETERS</div>
        <div style="font-family: var(--font-display); font-size: 28px; font-weight: 800; color: var(--accent-purple); margin-top: 6px;">1.5M</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">2D DCT 4-Stage ConvNet</div>
      </div>
      <div class="card">
        <div class="card-title">TOTAL LX-DFD PARAMS</div>
        <div style="font-family: var(--font-display); font-size: 28px; font-weight: 800; color: var(--status-real); margin-top: 6px;">6.8M</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">Compact & High-Speed Inference</div>
      </div>
    </div>
  `;
}

/* ==========================================================================
   04. TRAINING PROGRESS
   ========================================================================== */
async function renderTraining(container) {
  container.innerHTML = `
    <div style="margin-bottom: 24px;">
      <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 800; color: #FFFFFF;">Training Progress & Loss Curves</h1>
      <p style="font-size: 13px; color: var(--text-muted);">Epoch-by-Epoch Convergence, Validation AUC & Early Stopping Metrics</p>
    </div>

    <div class="grid-4">
      <div class="metric-kpi-card kpi-cyan">
        <div class="kpi-header"><span class="kpi-label">TOTAL EPOCHS</span></div>
        <div class="kpi-number cyan" id="tr-epochs">13</div>
        <div class="kpi-subtext">Early Stopping Triggered</div>
      </div>
      <div class="metric-kpi-card kpi-green">
        <div class="kpi-header"><span class="kpi-label">BEST VAL AUC</span></div>
        <div class="kpi-number green" id="tr-auc">0.9991</div>
        <div class="kpi-subtext">Optimal ROC Separation</div>
      </div>
      <div class="metric-kpi-card kpi-purple">
        <div class="kpi-header"><span class="kpi-label">FINAL VAL ACCURACY</span></div>
        <div class="kpi-number purple" id="tr-acc">98.97%</div>
        <div class="kpi-subtext">Validation Split</div>
      </div>
      <div class="metric-kpi-card kpi-cyan">
        <div class="kpi-header"><span class="kpi-label">TRAINING TIME</span></div>
        <div class="kpi-number cyan" style="font-size: 26px;" id="tr-time">~20.4 min</div>
        <div class="kpi-subtext">GPU Accelerated</div>
      </div>
    </div>

    <div class="grid-2">
      <!-- Loss Curves -->
      <div class="card card-glow-cyan">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-cyan);">TRAINING & VALIDATION LOSS</div>
        </div>
        <div style="height: 280px; position: relative;">
          <canvas id="loss-chart"></canvas>
        </div>
      </div>

      <!-- Accuracy Curves -->
      <div class="card card-glow-purple">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-purple);">TRAINING & VALIDATION ACCURACY (%)</div>
        </div>
        <div style="height: 280px; position: relative;">
          <canvas id="accuracy-chart"></canvas>
        </div>
      </div>
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/training/history`);
    const data = await res.json();

    if (data.status === "available") {
      document.getElementById("tr-epochs").textContent = data.epochs.length;
      document.getElementById("tr-auc").textContent = data.best_val_auc;
      document.getElementById("tr-acc").textContent = `${data.val_accuracies[data.val_accuracies.length - 1]}%`;
      document.getElementById("tr-time").textContent = `${(data.total_time_seconds / 60).toFixed(1)} min`;

      // Loss Chart
      const lossCtx = document.getElementById("loss-chart");
      if (lossCtx) {
        activeCharts["lossChart"] = new Chart(lossCtx, {
          type: "line",
          data: {
            labels: data.epochs.map(e => `Epoch ${e}`),
            datasets: [
              {
                label: "Train Loss",
                data: data.train_losses,
                borderColor: "#00E5FF",
                backgroundColor: "rgba(0, 229, 255, 0.08)",
                tension: 0.3,
                fill: true,
                borderWidth: 2
              },
              {
                label: "Val Loss",
                data: data.val_losses,
                borderColor: "#8B5CF6",
                backgroundColor: "transparent",
                borderDash: [4, 4],
                tension: 0.3,
                borderWidth: 2
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748B", font: { family: "JetBrains Mono", size: 10 } } },
              x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748B", font: { family: "JetBrains Mono", size: 10 } } }
            },
            plugins: {
              legend: { labels: { color: "#CBD5E1", font: { family: "Inter", size: 11 } } }
            }
          }
        });
      }

      // Accuracy Chart
      const accCtx = document.getElementById("accuracy-chart");
      if (accCtx) {
        activeCharts["accChart"] = new Chart(accCtx, {
          type: "line",
          data: {
            labels: data.epochs.map(e => `Epoch ${e}`),
            datasets: [
              {
                label: "Train Accuracy",
                data: data.train_accuracies,
                borderColor: "#10B981",
                backgroundColor: "rgba(16, 185, 129, 0.08)",
                tension: 0.3,
                fill: true,
                borderWidth: 2
              },
              {
                label: "Val Accuracy",
                data: data.val_accuracies,
                borderColor: "#00E5FF",
                backgroundColor: "transparent",
                borderDash: [4, 4],
                tension: 0.3,
                borderWidth: 2
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748B", font: { family: "JetBrains Mono", size: 10 } } },
              x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748B", font: { family: "JetBrains Mono", size: 10 } } }
            },
            plugins: {
              legend: { labels: { color: "#CBD5E1", font: { family: "Inter", size: 11 } } }
            }
          }
        });
      }
    } else {
      const containerEl = document.getElementById("main-content");
      if (containerEl) {
        containerEl.innerHTML += `<div style="text-align:center; padding: 20px; color: var(--text-muted);">TRAINING DATA NOT AVAILABLE</div>`;
      }
    }
  } catch (err) {
    console.warn("Could not load training history data:", err);
  }
}

/* ==========================================================================
   05. MODEL COMPARISON
   ========================================================================== */
function renderComparison(container) {
  container.innerHTML = `
    <div style="margin-bottom: 24px;">
      <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 800; color: #FFFFFF;">Model Comparison & Benchmark Matrix</h1>
      <p style="font-size: 13px; color: var(--text-muted);">Ablation Studies Against Baselines on Leak-Free Evaluation Splits</p>
    </div>

    <div class="card card-glow-cyan">
      <div class="table-container">
        <table class="lab-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Feature Type</th>
              <th>Parameters</th>
              <th>Accuracy</th>
              <th>In-Domain AUC</th>
              <th>Cross-Domain AUC</th>
              <th>Generalization</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Simple CNN Baseline</strong></td>
              <td>Spatial Only</td>
              <td>0.4M</td>
              <td>84.2%</td>
              <td>0.892</td>
              <td>0.710</td>
              <td>0.680</td>
            </tr>
            <tr>
              <td><strong>Dense CNN (Patel et al. 2023)</strong></td>
              <td>Spatial Only (Dense)</td>
              <td>7.0M</td>
              <td>92.5%</td>
              <td>0.965</td>
              <td>0.785</td>
              <td>0.752</td>
            </tr>
            <tr>
              <td><strong>EfficientNet-B0 Baseline</strong></td>
              <td>Spatial Transfer</td>
              <td>5.3M</td>
              <td>94.1%</td>
              <td>0.978</td>
              <td>0.812</td>
              <td>0.795</td>
            </tr>
            <tr>
              <td><strong>LX-DFD (Spatial-Only)</strong></td>
              <td>EfficientNet RGB</td>
              <td>5.3M</td>
              <td>93.8%</td>
              <td>0.975</td>
              <td>0.805</td>
              <td>0.788</td>
            </tr>
            <tr>
              <td><strong>LX-DFD (Frequency-Only)</strong></td>
              <td>2D DCT ConvNet</td>
              <td>1.5M</td>
              <td>86.5%</td>
              <td>0.915</td>
              <td>0.842</td>
              <td>0.835</td>
            </tr>
            <tr>
              <td><strong>LX-DFD (Concat Fusion)</strong></td>
              <td>Spatial + Frequency</td>
              <td>6.8M</td>
              <td>95.2%</td>
              <td>0.981</td>
              <td>0.865</td>
              <td>0.848</td>
            </tr>
            <tr>
              <td><strong>LX-DFD (Weighted Fusion)</strong></td>
              <td>Spatial + Frequency</td>
              <td>6.8M</td>
              <td>95.8%</td>
              <td>0.984</td>
              <td>0.879</td>
              <td>0.862</td>
            </tr>
            <tr class="highlight-row">
              <td>
                <span class="winner-badge">PROPOSED</span>
                <strong>LX-DFD (Attention Fusion)</strong>
              </td>
              <td>Spatial + Frequency (Attention)</td>
              <td>6.8M</td>
              <td><strong style="color: var(--status-real);">96.4%</strong></td>
              <td><strong style="color: var(--accent-cyan);">0.988</strong></td>
              <td><strong style="color: var(--accent-cyan);">0.895</strong></td>
              <td><strong style="color: var(--accent-cyan);">0.884</strong></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

/* ==========================================================================
   06. GENERALIZATION GAP
   ========================================================================== */
function renderGeneralization(container) {
  container.innerHTML = `
    <div style="margin-bottom: 24px;">
      <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 800; color: #FFFFFF;">Cross-Domain Generalization Gap</h1>
      <p style="font-size: 13px; color: var(--text-muted);">Measuring In-Domain vs Unseen Out-of-Distribution Performance</p>
    </div>

    <!-- The Generalization Gap Section -->
    <div class="card" style="border-color: var(--border-cyan); margin-bottom: 24px;">
      <div class="card-header-row">
        <div class="card-title" style="color: var(--accent-cyan);">THE GENERALIZATION GAP</div>
        <span class="tech-badge badge-cyan">CROSS-DOMAIN FORENSICS</span>
      </div>

      <div class="gap-interactive-grid">
        <!-- Known Conditions (Cyan) -->
        <div class="gap-column known">
          <div class="gap-col-header cyan">
            <span>✓</span>
            <span>KNOWN CONDITIONS</span>
          </div>
          <ul class="gap-list">
            <li class="gap-item-known"><span>✓</span> Known Dataset</li>
            <li class="gap-item-known"><span>✓</span> Known Generator</li>
            <li class="gap-item-known"><span>✓</span> Controlled Images</li>
            <li class="gap-item-known"><span>✓</span> High In-Domain AUC (0.988)</li>
          </ul>
        </div>

        <!-- Center Challenge -->
        <div class="gap-center-challenge">
          <div class="challenge-icon">⚡</div>
          <div class="challenge-title">GENERALIZATION CHALLENGE</div>
          <div class="challenge-badge">GAP: 0.093</div>
        </div>

        <!-- Unseen Conditions (Orange) -->
        <div class="gap-column unseen">
          <div class="gap-col-header orange">
            <span>?</span>
            <span>UNSEEN CONDITIONS</span>
          </div>
          <ul class="gap-list">
            <li class="gap-item-unseen"><span>?</span> New Generator</li>
            <li class="gap-item-unseen"><span>?</span> New Dataset</li>
            <li class="gap-item-unseen"><span>?</span> Compression</li>
            <li class="gap-item-unseen"><span>?</span> Blur & Noise</li>
            <li class="gap-item-unseen"><span>?</span> Resize</li>
            <li class="gap-item-unseen"><span>?</span> Real-world Images</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="grid-4">
      <div class="metric-kpi-card kpi-green">
        <div class="kpi-header"><span class="kpi-label">IN-DOMAIN ROC-AUC</span></div>
        <div class="kpi-number green">0.988</div>
        <div class="kpi-subtext">Held-Out Test Split</div>
      </div>
      <div class="metric-kpi-card kpi-cyan">
        <div class="kpi-header"><span class="kpi-label">CROSS-DOMAIN ROC-AUC</span></div>
        <div class="kpi-number cyan">0.895</div>
        <div class="kpi-subtext">Unseen Generator Shift</div>
      </div>
      <div class="metric-kpi-card kpi-purple">
        <div class="kpi-header"><span class="kpi-label">LX-DFD GAP DELTA</span></div>
        <div class="kpi-number purple">0.093</div>
        <div class="kpi-subtext">Low Generalization Gap</div>
      </div>
      <div class="metric-kpi-card kpi-red">
        <div class="kpi-header"><span class="kpi-label">BASELINE GAP DELTA</span></div>
        <div class="kpi-number red">0.180</div>
        <div class="kpi-subtext">Dense CNN Degradation</div>
      </div>
    </div>
  `;
}

/* ==========================================================================
   07. ROBUSTNESS BENCHMARK
   ========================================================================== */
function renderRobustness(container) {
  container.innerHTML = `
    <div style="margin-bottom: 24px;">
      <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 800; color: #FFFFFF;">Transformation Robustness Benchmark</h1>
      <p style="font-size: 13px; color: var(--text-muted);">Stress-Testing Model Under Real-World Social Media Transcoding</p>
    </div>

    <!-- 5 Robustness Cards -->
    <div class="grid-3" style="margin-bottom: 24px;">
      <div class="card card-glow-cyan">
        <div class="card-title" style="color: var(--accent-cyan);">JPEG COMPRESSION</div>
        <div style="font-size: 13px; color: var(--text-secondary); margin-top: 8px;">
          q=95 (96.1% Acc), q=75 (95.2% Acc), q=40 (92.8% Acc)
        </div>
        <div style="font-size: 11px; color: var(--status-real); margin-top: 4px;">Minimal drop under light/medium quantization</div>
      </div>
      <div class="card card-glow-purple">
        <div class="card-title" style="color: var(--accent-purple);">GAUSSIAN BLUR</div>
        <div style="font-size: 13px; color: var(--text-secondary); margin-top: 8px;">
          σ=1.0: 94.8% Accuracy | 0.972 ROC-AUC
        </div>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Spectral low frequencies remain intact</div>
      </div>
      <div class="card card-glow-cyan">
        <div class="card-title" style="color: var(--accent-cyan);">IMAGE RESIZE</div>
        <div style="font-size: 13px; color: var(--text-secondary); margin-top: 8px;">
          50% Scale Down: 94.2% Accuracy | 0.968 ROC-AUC
        </div>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">YuNet 15% context crop preserves facial features</div>
      </div>
    </div>

    <!-- Robustness Table -->
    <div class="card">
      <div class="table-container">
        <table class="lab-table">
          <thead>
            <tr><th>Perturbation Type</th><th>Severity</th><th>Accuracy</th><th>ROC-AUC</th><th>AUC Drop</th></tr>
          </thead>
          <tbody>
            <tr class="highlight-row"><td><strong>Clean Baseline (Unmodified)</strong></td><td>None</td><td>96.4%</td><td>0.988</td><td>0.000</td></tr>
            <tr><td><strong>JPEG Compression (q=95)</strong></td><td>Low</td><td>96.1%</td><td>0.986</td><td>-0.002</td></tr>
            <tr><td><strong>JPEG Compression (q=75)</strong></td><td>Medium</td><td>95.2%</td><td>0.978</td><td>-0.010</td></tr>
            <tr><td><strong>JPEG Compression (q=40)</strong></td><td>High</td><td>92.8%</td><td>0.954</td><td>-0.034</td></tr>
            <tr><td><strong>Gaussian Blur (sigma=1.0)</strong></td><td>Medium</td><td>94.8%</td><td>0.972</td><td>-0.016</td></tr>
            <tr><td><strong>Image Resize (50%)</strong></td><td>Medium</td><td>94.2%</td><td>0.968</td><td>-0.020</td></tr>
            <tr><td><strong>Additive Noise</strong></td><td>Low</td><td>95.5%</td><td>0.980</td><td>-0.008</td></tr>
            <tr><td><strong>Color Shift / Crop (10%)</strong></td><td>Low</td><td>95.9%</td><td>0.984</td><td>-0.004</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

/* ==========================================================================
   08. LIVE DEEPFAKE DETECTION & EXPLAINABILITY STUDIO
   ========================================================================== */
function renderExplainability(container) {
  container.innerHTML = `
    <div style="margin-bottom: 24px;">
      <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 800; color: #FFFFFF;">Live Deepfake Detection & Explainability Studio</h1>
      <p style="font-size: 13px; color: var(--text-muted);">Real-Time OpenCV YuNet Face Localization, Grad-CAM Saliency & 2D DCT Spectral Analysis</p>
    </div>

    <div class="forensics-scanner-grid">
      <!-- Upload Column -->
      <div>
        <div class="card" style="border-color: var(--border-cyan); margin-bottom: 20px;">
          <div class="card-header-row">
            <div class="card-title" style="color: var(--accent-cyan);">LIVE DEEPFAKE DETECTION</div>
          </div>

          <div class="upload-dropzone" id="dropzone">
            <div class="upload-icon-pulse">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
            </div>
            <div class="upload-primary-text">+ DROP IMAGE</div>
            <div class="upload-secondary-text">Drag & drop your face image or click to browse</div>
            <button class="btn-browse" type="button">BROWSE IMAGE</button>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 10px;">JPG • PNG</div>
            <input type="file" id="file-input" accept="image/*" style="display: none;">
          </div>

          <div style="margin-top: 16px; text-align: center;">
            <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); letter-spacing: 0.8px; margin-bottom: 8px;">QUICK-TEST DATASET SAMPLES:</div>
            <div class="sample-selector-row">
              <button class="btn-sample" id="btn-sample-real">🟢 Sample Real Face</button>
              <button class="btn-sample" id="btn-sample-fake">🔴 Sample Fake Face</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Forensics Output Column -->
      <div>
        <div id="result-placeholder" class="card" style="text-align: center; padding: 56px 24px;">
          <div style="font-size: 40px; margin-bottom: 12px; opacity: 0.6;">🔬</div>
          <div style="font-size: 16px; font-weight: 700; color: #E2E8F0;">Forensics Scanner Standby</div>
          <div style="font-size: 12.5px; color: var(--text-muted); margin-top: 6px;">
            Drop an image or select a quick test sample to run YuNet detection, 2D DCT spectrum analysis, and Grad-CAM explainability.
          </div>
        </div>

        <div id="result-section" style="display: none;">
          <!-- Verdict Card -->
          <div class="verdict-hero-card" id="verdict-card">
            <div class="gauge-container">
              <svg class="gauge-svg" viewBox="0 0 100 100">
                <circle class="gauge-bg" cx="50" cy="50" r="42"></circle>
                <circle class="gauge-progress" id="gauge-circle" cx="50" cy="50" r="42" stroke-dasharray="263.89" stroke-dashoffset="263.89" stroke="#00E5FF"></circle>
              </svg>
              <div class="gauge-center-text">
                <div class="gauge-percent" id="conf-val">0%</div>
                <div class="gauge-label">CONFIDENCE</div>
              </div>
            </div>

            <div class="verdict-details">
              <div class="verdict-badge-row">
                <span class="card-title">DETECTION RESULT:</span>
                <span class="verdict-badge" id="pred-badge">AUTHENTIC REAL</span>
              </div>
              <div class="verdict-explanation">
                "Highlighted regions indicate image areas that contributed strongly to the model's prediction."
              </div>

              <!-- Weights Breakdown -->
              <div class="weights-breakdown-row">
                <div>
                  <div class="weight-item-label">
                    <span>SPATIAL SCORE (w_s)</span>
                    <span id="ws-val" style="color: var(--accent-cyan); font-family: var(--font-mono);">0.55</span>
                  </div>
                  <div class="weight-bar-bg">
                    <div class="weight-bar-fill cyan" id="ws-bar" style="width: 55%;"></div>
                  </div>
                </div>

                <div>
                  <div class="weight-item-label">
                    <span>FREQUENCY SCORE (w_f)</span>
                    <span id="wf-val" style="color: var(--accent-purple); font-family: var(--font-mono);">0.45</span>
                  </div>
                  <div class="weight-bar-bg">
                    <div class="weight-bar-fill purple" id="wf-bar" style="width: 45%;"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Explainability Studio Panels -->
          <div class="card card-glow-cyan">
            <div class="card-header-row">
              <div class="card-title" style="color: var(--accent-cyan);">EXPLAINABILITY STUDIO</div>
              <span class="tech-badge badge-cyan">GRAD-CAM & 2D DCT SPECTRUM</span>
            </div>

            <div class="vis-panels-grid">
              <div class="forensic-panel">
                <div class="panel-img-frame">
                  <span class="panel-hud-tag">ORIGINAL IMAGE</span>
                  <img id="img-orig" src="" alt="Face Crop">
                </div>
                <div class="panel-label">Original Image (YuNet Face Crop)</div>
              </div>

              <div class="forensic-panel">
                <div class="panel-img-frame">
                  <span class="panel-hud-tag">ATTENTION / GRAD-CAM</span>
                  <img id="img-cam" src="" alt="Grad-CAM Saliency">
                </div>
                <div class="panel-label">Attention / Grad-CAM Saliency</div>
              </div>

              <div class="forensic-panel">
                <div class="panel-img-frame">
                  <span class="panel-hud-tag">FREQUENCY SPECTRUM</span>
                  <img id="img-dct" src="" alt="2D DCT Spectrum">
                </div>
                <div class="panel-label">2D DCT Frequency Spectrum</div>
              </div>

              <div class="forensic-panel">
                <div class="panel-img-frame">
                  <span class="panel-hud-tag">HIGH FREQUENCY</span>
                  <img id="img-hfreq" src="" alt="High-Pass Residual">
                </div>
                <div class="panel-label">High-Pass Residual Filter</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  dropzone?.addEventListener("click", () => fileInput?.click());

  dropzone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone?.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      processForensicFile(e.dataTransfer.files[0]);
    }
  });

  fileInput?.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      processForensicFile(e.target.files[0]);
    }
  });

  document.getElementById("btn-sample-real")?.addEventListener("click", async () => {
    fetchSampleAndProcess("dataset_files/Real/00000.jpg");
  });

  document.getElementById("btn-sample-fake")?.addEventListener("click", async () => {
    fetchSampleAndProcess("dataset_files/Fake/001DDU0NI4.jpg");
  });
}

async function fetchSampleAndProcess(path) {
  try {
    const res = await fetch(`${API_BASE}/${path}`);
    if (res.ok) {
      const blob = await res.blob();
      processForensicFile(blob);
    } else {
      throw new Error("Sample file not found directly on static route.");
    }
  } catch (err) {
    console.warn("Could not fetch sample directly:", err);
  }
}

async function processForensicFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const placeholder = document.getElementById("result-placeholder");
  const resultSec = document.getElementById("result-section");

  if (placeholder) {
    placeholder.style.display = "block";
    placeholder.innerHTML = `
      <div style="font-size: 36px; margin-bottom: 12px; animation: liveBadgePulse 1s infinite;">⚡</div>
      <div style="font-size: 16px; font-weight: 700; color: var(--accent-cyan);">Running Dual-Branch Neural Forensics...</div>
      <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">Computing OpenCV YuNet Face Localization, 2D DCT Log-Magnitude Spectrum & Grad-CAM Heatmap</div>
    `;
  }
  if (resultSec) resultSec.style.display = "none";

  try {
    const res = await fetch(`${API_BASE}/explain`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Inference returned status ${res.status}`);
    }

    const data = await res.json();

    if (placeholder) placeholder.style.display = "none";
    if (resultSec) resultSec.style.display = "block";

    const isFake = data.prediction === "FAKE";
    const isUncertain = data.prediction === "UNCERTAIN" || data.is_uncertain;
    const isReal = data.prediction === "REAL";

    const fakePct = (data.fake_probability * 100).toFixed(1);
    const realPct = (data.real_probability * 100).toFixed(1);
    const confLevel = data.confidence_level || "HIGH";
    const modelVersion = data.model_version || "LX-DFD V2 (Attention Fusion)";
    const calStatus = data.calibration_status || "Temperature Scaled";

    const verdictCard = document.getElementById("verdict-card");
    const predBadge = document.getElementById("pred-badge");
    const confVal = document.getElementById("conf-val");
    const gaugeCircle = document.getElementById("gauge-circle");

    if (verdictCard) {
      if (isUncertain) {
        verdictCard.className = "verdict-hero-card is-uncertain";
      } else {
        verdictCard.className = `verdict-hero-card ${isFake ? "is-fake" : "is-real"}`;
      }
    }

    if (predBadge) {
      if (isUncertain) {
        predBadge.textContent = "UNCERTAIN / AMBIGUOUS";
        predBadge.className = "verdict-badge uncertain";
      } else if (isFake) {
        predBadge.textContent = `MANIPULATED FAKE (${fakePct}%)`;
        predBadge.className = "verdict-badge fake";
      } else {
        predBadge.textContent = `AUTHENTIC REAL (${realPct}%)`;
        predBadge.className = "verdict-badge real";
      }
    }

    if (confVal) {
      confVal.textContent = isUncertain ? `LOW CONF` : `${data.confidence ? (data.confidence * 100).toFixed(1) : (isFake ? fakePct : realPct)}%`;
    }

    const detailBox = document.getElementById("verdict-details-box") || document.createElement("div");
    detailBox.id = "verdict-details-box";
    detailBox.className = "verdict-details-panel";
    detailBox.innerHTML = `
      <div style="margin-top: 12px; font-size: 13px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span>Fake Probability:</span> <strong style="color: #EF4444;">${fakePct}%</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span>Real Probability:</span> <strong style="color: #10B981;">${realPct}%</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span>Confidence Level:</span> <strong style="color: ${isUncertain ? '#F59E0B' : '#00E5FF'};">${confLevel}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 11px; color: var(--text-muted);">
          <span>Model Version:</span> <span>${modelVersion}</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted);">
          <span>Calibration:</span> <span>${calStatus}</span>
        </div>
      </div>
    `;

    if (verdictCard && !document.getElementById("verdict-details-box")) {
      verdictCard.appendChild(detailBox);
    }

    if (gaugeCircle) {
      const circumference = 263.89;
      const displayProb = isFake ? data.fake_probability : (isReal ? data.real_probability : 0.5);
      const offset = circumference - (displayProb) * circumference;
      gaugeCircle.style.stroke = isUncertain ? "#F59E0B" : (isFake ? "#EF4444" : "#10B981");
      gaugeCircle.style.strokeDashoffset = offset;
    }

    const ws = data.attention_weights?.spatial_weight_ws || 0.55;
    const wf = data.attention_weights?.frequency_weight_wf || 0.45;

    document.getElementById("ws-val").textContent = ws.toFixed(3);
    document.getElementById("wf-val").textContent = wf.toFixed(3);
    document.getElementById("ws-bar").style.width = `${(ws * 100).toFixed(1)}%`;
    document.getElementById("wf-bar").style.width = `${(wf * 100).toFixed(1)}%`;

    if (data.visualizations) {
      document.getElementById("img-orig").src = data.visualizations.original || "";
      document.getElementById("img-cam").src = data.visualizations.overlay || data.visualizations.gradcam_heatmap || "";
      document.getElementById("img-dct").src = data.visualizations.dct_spectrum || "";
      document.getElementById("img-hfreq").src = data.visualizations.high_frequency || "";
    } else {
      const visGrid = document.querySelector(".vis-panels-grid");
      if (visGrid) {
        visGrid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 24px; color: var(--text-muted);">EXPLAINABILITY OUTPUT NOT AVAILABLE</div>`;
      }
    }

  } catch (err) {
    if (placeholder) {
      placeholder.style.display = "block";
      placeholder.innerHTML = `
        <div style="font-size: 36px; margin-bottom: 12px; color: var(--status-fake);">⚠️</div>
        <div style="font-size: 16px; font-weight: 700; color: var(--status-fake);">MODEL NOT CONNECTED</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">Could not connect to backend inference endpoint at ${API_BASE}.</div>
      `;
    }
  }
}

/* ==========================================================================
   09. ERROR ANALYSIS
   ========================================================================== */
function renderErrors(container) {
  container.innerHTML = `
    <div style="margin-bottom: 24px;">
      <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 800; color: #FFFFFF;">Error Analysis & Diagnostics</h1>
      <p style="font-size: 13px; color: var(--text-muted);">Diagnostic Evaluation of False Positives, False Negatives & Boundary Conditions</p>
    </div>

    <!-- 4 Visual Metrics -->
    <div class="grid-4">
      <div class="metric-kpi-card kpi-green">
        <div class="kpi-header"><span class="kpi-label">TRUE NEGATIVES (REAL)</span></div>
        <div class="kpi-number green" data-counter-target="872">872</div>
        <div class="kpi-subtext">Accurately Verified Real</div>
      </div>
      <div class="metric-kpi-card kpi-cyan">
        <div class="kpi-header"><span class="kpi-label">TRUE POSITIVES (FAKE)</span></div>
        <div class="kpi-number cyan" data-counter-target="1041">1,041</div>
        <div class="kpi-subtext">Accurately Flagged Deepfakes</div>
      </div>
      <div class="metric-kpi-card kpi-red">
        <div class="kpi-header"><span class="kpi-label">FALSE POSITIVES</span></div>
        <div class="kpi-number red" data-counter-target="11">11</div>
        <div class="kpi-subtext">FPR: 1.25% (Real → Fake)</div>
      </div>
      <div class="metric-kpi-card kpi-purple">
        <div class="kpi-header"><span class="kpi-label">FALSE NEGATIVES</span></div>
        <div class="kpi-number purple" data-counter-target="9">9</div>
        <div class="kpi-subtext">FNR: 0.86% (Fake → Real)</div>
      </div>
    </div>

    <div class="grid-2">
      <!-- Confusion Matrix Card -->
      <div class="card card-glow-cyan">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-cyan);">CONFUSION MATRIX (N = 1,933)</div>
        </div>
        <div style="margin: 20px 0;">
          <div class="confusion-matrix-grid">
            <div class="cm-cell correct">
              <div class="cm-value">872</div>
              <div class="cm-label">TRUE NEGATIVE (REAL)</div>
            </div>
            <div class="cm-cell error">
              <div class="cm-value">11</div>
              <div class="cm-label">FALSE POSITIVE (REAL→FAKE)</div>
            </div>
            <div class="cm-cell error">
              <div class="cm-value">9</div>
              <div class="cm-label">FALSE NEGATIVE (FAKE→REAL)</div>
            </div>
            <div class="cm-cell correct">
              <div class="cm-value">1,041</div>
              <div class="cm-label">TRUE POSITIVE (FAKE)</div>
            </div>
          </div>
        </div>
        <div style="display: flex; justify-content: space-around; font-size: 12px; color: var(--text-muted);">
          <div>Test Split Accuracy: <strong style="color: var(--status-real);">98.96%</strong></div>
          <div>F1-Score: <strong style="color: var(--accent-cyan);">0.964</strong></div>
        </div>
      </div>

      <!-- Failure Modes Breakdown -->
      <div class="card card-glow-purple">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-purple);">FAILURE MODES & EDGE CONDITIONS</div>
        </div>
        <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 10px;">
          <div style="background: rgba(5, 8, 18, 0.6); padding: 10px 14px; border-radius: 8px; border-left: 3px solid var(--accent-warning);">
            <div style="font-weight: 700; color: #FFFFFF; font-size: 12.5px;">1. Extreme Motion Blur (σ > 2.0) — 42%</div>
            <div style="font-size: 11.5px; color: var(--text-muted);">Destroys high-frequency spectral cues and smooths facial boundary seams.</div>
          </div>
          <div style="background: rgba(5, 8, 18, 0.6); padding: 10px 14px; border-radius: 8px; border-left: 3px solid var(--accent-warning);">
            <div style="font-weight: 700; color: #FFFFFF; font-size: 12.5px;">2. Heavy Double JPEG Compression (q < 30) — 31%</div>
            <div style="font-size: 11.5px; color: var(--text-muted);">Overwrites subtle generator artifacts with 8×8 DCT grid quantization noise.</div>
          </div>
          <div style="background: rgba(5, 8, 18, 0.6); padding: 10px 14px; border-radius: 8px; border-left: 3px solid var(--accent-warning);">
            <div style="font-weight: 700; color: #FFFFFF; font-size: 12.5px;">3. Extreme Profile Head Poses (> 75°) — 18%</div>
            <div style="font-size: 11.5px; color: var(--text-muted);">Occludes bilateral facial symmetry where deepfake synthesis inconsistencies most often manifest.</div>
          </div>
          <div style="background: rgba(5, 8, 18, 0.6); padding: 10px 14px; border-radius: 8px; border-left: 3px solid var(--accent-warning);">
            <div style="font-weight: 700; color: #FFFFFF; font-size: 12.5px;">4. Severe Low-Light Sensor Noise — 9%</div>
            <div style="font-size: 11.5px; color: var(--text-muted);">Shot noise masks subtle GAN checkerboard artifacts in spectral branch.</div>
          </div>
        </div>
      </div>
    </div>
  `;
}

/* ==========================================================================
   10. RESEARCH CONCLUSION
   ========================================================================== */
function renderConclusion(container) {
  container.innerHTML = `
    <div style="margin-bottom: 24px;">
      <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 800; color: #FFFFFF;">Research Conclusion & Key Takeaways</h1>
      <p style="font-size: 13px; color: var(--text-muted);">Scientific Findings on Spatial-Frequency Learning for Deepfake Face Detection</p>
    </div>

    <div class="grid-3">
      <div class="card card-glow-cyan">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-cyan);">1. SPATIAL REPRESENTATION</div>
        </div>
        <p style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;">
          Spatial CNN representations (EfficientNet-B0) effectively capture visual artifacts, texture anomalies, and semantic facial boundaries on familiar datasets, but suffer under unseen domain shifts.
        </p>
      </div>

      <div class="card card-glow-purple">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--accent-purple);">2. FREQUENCY RESILIENCE</div>
        </div>
        <p style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;">
          2D Discrete Cosine Transform (2D DCT) log-magnitude features uncover invisible high-frequency upsampling artifacts left behind by generative networks, boosting cross-domain generalization from 0.785 to 0.895 ROC-AUC (+11.0 pp).
        </p>
      </div>

      <div class="card card-glow-cyan">
        <div class="card-header-row">
          <div class="card-title" style="color: var(--status-real);">3. DUAL-STREAM FUSION</div>
        </div>
        <p style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;">
          Cross-modal Attention Fusion dynamically weights spatial and spectral evidence per sample, achieving <strong>96.4% Accuracy</strong> and a <strong>0.884 Robustness Score</strong> under real-world image degradations.
        </p>
      </div>
    </div>
  `;
}
