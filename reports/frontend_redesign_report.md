# Frontend Architecture & Interface Redesign Report
## Transforming a Prototype Dashboard into a Research-Grade Forensics Workstation

**Author:** Senior Product Designer & Scientific Visualization Engineer  
**Document Identifier:** `REP-UI-2026-V1`  
**Technology Stack:** Vanilla HTML5, Vanilla CSS3 (Custom Design System), JavaScript (ES6+), Chart.js 4.4, FastAPI Backend  
**Design Paradigm:** Deep Charcoal AI Intelligence Workstation with Restrained Cyan & Violet Accents  

---

## 1. Executive Summary & Aesthetic Transformation

The previous LX-DFD user interface suffered from standard prototype pitfalls: excessive neon glow effects, giant low-information confidence gauges, dense unstructured cards, decorative percentages, and confusingly mislabeled telemetry (such as presenting fusion attention weights as branch probabilities).

The goal of this redesign was to transform the interface into a **tier-1 scientific forensics workstation** suitable for digital forensics laboratories, intelligence analysts, academic researchers, and judicial proceedings. The new workstation prioritizes:
1. **Visual Restraint & Clarity:** Deep charcoal palettes (`#080D17`, `#0D1422`, `#121C2D`) that minimize eye fatigue during extended forensic sessions.
2. **Scientific Integrity:** Visual distinction between raw model outputs, temperature-calibrated probabilities, measured evaluation benchmarks, and qualitative research interpretations.
3. **Traceability:** Every metric, chart, and heatmap is connected to reproducible backend endpoints, versioned checkpoints, and cryptographic hashes.

---

## 2. Design System & Typography Specifications

```
Color Palette Token Architecture
=============================================================================
Token Name                 Hex Code       Role & Semantic Usage
-----------------------------------------------------------------------------
--bg-app                   #080D17        Primary workstation backdrop
--bg-surface               #0D1422        Standard card & container background
--bg-surface-elevated      #121C2D        Header, sidebar, and elevated panels
--bg-input                 #0A0F1A        Dropzone and preview framing
--border-subtle            #1A253A        Default structural divider
--border-medium            #24334E        Interactive card border
--text-primary             #F1F5F9        Headings, verdicts, primary telemetry
--text-secondary           #CBD5E1        Standard body copy, card subheaders
--text-muted               #94A3B8        Labels, metric captions, hints
--accent-cyan              #00E5FF        Spatial stream accent, interactive focus
--accent-violet            #8B5CF6        Frequency stream accent, research tools
--status-real              #10B981        Authentic verified real (Emerald)
--status-fake              #EF4444        Manipulated synthetic fake (Crimson)
--status-uncertain         #F59E0B        Ambiguous / OOD Advisory (Amber)
=============================================================================
```

### Typography Hierarchy
* **Interface Text:** `Inter` (sans-serif, weights 400, 500, 600) — clean legibility at small sizes.
* **Structural Headings:** `Space Grotesk` (weights 600, 700) — scientific authority and distinct branding.
* **Telemetry & Numeric Values:** `JetBrains Mono` (weights 400, 500, 700) — tabular alignment, preventing layout jitter as probabilities update.

---

## 3. Workstation Layout Architecture

The workstation utilizes a full-viewport, zero-clutter grid optimized for **1366×768 laptop displays and larger dual-monitor forensic setups**:

```
┌──────────────┬─────────────────────────────────────────────────────────────┐
│ SIDEBAR      │ TOP ACTION BAR: Health Status • Active Checkpoint • IEEE    │
│  (240px)     ├─────────────────────────────────────────────────────────────┤
│              │ MAIN CONTENT WORKSPACE                                      │
│ • Live Lab   │                                                             │
│ • Crop Lab   │  ┌──────────────┬───────────────────┬────────────────────┐  │
│ • Shortcut   │  │ LEFT COLUMN  │ CENTER COLUMN     │ RIGHT COLUMN       │  │
│ • Counterf.  │  │ Image Input  │ Forensic Verdict  │ Dual-Branch Stream │  │
│ • Overview   │  │ Dropzone &   │ Calibrated Prob.  │ Spatial Efficient  │  │
│ • Model Arch │  │ YuNet Face   │ Reliability Badge │ Frequency 2D-DCT   │  │
│ • Dataset    │  │ Localization │ Uncertainty Banner│ Dynamic Attention  │  │
│ • Training   │  └──────────────┴───────────────────┴────────────────────┘  │
│ • Comparison │  ┌───────────────────────────────────────────────────────┐  │
│ • Generaliz. │  │ FOUR-PANEL VISUAL EVIDENCE SUITE                      │  │
│ • Robustness │  │ Face Crop • Grad-CAM • 2D-DCT Log • High-Pass Spatial │  │
│ • Errors     │  └───────────────────────────────────────────────────────┘  │
│ • Reports    ├─────────────────────────────────────────────────────────────┤
│              │ STATUS FOOTER: Model Version • YuNet ONNX • PyTorch 2D-DCT  │
└──────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Component Engineering

### 4.1 Three-Column Live Forensics Suite
* **Left Column (Image Input & Localization):**
  * Drag-and-drop file upload with 15MB safety ceiling.
  * Real-time preview with dimensions ($W \times H$) and file size.
  * Preset sample selectors (Authentic Real, SD 1.5, StyleGAN2, Gemini Imagen 3) for zero-latency testing.
  * YuNet face crop coordinate indicator (`[x1, y1, x2, y2]`).
* **Center Column (Forensic Verdict & Calibration):**
  * Executive Verdict Badge (`AUTHENTIC REAL`, `SYNTHETIC FAKE`, or `UNCERTAIN`).
  * Horizontal probability split bar with calibrated percentages.
  * Empirical Reliability Indicator (`HIGH`, `MODERATE`, `UNCERTAIN`).
  * Dynamic Out-of-Distribution Scope Banner when branch discrepancy exceeds 15%.
  * "Generate Official Forensic Certificate" action button.
* **Right Column (Dual-Branch Stream Telemetry):**
  * Explicit separation of **Branch Predictions** ($P_s, P_f$) from **Attention Weights** ($w_s, w_f$).
  * Dynamic attention weighting bar visualizing the relative contribution of spatial vs. spectral evidence.
  * Branch agreement/divergence diagnostic telemetry.

### 4.2 Four-Panel Synchronized Visual Evidence Suite
Displays synchronized, multi-modal feature representations side-by-side:
1. **Original Detected Face Crop:** Standard portrait context (margin 0.50).
2. **Grad-CAM Activation Heatmap:** Superimposed over facial anatomy, highlighting the exact receptive field regions driving the spatial classification.
3. **2D-DCT Log-Magnitude Spectrum:** Colored spectral distribution showing low-frequency energy vs. high-frequency peripheral harmonics.
4. **High-Pass Spatial Residual:** Laplacian filter output isolating fine edges, skin pores, and noise textures.
* **Interactive Lightbox:** Clicking any panel opens a high-resolution modal inspection view.

### 4.3 New Research Features
* **Crop Stability Lab (Module 02):** Multi-margin evaluation cards (margins 0.15, 0.50, 0.75) with probability delta ($\Delta$) and stability classification.
* **Dataset Shortcut Detector (Module 03):** 6-panel controlled diagnostic occlusion suite (background masking, facial interior, border-only, resolution normalization, web compression) with automated risk categorization.
* **Counterfactual Robustness Lab (Module 04):** 9-card perturbation matrix testing JPEG Q=95/75/40, Gaussian blur, sensor noise, downsampling, grayscale, and exposure shifts with verdict flipping detection.

### 4.4 Executive Forensic Report Generation
Clicking "Generate Official Forensic Certificate" launches a printable executive modal:
* Formatted in high-contrast print typography (`@media print` clean layout).
* Embeds candidate image, face localization bounding box, active checkpoint SHA-256 hash, spatial/frequency attention weights, and temperature calibration vector.
* Features a single-click **"Print / Save PDF"** action and a **"Download Audit JSON"** export button for legal chain-of-custody logging.

---

## 5. Performance, Compatibility & Lifecycle Validation

* **Zero Build Frameworks:** Built entirely with native ES6 JavaScript and Vanilla CSS. No React, Vite, Node build pipelines, or webpack bundlers. Serves immediately from FastAPI StaticFiles.
* **Chart.js Lifecycle Safety:** The custom `destroyActiveCharts()` lifecycle hook guarantees that canvas contexts are purged before navigating between research modules, eliminating memory leaks and overlapping canvas artifacts.
* **Browser Compatibility:** Verified in modern Chromium, Firefox, and WebKit browsers.

---

*Report certified by LX-DFD Frontend Design & Human-Computer Interaction Team.*
