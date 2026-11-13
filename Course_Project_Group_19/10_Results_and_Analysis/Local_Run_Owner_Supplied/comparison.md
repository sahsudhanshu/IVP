# Experimental Comparison: Historical Baseline vs. Data Augmentation Pipeline

This report compares the historical baseline results from our team's presentation against the new **Clip-Consistent Data Augmentation Pipeline**.

---

## 1. Multi-Protocol Performance Matrix

| Evaluation Protocol | Historical Baseline (Results Slide) | **Current Augmentation Pipeline (12 Epochs)** | **Delta vs. Historical Baseline** | Base Paper [1] (80 Epochs) |
|:---|:---:|:---:|:---:|:---:|
| **Cross-subject** | 71.8% (Macro-F1: 0.708) | **79.2% (Macro-F1: 0.785)** | **+7.4%** (+0.077 F1) | 93.81% |
| **Cross-view → 1** | 71.3% (Macro-F1: 0.702) | **79.4% (Macro-F1: 0.788)** | **+8.1%** (+0.086 F1) | 82.58% |
| **Cross-view → 2** | 70.2% (Macro-F1: 0.691) | **77.8% (Macro-F1: 0.771)** | **+7.6%** (+0.080 F1) | 81.00% |
| **Cross-view → 3 (Held-out Target)** | **69.6% (Macro-F1: 0.681)** | **78.5% (Macro-F1: 0.785)** | **+8.9%** (+0.104 F1) | **89.76%** |
| **Cross-view Mean** | **70.4% (Macro-F1: 0.691)** | **78.6% (Macro-F1: 0.781)** | **+8.2%** (+0.090 F1) | **84.40%** |

---

## 2. Technical Comparison of Implementation Setups

| Parameter | Historical Baseline (Results Slide) | Current Data Augmentation Pipeline |
|:---|:---|:---|
| **Backbone Feature Extractor** | Inception-v3 (Pretrained ImageNet) | Inception-v3 (Pretrained ImageNet) |
| **Temporal Sampling** | 4 ordered segments (TSN) | 4 ordered segments with segment jittering |
| **Temporal Flip Consistency** | Frame-independent flip | **Strict clip-consistent flip ($p=0.5$)** |
| **Photometric Regularization** | None | **Contrast $[0.9, 1.1]$, Brightness $[-8, 8]$ ($p=0.5$)** |
| **Sequential Modeling** | Shared GRU (hidden dim 128) | Shared GRU (hidden dim 128) |
| **Multi-View Fusion** | Concatenation of view GRU states | Concatenation of view GRU states |
| **Training Epochs** | 12 epochs | 12 epochs |
| **Learning Rate Schedule** | 0.001 (decay $\div 10$ at ep 4, 9) | 0.001 (decay $\div 10$ at ep 4, 9) |
| **View 3 Target Accuracy** | **69.6%** (291 / 418 correct) | **78.5%** (328 / 418 correct) |
| **View 3 Target Macro-F1** | **0.681** | **0.785** |

---

## 3. Analysis of Empirical Improvements

### A. Fixing Temporal Motion Inconsistency
* The historical baseline suffered from jitter when applying transformations frame-by-frame.
* By enforcing **clip-consistent transformations** (sampling the horizontal flip once per video clip and applying it synchronously to all 4 temporal snippets), coherent motion trajectories (e.g., throwing arm swing, sitting downward transition) are preserved for GRU recurrent modeling.

### B. Overcoming Camera Viewpoint & Illumination Shift
* NW-UCLA Camera 3 observes the action from an elevated diagonal angle with distinct contrast and shadow patterns compared to Cameras 1 and 2.
* Clip-consistent photometric perturbation forces the convolutional feature extractor to generalize across lighting variations rather than overfitting to source camera sensor properties.

### C. Quantitative Gains
1. **Target View 3 Gain**: Accuracy increased by **+8.9 percentage points** ($69.6\% \rightarrow 78.5\%$), correctly classifying $37$ additional video groups.
2. **Macro-F1 Jump**: F1 increased from **0.681 to 0.785**, indicating balanced improvements across all action classes.
3. **Cross-View Mean**: Advanced from **70.4% to 78.6%**, closing the gap to the base paper's 80-epoch target ($84.4\%$).
