# 🦴 CPAK · Automated Coronal Alignment Measurement

Deep learning tool for **automated coronal alignment measurement on long-limb
radiographs (LLR)**. It detects the hip, knee and ankle, localizes anatomical
landmarks, and computes the femorotibial angles used in knee surgery planning.

## What it computes

| Metric | Description |
|:---|:---|
| **HKA** | Hip-Knee-Ankle angle |
| **mLDFA** | Mechanical lateral distal femoral angle |
| **mMPTA** | Mechanical medial proximal tibial angle |
| **aHKA** | Arithmetic HKA (mMPTA − mLDFA) |
| **JLO** | Joint line obliquity (mMPTA + mLDFA) |
| **MAD** | Mechanical axis deviation from the Mikulicz line (px) |
| **CPAK** | Automatic CPAK phenotype classification (I–IX) |

## How to use

1. Upload a long-limb radiograph, **or paste a screenshot directly with `Ctrl+V`**.
2. Press **▶ Ejecutar inferencia**.
3. Review the metrics, the overlay (bounding boxes, anatomical axes, Mikulicz line)
   and the per-joint zoom.
4. Optionally drag the detected landmarks in the zoom panel to correct them —
   metrics are recalculated automatically.
5. Download the results as JSON or as a rendered JPG overlay.

> The interface is in Spanish (the target clinical users are Spanish-speaking).

## Model

| Stage | Architecture | Training data |
|:---|:---|:---|
| Joint detection | YOLO (open source) | 280 radiographs (80/20 split) |
| Landmark localization | YOLO pose (fine-tuned) | 1,734 anatomical crops |

Both models are included in this repository (≈40 MB total).
CPU inference takes **≈0.7–3 s per radiograph**.

## Validation

Validated against two expert surgeons on **112 limbs from 57 LLR**, independent
of training.

| Metric | Agreement with experts (ICC[2,1]) | 95% CI | Bias |
|:---|:---:|:---:|:---:|
| HKA | 0.995 | 0.99–1.00 | +0.20° |
| mLDFA | 0.873 | 0.83–0.91 | −0.15° |
| mMPTA | 0.873 | 0.80–0.92 | −0.81° |

Interobserver agreement between the two experts: HKA 0.997, mLDFA 0.930,
mMPTA 0.928.

**Sample size:** calculated a priori for two raters (α = 0.05, 80% power,
expected ICC 0.90 vs. minimum acceptable 0.70) → 19 subjects required. The study
cohort provides **>99% power** to detect ICCs above 0.70 for all three angles.

## ⚠️ Privacy notice

This is a **research and educational demo**, not a certified medical device.

**Do not upload identifiable patient data.** Radiographs may contain personal
identifiers (names, national ID numbers, dates). Uploading them to a third-party
hosting service may violate data-protection regulations (e.g. Chile's Ley 21.719,
GDPR, HIPAA). Use anonymised images only.

Results must always be reviewed by a qualified surgeon and are **not** intended
for standalone clinical decision-making.

## License

Released under the **GNU AGPL-3.0** license. This is required because Ultralytics
YOLO is AGPL-3.0: providing this service over a network obliges the operator to
offer the complete corresponding source code to its users. See `LICENSE` and
`NOTICE.md`.

## Run locally

```bash
conda create -n cpak python=3.10 -y
conda activate cpak
pip install -r requirements.txt
streamlit run scripts/30_streamlit_app.py
```

## References

Walter SD, Eliasziw M, Donner A. *Sample size and optimal designs for reliability
studies.* Stat Med. 1998;17(1):101-110.

Bonett DG. *Sample size requirements for estimating intraclass correlations with
desired precision.* Stat Med. 2002;21(9):1331-1335.
