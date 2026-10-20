# Corrected comparison: only four requested methods

1. Authors hand-labelled FCN (v0v0 notebook; checkpoint trained on dataset v1.0).
2. Your original hand-labelled FCN v1.1, epoch051. Not the separately corrected epoch72 model.
3. Your current hand-labelled CROMA + FCN, own float preprocessing.
4. Stored Otsu baseline.

All evaluated on v1.1. No S1-weak, S2-weak or permanent-water-trained model is included.

Ten 16:9 figures: scores, category scores, six full-scene comparisons, two error close-ups. Each is a 3200x1800 PNG and PDF. `all_figures.pdf` contains all ten.

Figures use common valid pixels (valid labels and a prediction from every method). Otsu missing predictions are excluded for all models. Fully unscored images are omitted. Therefore values may differ from earlier tables. summary.csv also includes native-mask scores matching the existing CROMA convention. Truly dry empty-union IoU is 1. Category exclusion rules are preserved.

Case selection: fixed seed42 among chips with >=10000 common-valid pixels and >=1000 labeled water pixels. No prediction scores are used to select full scenes. Close-ups explicitly maximize disagreement within selected scenes.

PNG/PDF titles are in English for presentation use. CROMA and personal FCN run identifiers and checkpoint provenance are in provenance.json.