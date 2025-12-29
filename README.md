GeoAdapt-style domain adaptation training pipeline (audio)

This repository contains a small domain-adaptive training pipeline that uses:
- `models/feature_extractor.py`: backbone Cnn14LogMel
- `models/classifier.py`: classifier head
- `data/dataloader.py`: ASCDataset loader that reads dataset JSON and .npy features
- `objectives/geo_adapt.py`: GeoAdapt-like adaptation losses (MMD/CORAL + entropy)
- `train.py`: training entrypoint
- `smoke_tests/smoke_test.py`: creates synthetic data and runs a one-epoch smoke test

Setup (recommended in a virtualenv):

Windows (cmd.exe):

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Run smoke test:

    python smoke_tests\smoke_test.py

Run training manually:

    python train.py --config configs/default.yaml

Notes:
- This implementation uses a simple MMD/CORAL adaptation loss as a substitute for the GeoAdapt objectives (offline-friendly).
- The dataset JSON should follow the sample format in `data/sample_dataset.json` (fields: file, label, domain). Features are expected as .npy arrays.

