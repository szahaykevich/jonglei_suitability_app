# jonglei-suitability-app

Interactive suitability map prototype for UNMISS planning support in Jonglei State, South Sudan.

## Repository structure

```text
jonglei-suitability-app/
│
├── app/
│   ├── main.py
│   ├── raster_utils.py
│   ├── weights.py
│   └── popup_logic.py
│
├── notebooks/
│   └── suitability_map_prototype.ipynb
│
├── data/
│   ├── processed/
│   │   ├── jonglei_boundary.geojson
│   │   └── README.md
│   └── sample/
│       └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app/main.py
   ```

## Notes

- `app/main.py` now loads configured reclassified rasters from `data/processed/` by default.
- If local TIFFs are unavailable (or Git LFS pointers are present), the app gracefully falls back to placeholders and shows a warning banner.
- To point at a different raster folder, set `JONGLEI_RASTER_DIR` before launching the app.
