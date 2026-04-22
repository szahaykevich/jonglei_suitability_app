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

- The current `app/main.py` uses demo grids so you can test slider behavior and hover values quickly.
- Replace demo arrays with your processed rasters (`fhi.tif`, `svi.tif`, and final suitability output) as your next step.
