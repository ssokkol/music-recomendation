# Music Recommender (Favorite Tracks)

Pet project: music recommendations from favorite tracks + a dark-themed web page.

## What this project does
- Loads a small dataset of favorite tracks from CSV.
- Trains a tiny "from scratch" neural network in pure Python.
- Trains a deeper model in Keras.
- Generates track recommendations (demo).
- Provides a dark-themed web page with purple + teal accents.

## Coverage checklist
- Python data types and files: `src/utils_io.py`, `data/sample_tracks.csv`
- Basic neural network in Python: `src/simple_nn.py`
- Deep learning in Keras: `src/keras_model.py`
- Markup and web app concepts: `web/index.html`
- Web pages: semantic HTML + CSS layout
- Links and illustrations: external links + `web/assets/cover.svg`
- Frames and forms: iframe + form inputs

## Repo structure
- `data/sample_tracks.csv` - demo dataset.
- `src/utils_io.py` - file I/O helpers.
- `src/data_prep.py` - basic preprocessing.
- `src/simple_nn.py` - minimal neural network (no frameworks).
- `src/keras_model.py` - Keras model example.
- `web/index.html` - dark themed page with links, images, iframe, forms.
- `web/styles.css` - styles and animations.
- `web/assets/cover.svg` - illustration.

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/simple_nn.py
python src/keras_model.py
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```

## Web page
Open `web/index.html` in a browser. The page includes:
- dark theme, purple + teal accents
- external links
- images (SVG)
- iframe (video embed)
- forms and inputs

## License
MIT
