# Detection Testing Tool (local, presentation-only)

Standalone Streamlit dashboard to upload an image/video and verify slot-occupancy
detection. Runs **locally on the laptop** — fully independent from the production
backend and the edge device. Nothing here talks to FastAPI or the Jetson.

## Run

```bash
cd testing
python -m venv .venv && .venv\Scripts\activate      # optional
pip install -r requirements-testing.txt
streamlit run app.py
```

Opens at http://localhost:8501. The admin panel embeds this URL at
`/admin/testing`, so start this tool first, then open that page during the demo.

## Layout

```
testing/
  app.py                    # Streamlit dashboard
  models/                   # detection weights (best.pt / best.onnx)
  assets/roi/               # ROI polygon coordinate files
  assets/samples/           # bundled sample images
  .streamlit/config.toml    # headless + iframe-friendly config
```

## Notes

- Target detection classes are referenced by **numeric class id only**
  (`_TARGET_IDS` in `app.py`) — no textual labels are stored or shown.
- If your weights use a different class map, edit `_TARGET_IDS`.
- CPU inference on video is slow; raise/lower `imgsz` in the sidebar to trade
  speed vs accuracy.
