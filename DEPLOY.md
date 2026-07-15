# Deploying the Pedestrian Fatality Explorer to Hugging Face Spaces

HF deprecated the Streamlit SDK -- Streamlit apps now use the **Docker SDK**.
Everything needed is in this folder:
  Dockerfile, README.md, app.py, backend_route.py, requirements.txt, ped_map_data.parquet

## Steps (10 minutes)

1. Go to https://huggingface.co/new-space
   - Owner: your account | Space name: e.g. `pedestrian-fatality-explorer`
   - SDK: **Docker** (choose "Blank" template if asked)
   - Hardware: CPU basic (free) | Visibility: Public

2. Upload ALL files from this folder EXCEPT DEPLOY.md and __pycache__
   (Files tab -> Contribute -> Upload files). The README.md contains the
   Space config (sdk: docker, app_port: 8501) -- upload it as-is.

3. Optional (only for the route-check mode):
   Settings -> Variables and secrets -> New secret
     Name:  ORS_API_KEY
     Value: <free key from https://openrouteservice.org>
   Explore mode works without it.

4. Build takes ~3-5 min. Test on your phone: Alabama dots should load,
   tapping a dot shows the narrative.

5. Send Claude the URL -> QR code + poster screenshots.

## Notes
- Free Spaces SLEEP after ~48h of no traffic; first visitor waits ~1-2 min.
  Open the app on your phone the morning of the poster session to warm it.
- To update data after a model change: replace ped_map_data.parquet in the
  Space (pipeline: 02 -> 03 -> 04 labels -> 05 -> 06).
