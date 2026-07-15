# Deploy to Streamlit Community Cloud (free)

## One-time setup (15 min)

1. Create a PUBLIC GitHub repo, e.g. `ped-fatality-explorer`
   (public is required on the free tier).

2. Push this folder's files to it:
     app.py, backend_route.py, requirements.txt,
     ped_map_data.parquet, .gitignore
   (skip: Dockerfile, README.md, DEPLOY*.md, __pycache__)

   From this folder:
     git init
     git add app.py backend_route.py requirements.txt ped_map_data.parquet .gitignore
     git commit -m "Pedestrian Fatality Explorer - ITE 2026 poster app"
     git branch -M main
     git remote add origin https://github.com/<YOU>/ped-fatality-explorer.git
     git push -u origin main

3. Go to https://share.streamlit.io -> Sign in with GitHub -> "Create app"
   -> pick the repo, branch main, main file `app.py` -> Deploy.

4. Route mode key (optional): app page -> Settings -> Secrets, paste:
     ORS_API_KEY = "your-key-here"
   (free key: https://openrouteservice.org)

5. First build ~3-4 min. Test on phone. Send Claude the URL
   (looks like https://<something>.streamlit.app) for the QR code.

## Poster-day tip
Free apps sleep after ~7 idle days and wake in ~30-60 s.
Open the app once on your phone the morning of the session.
