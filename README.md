# RouteGuide

AI-powered smart multi-stop itinerary & transit planner (Streamlit).

## Features
- Browser geolocation (permission prompt) **or** manual address/city start point
- Up to 5 stops (6th blocked with an in-app message), reorder/remove
- Per-stop category -> Google Places candidates ranked by popularity and personalised by Gemini
- Leg-by-leg public transit **and** driving durations, total time, Google-returned transit fare
  (shown as "fare not available" when Google returns none; never estimated)
- Optional stop-order optimisation (own Python logic, not Gemini)
- Route map, Google Maps deep links (full route + leg-by-leg for transit)
- Personalisation, timed day plan, AI day guide, trip mode with live "navigate to next stop", Q&A
- Stateless: nothing stored, session cache only

## Google Cloud setup
Create a project with billing enabled, then enable:
1. **Places API (New)**
2. **Routes API**

Create an API key and restrict it to those two APIs. Get a Gemini key from Google AI Studio.
Check current free-tier quotas/pricing of both before heavy testing.

## Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your keys
streamlit run app.py
```

## Deploy (GitHub -> Streamlit Community Cloud)
1. Push `app.py`, `requirements.txt`, `.streamlit/config.toml`, `.gitignore`, `README.md` to a GitHub repo
   (do **not** push `secrets.toml`).
2. share.streamlit.io -> New app -> pick the repo, branch, main file `app.py`.
3. App settings -> Secrets -> paste `GOOGLE_MAPS_API_KEY` and `GEMINI_API_KEY`.

Note: browser geolocation requires HTTPS (Streamlit Cloud provides it).
