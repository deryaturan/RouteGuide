# RouteGuide

AI-powered multi-stop itinerary and transit planner (System Analysis and Design project, code RG).

## Run locally
```
pip install -r requirements.txt
mkdir .streamlit && cp secrets.toml.example .streamlit/secrets.toml   # then add your keys
streamlit run app.py
```

## Google Cloud setup
Enable these APIs on the project that owns your Maps key: **Places API (New)**, **Routes API**, **Geocoding API**.
Restrict the key to those three APIs. Get the Gemini key from Google AI Studio.

## Deploy (Streamlit Community Cloud)
1. Push `app.py`, `requirements.txt`, `README.md`, `.gitignore` to a new GitHub repo (never commit real keys).
2. share.streamlit.io > New app > pick the repo, branch and `app.py`.
3. App settings > Secrets: paste the contents of `secrets.toml.example` with your real keys.

## Notes
- Stateless: no login and no database; everything lives in `st.session_state`.
- Fares come only from the Routes API; missing fares are shown as "fare not available".
- Transit duration and fare use the Routes API `TRANSIT` mode; the stop order is chosen by Python, not Gemini.
