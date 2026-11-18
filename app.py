"""RouteGuide - AI-powered smart multi-stop itinerary & transit planner.

Streamlit app (deployable on Streamlit Community Cloud via GitHub).
External services: Google Places API (New), Google Routes API, Gemini API.
Stateless: nothing is written to disk or a database (NFR-04).
"""
from __future__ import annotations

import html
import itertools
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from datetime import time as dtime
from urllib.parse import quote, urlencode

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

try:  # browser geolocation component (permission prompt in the browser)
    from streamlit_geolocation import streamlit_geolocation
except Exception:  # pragma: no cover
    streamlit_geolocation = None

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MAX_STOPS = 5  # BR-01

st.set_page_config(
    page_title="RouteGuide",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name, default)


KEY_MAPS = secret("GOOGLE_MAPS_API_KEY")
KEY_GEM = secret("GEMINI_API_KEY")

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CUSTOM = "✍️ Custom search…"
CATEGORIES = {  # label -> (Places text query, default visit minutes)
    "🍽️ Restaurant": ("popular local restaurant", 60),
    "☕ Cafe": ("popular cafe", 40),
    "🥐 Breakfast": ("best breakfast place", 45),
    "🍰 Dessert & bakery": ("popular dessert shop bakery", 30),
    "🏛️ Museum": ("museum", 90),
    "🏰 Historic site": ("historical landmark", 60),
    "⭐ Top attraction": ("top tourist attraction", 60),
    "🌳 Park & nature": ("park garden", 45),
    "🌇 Viewpoint": ("scenic viewpoint", 30),
    "🛍️ Shopping": ("shopping mall or bazaar", 60),
    "🍸 Bar & nightlife": ("popular bar", 90),
    "🏖️ Beach": ("beach", 90),
    CUSTOM: ("", 45),
}
PRICE = {
    "PRICE_LEVEL_FREE": "Free",
    "PRICE_LEVEL_INEXPENSIVE": "$",
    "PRICE_LEVEL_MODERATE": "$$",
    "PRICE_LEVEL_EXPENSIVE": "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}
MODES = {"Public transit": "transit", "Driving": "driving", "Walking": "walking", "Cycling": "bicycling"}
INTERESTS = [
    "Local food", "History", "Art", "Nature", "Shopping", "Photography", "Nightlife",
    "Family-friendly", "Hidden gems", "Vegetarian-friendly", "Budget-friendly", "Accessible",
]
DISCLAIMER = (
    "Public transit fares are estimates returned by Google Maps and may differ from actual fares "
    "depending on real-time municipal updates or ticket types."
)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container{max-width:1100px;padding-top:1.6rem;padding-bottom:3rem}
        .rg-hero h1{margin:0;font-size:2.3rem;letter-spacing:-.5px}
        .rg-hero p{margin:.1rem 0 1rem;opacity:.75}
        .rg-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
                    gap:.75rem;margin:.5rem 0 1rem}
        .rg-metric{border:1px solid rgba(128,128,128,.3);border-radius:14px;
                   padding:.8rem 1rem;background:rgba(128,128,128,.08)}
        .rg-metric .l{font-size:.8rem;opacity:.75}
        .rg-metric .v{font-size:1.35rem;font-weight:700;line-height:1.3}
        .rg-metric .s{font-size:.78rem;opacity:.7}
        .rg-pick{border:1px solid rgba(26,115,232,.45);border-left:5px solid #1a73e8;
                 border-radius:12px;padding:.7rem .9rem;margin:.4rem 0;
                 background:rgba(26,115,232,.07)}
        .rg-pick .meta{font-size:.85rem;opacity:.8}
        .rg-pick .why{margin-top:.35rem;font-size:.9rem}
        .rg-next{border-radius:14px;padding:1rem;background:rgba(52,168,83,.12);
                 border:1px solid rgba(52,168,83,.5);margin:.5rem 0 1rem}
        .rg-tl{border-left:3px solid rgba(26,115,232,.5);padding:.1rem 0 .1rem .8rem;margin:.1rem 0 .7rem}
        .rg-tl .t{font-weight:700}
        .rg-tl .tip{font-size:.9rem;opacity:.9;margin-top:.2rem}
        @media (max-width:640px){
            .block-container{padding:.8rem .7rem 3rem}
            .rg-hero h1{font-size:1.8rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def fmt_dur(sec) -> str:
    if sec is None:
        return "Unavailable"
    minutes = round(sec / 60)
    h, m = divmod(minutes, 60)
    return f"{h} h {m} min" if h else f"{m} min"


def fmt_km(meters) -> str:
    return f"{meters / 1000:.1f} km" if meters is not None else "-"


def _err(resp: requests.Response) -> str:
    try:
        return resp.json()["error"]["message"]
    except Exception:
        return f"HTTP {resp.status_code}"


def bayes_score(p: dict) -> float:
    """Popularity score: rating pulled toward 4.2 when there are few reviews."""
    r, n = p.get("rating") or 0.0, p.get("reviews") or 0
    return (r * n + 4.2 * 50) / (n + 50) if n else 0.0


def decode_polyline(encoded: str) -> list[list[float]]:
    coords, index, lat, lng = [], 0, 0, 0
    while index < len(encoded):
        for is_lng in (False, True):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_lng:
                lng += delta
            else:
                lat += delta
        coords.append([lng / 1e5, lat / 1e5])
    return coords


def maps_url(dest: dict, origin: dict | None = None, waypoints: list[dict] | None = None,
             mode: str = "transit") -> str:
    """Google Maps URL. If origin is None Google Maps uses the user's live position."""
    params = {
        "api": 1,
        "destination": f"{dest['lat']},{dest['lng']}",
        "travelmode": mode,
    }
    if dest.get("place_id"):
        params["destination_place_id"] = dest["place_id"]
    if origin:
        params["origin"] = f"{origin['lat']},{origin['lng']}"
        if origin.get("place_id"):
            params["origin_place_id"] = origin["place_id"]
    if waypoints:
        params["waypoints"] = "|".join(f"{w['lat']},{w['lng']}" for w in waypoints)
        if all(w.get("place_id") for w in waypoints):
            params["waypoint_place_ids"] = "|".join(w["place_id"] for w in waypoints)
    return "https://www.google.com/maps/dir/?" + urlencode(params, safe=",|", quote_via=quote)


# --------------------------------------------------------------------------- #
# External APIs (plain functions, no Streamlit calls -> safe in threads)
# --------------------------------------------------------------------------- #
def places_search(query: str, lat=None, lng=None, radius=6000, n=10):
    body: dict = {"textQuery": query, "maxResultCount": n, "languageCode": "en"}
    if lat is not None:
        body["locationBias"] = {
            "circle": {"center": {"latitude": lat, "longitude": lng}, "radius": float(radius)}
        }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": KEY_MAPS,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,places.location,"
            "places.rating,places.userRatingCount,places.priceLevel,"
            "places.primaryTypeDisplayName,places.editorialSummary,places.googleMapsUri"
        ),
    }
    try:
        r = requests.post(PLACES_URL, headers=headers, json=body, timeout=15)
    except requests.RequestException as e:
        return [], f"Network error: {e}"
    if r.status_code != 200:
        return [], f"Google Places: {_err(r)}"
    out = []
    for p in r.json().get("places", []):
        loc = p.get("location") or {}
        if "latitude" not in loc:
            continue
        out.append({
            "id": p.get("id", ""),
            "place_id": p.get("id", ""),
            "name": (p.get("displayName") or {}).get("text", "Unnamed place"),
            "label": (p.get("displayName") or {}).get("text", "Unnamed place"),
            "address": p.get("formattedAddress", ""),
            "lat": loc["latitude"],
            "lng": loc["longitude"],
            "rating": p.get("rating"),
            "reviews": p.get("userRatingCount", 0),
            "price": PRICE.get(p.get("priceLevel")),
            "type": (p.get("primaryTypeDisplayName") or {}).get("text", ""),
            "summary": (p.get("editorialSummary") or {}).get("text", ""),
            "url": p.get("googleMapsUri", ""),
        })
    return out, None


def gemini(prompt: str, system: str, as_json: bool = True, temperature: float = 0.4):
    if not KEY_GEM:
        return None, "Gemini API key is not configured."
    model = secret("GEMINI_MODEL", "gemini-2.5-flash")
    cfg: dict = {"temperature": temperature}
    if as_json:
        cfg["responseMimeType"] = "application/json"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": cfg,
    }
    try:
        r = requests.post(
            GEMINI_URL.format(model=model),
            headers={"x-goog-api-key": KEY_GEM, "Content-Type": "application/json"},
            json=body,
            timeout=45,
        )
    except requests.RequestException as e:
        return None, f"Gemini network error: {e}"
    if r.status_code != 200:
        return None, f"Gemini: {_err(r)}"
    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except (KeyError, IndexError):
        return None, "Gemini returned an empty answer."
    if not as_json:
        return text.strip(), None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text), None
    except ValueError:
        return None, "Gemini returned an invalid answer."


def _secs(d: str) -> int:
    return int(float(str(d).rstrip("s")))


def fetch_leg(key: str, o: dict, d: dict, mode: str) -> dict:
    """One pairwise leg (A->B) for TRANSIT or DRIVE via Routes API."""
    body = {
        "origin": {"location": {"latLng": {"latitude": o["lat"], "longitude": o["lng"]}}},
        "destination": {"location": {"latLng": {"latitude": d["lat"], "longitude": d["lng"]}}},
        "travelMode": mode,
        "languageCode": "en-US",
        "units": "METRIC",
    }
    mask = "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"
    if mode == "TRANSIT":
        mask += ",routes.travelAdvisory.transitFare,routes.legs.steps.transitDetails"
    try:
        r = requests.post(
            ROUTES_URL,
            headers={"Content-Type": "application/json", "X-Goog-Api-Key": key, "X-Goog-FieldMask": mask},
            json=body,
            timeout=20,
        )
    except requests.RequestException:
        return {"ok": False, "error": "Network error - please retry", "transient": True}
    if r.status_code != 200:
        return {"ok": False, "error": f"Google Routes: {_err(r)}", "transient": True}
    routes = r.json().get("routes")
    if not routes:
        return {"ok": False, "error": "No route found", "transient": False}
    rt = routes[0]
    fare = None
    tf = (rt.get("travelAdvisory") or {}).get("transitFare")
    if tf:  # never estimated by us: only what Google returns (R-02)
        amount = float(tf.get("units", 0) or 0) + (tf.get("nanos", 0) or 0) / 1e9
        fare = {"amount": amount, "currency": tf.get("currencyCode", "")}
    lines: list[str] = []
    for leg in rt.get("legs", []):
        for step in leg.get("steps", []):
            td = step.get("transitDetails")
            if td:
                tl = td.get("transitLine") or {}
                nm = tl.get("nameShort") or tl.get("name")
                if nm and nm not in lines:
                    lines.append(nm)
    return {
        "ok": True,
        "seconds": _secs(rt["duration"]),
        "meters": rt.get("distanceMeters"),
        "fare": fare,
        "lines": lines,
        "polyline": (rt.get("polyline") or {}).get("encodedPolyline"),
        "transient": False,
    }


def fetch_matrix(key: str, pts: list[dict]):
    """Transit duration matrix between all points (one call)."""
    wp = [{"waypoint": {"location": {"latLng": {"latitude": p["lat"], "longitude": p["lng"]}}}} for p in pts]
    try:
        r = requests.post(
            MATRIX_URL,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": key,
                "X-Goog-FieldMask": "originIndex,destinationIndex,duration,condition",
            },
            json={"origins": wp, "destinations": wp, "travelMode": "TRANSIT"},
            timeout=25,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    rows = r.json()
    if not isinstance(rows, list):
        return None
    n = len(pts)
    m = [[None] * n for _ in range(n)]
    for e in rows:
        if e.get("condition") == "ROUTE_EXISTS" and "duration" in e:
            m[e["originIndex"]][e["destinationIndex"]] = _secs(e["duration"])
    return m


def best_order(m: list[list]):
    """Own sequencing logic (not Gemini): start fixed, brute force <=5 stops."""
    n, best = len(m), None
    for perm in itertools.permutations(range(1, n)):
        total, prev, ok = 0, 0, True
        for j in perm:
            t = m[prev][j]
            if t is None:
                ok = False
                break
            total += t
            prev = j
        if ok and (best is None or total < best[0]):
            best = (total, perm)
    return best


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
ss = st.session_state


def new_stop() -> dict:
    sid = uuid.uuid4().hex[:6]
    cat = next(iter(CATEGORIES))
    ss[f"cat_{sid}"] = cat
    ss[f"visit_{sid}"] = CATEGORIES[cat][1]
    return {"id": sid, "category": cat, "custom": "", "pick": None, "alts": []}


def init_state() -> None:
    ss.setdefault("start", None)
    ss.setdefault("start_cands", [])
    ss.setdefault("leg_cache", {})
    ss.setdefault("matrix_cache", {})
    ss.setdefault("rec_cache", {})
    ss.setdefault("result", None)
    ss.setdefault("guide", None)
    ss.setdefault("qa", [])
    ss.setdefault("flash", None)
    if "stops" not in ss:
        ss.stops = [new_stop()]


def find_stop(sid: str):
    return next((s for s in ss.stops if s["id"] == sid), None)


def add_stop() -> None:
    if len(ss.stops) >= MAX_STOPS:  # BR-01: block the 6th stop
        ss.flash = f"You can add at most {MAX_STOPS} stops. Remove one to add another."
        return
    ss.stops.append(new_stop())


def remove_stop(i: int) -> None:
    if len(ss.stops) > 1:
        ss.stops.pop(i)


def move_stop(i: int, d: int) -> None:
    j = i + d
    if 0 <= j < len(ss.stops):
        ss.stops[i], ss.stops[j] = ss.stops[j], ss.stops[i]


def on_category_change(sid: str) -> None:
    s = find_stop(sid)
    if s:
        s["category"] = ss[f"cat_{sid}"]
        s["pick"], s["alts"] = None, []
        ss[f"visit_{sid}"] = CATEGORIES[s["category"]][1]


def on_alt_change(sid: str) -> None:
    s = find_stop(sid)
    if s:
        chosen = ss.get(f"alt_{sid}")
        s["pick"] = next((p for p in s["alts"] if p["id"] == chosen), s["pick"])


def prefs() -> dict:
    return {
        "interests": ss.get("pref_interests", []),
        "budget": ss.get("pref_budget", "Any"),
        "pace": ss.get("pref_pace", "Balanced"),
        "notes": (ss.get("pref_notes", "") or "").strip(),
    }


def anchor_for(idx: int):
    for j in range(idx - 1, -1, -1):
        if ss.stops[j]["pick"]:
            return ss.stops[j]["pick"]
    return ss.start


# --------------------------------------------------------------------------- #
# Recommendations (Places = candidates, Gemini = personalised ranking)
# --------------------------------------------------------------------------- #
def get_recommendations(idx: int):
    s = ss.stops[idx]
    anchor = anchor_for(idx)
    if not anchor:
        return "Set your starting point first."
    query = s["custom"].strip() if s["category"] == CUSTOM else CATEGORIES[s["category"]][0]
    if not query:
        return "Type what you are looking for."
    pf = prefs()
    ck = json.dumps([query, round(anchor["lat"], 3), round(anchor["lng"], 3), pf], sort_keys=True)
    ordered = ss.rec_cache.get(ck)
    if ordered is None:
        places, err = places_search(query, anchor["lat"], anchor["lng"], radius=6000, n=15)
        if err:
            return err
        if not places:
            return "No results found. Try another category or search a specific place."
        places.sort(key=bayes_score, reverse=True)
        top = places[:8]
        reasons: dict[str, str] = {}
        rank_ids: list[str] = []
        payload = [
            {"id": p["id"], "name": p["name"], "rating": p["rating"], "reviews": p["reviews"],
             "price": p["price"], "type": p["type"], "about": p["summary"], "address": p["address"]}
            for p in top
        ]
        data, gerr = gemini(
            "Rank these candidate places for a traveller looking for: "
            f"{s['category'] if s['category'] != CUSTOM else query}.\n"
            f"They are near: {anchor['label']}.\n"
            f"Traveller preferences: {json.dumps(pf)}\n"
            "Prefer places that are genuinely popular (high rating AND many reviews) and fit the preferences.\n"
            'Return JSON: {"ranking":[{"id":"<candidate id>","reason":"<max 22 words>"}]} '
            "with the best 5 candidates first.\nCandidates: " + json.dumps(payload),
            "You are RouteGuide, a concise local travel guide. Use only the candidate list; "
            "never invent places, prices or opening hours.",
        )
        if data and isinstance(data.get("ranking"), list):
            valid = {p["id"] for p in top}
            for item in data["ranking"]:
                if isinstance(item, dict) and item.get("id") in valid and item["id"] not in reasons:
                    reasons[item["id"]] = str(item.get("reason", ""))[:200]
                    rank_ids.append(item["id"])
        elif gerr:
            ss.flash = f"AI ranking unavailable ({gerr}). Showing the most popular places instead."
        by_id = {p["id"]: p for p in top}
        ordered = [by_id[i] for i in rank_ids] + [p for p in top if p["id"] not in rank_ids]
        for p in ordered:
            p["reason"] = reasons.get(p["id"]) or (
                f"Highly rated ({p['rating']}) with {p['reviews']:,} reviews." if p.get("rating") else "Popular nearby."
            )
        ss.rec_cache[ck] = ordered
    taken = {o["pick"]["id"] for k, o in enumerate(ss.stops) if k != idx and o["pick"]}
    options = [p for p in ordered if p["id"] not in taken]
    if not options:
        return "All results are already used in other stops. Try another category."
    s["alts"], s["pick"] = options, options[0]
    ss[f"alt_{s['id']}"] = options[0]["id"]
    return None


# --------------------------------------------------------------------------- #
# Legs, totals, timeline
# --------------------------------------------------------------------------- #
def leg_key(a: dict, b: dict, mode: str) -> str:
    return f"{a['lat']:.5f},{a['lng']:.5f}>{b['lat']:.5f},{b['lng']:.5f}|{mode}"


def ensure_legs(points: list[dict]) -> None:
    """Session-state cache at origin-destination-mode level (NFR-01)."""
    cache = ss.leg_cache
    todo = []
    for a, b in zip(points, points[1:]):
        for m in ("TRANSIT", "DRIVE"):
            k = leg_key(a, b, m)
            if k not in cache or cache[k].get("transient"):
                todo.append((k, a, b, m))
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda t: fetch_leg(KEY_MAPS, t[1], t[2], t[3]), todo))
        for t, res in zip(todo, results):
            cache[t[0]] = res


def get_legs(points: list[dict]) -> list[dict]:
    miss = {"ok": False, "error": "Not calculated", "transient": True}
    return [
        {
            "from": a, "to": b,
            "TRANSIT": ss.leg_cache.get(leg_key(a, b, "TRANSIT"), miss),
            "DRIVE": ss.leg_cache.get(leg_key(a, b, "DRIVE"), miss),
        }
        for a, b in zip(points, points[1:])
    ]


def mode_total(legs: list[dict], mode: str):
    """Partial totals are never shown (report 6.4)."""
    if legs and all(l[mode]["ok"] for l in legs):
        return sum(l[mode]["seconds"] for l in legs), sum(l[mode].get("meters") or 0 for l in legs)
    return None, None


def fare_summary(legs: list[dict]):
    got, missing = [], []
    for i, l in enumerate(legs, 1):
        t = l["TRANSIT"]
        (got if t["ok"] and t.get("fare") else missing).append((i, t.get("fare")))
    if not got:
        return "Fare not available", "Google Maps returned no fare for this route"
    by: dict[str, float] = {}
    for _, f in got:
        by[f["currency"]] = by.get(f["currency"], 0.0) + f["amount"]
    text = " + ".join(f"{amt:,.2f} {cur}" for cur, amt in by.items())
    if missing:
        return text, "Incomplete: fare not available for leg(s) " + ", ".join(str(i) for i, _ in missing)
    return text, "Estimate returned by Google Maps"


def build_timeline(points, legs, mode, start_t, visits):
    if any(not l[mode]["ok"] for l in legs):
        return None
    t = datetime.combine(date.today(), start_t)
    rows = []
    for i, l in enumerate(legs):
        arrive = t + timedelta(seconds=l[mode]["seconds"])
        leave = arrive + timedelta(minutes=visits[i])
        rows.append({"i": i + 1, "travel": l[mode]["seconds"], "arrive": arrive, "leave": leave,
                     "place": points[i + 1], "visit": visits[i]})
        t = leave
    return rows


def route_sig(points: list[dict]) -> tuple:
    return tuple((round(p["lat"], 5), round(p["lng"], 5)) for p in points)


# --------------------------------------------------------------------------- #
# UI pieces
# --------------------------------------------------------------------------- #
def place_card(p: dict) -> None:
    stars = f"⭐ {p['rating']:.1f} ({p['reviews']:,} reviews)" if p.get("rating") else "No rating yet"
    meta = " · ".join(x for x in [p.get("type"), p.get("price"), stars] if x)
    why = f"<div class='why'>✨ {esc(p['reason'])}</div>" if p.get("reason") else ""
    link = f" · <a href='{esc(p['url'])}' target='_blank'>View on Google Maps</a>" if p.get("url") else ""
    st.markdown(
        f"<div class='rg-pick'><b>{esc(p['name'])}</b><br><span class='meta'>{esc(meta)}</span><br>"
        f"<span class='meta'>{esc(p['address'])}{link}</span>{why}</div>",
        unsafe_allow_html=True,
    )


def render_personalization() -> None:
    with st.expander("🎛️ Personalize your day", expanded=False):
        c1, c2 = st.columns(2)
        c1.multiselect("Interests", INTERESTS, key="pref_interests")
        c2.selectbox("Budget", ["Any", "Budget ($)", "Mid-range ($$)", "Premium ($$$)"], key="pref_budget")
        c1.selectbox("Pace", ["Relaxed", "Balanced", "Packed"], index=1, key="pref_pace")
        c2.text_area("Anything else? (diet, kids, mobility, mood…)", key="pref_notes", height=68,
                     placeholder="e.g. vegetarian, travelling with a toddler, love sea views")


def render_start() -> None:
    st.subheader("1 · Where do you start?")
    t_gps, t_manual = st.tabs(["📍 Use my location", "⌨️ Type an address or city"])
    with t_gps:
        st.caption("Tap the button and allow location access in your browser.")
        loc = streamlit_geolocation() if streamlit_geolocation else None
        lat, lng = (loc or {}).get("latitude"), (loc or {}).get("longitude")
        if lat is not None and lng is not None:
            if ss.get("gps_applied") != (lat, lng):
                ss.start = {"label": "My current location", "lat": lat, "lng": lng, "place_id": None}
                ss.gps_applied = (lat, lng)
                ss.result = None
            st.success("Location detected.")
        else:
            st.info("Waiting for location permission. If it is blocked or unavailable, "
                    "use the next tab - you can always type your start point.")
    with t_manual:
        with st.form("start_form", clear_on_submit=False):
            q = st.text_input("Address, place or city", placeholder="e.g. Konak Square, İzmir")
            go = st.form_submit_button("Set start point")
        if go:
            if not q.strip():
                st.warning("Please type an address or city.")
            else:
                with st.spinner("Searching…"):
                    res, err = places_search(q.strip(), n=5)
                if err:
                    st.error(err)
                elif not res:
                    st.warning("No match found. Try a more specific address.")
                else:
                    ss.start_cands = res
                    ss.start = res[0]
                    ss.result = None
                    ss["start_pick"] = res[0]["id"]
        if len(ss.start_cands) > 1:
            def _pick_start():
                m = next((c for c in ss.start_cands if c["id"] == ss.start_pick), None)
                if m:
                    ss.start, ss.result = m, None
            st.selectbox(
                "Not the right place?", [c["id"] for c in ss.start_cands], key="start_pick",
                format_func=lambda i: next(f"{c['name']} - {c['address']}" for c in ss.start_cands if c["id"] == i),
                on_change=_pick_start,
            )
    if ss.start:
        st.success(f"📍 Start: **{ss.start['label']}**")


def render_stop(i: int, s: dict) -> None:
    sid = s["id"]
    with st.container(border=True):
        st.markdown(f"**Stop {i + 1}**")
        st.selectbox("What are you in the mood for?", list(CATEGORIES), key=f"cat_{sid}",
                     on_change=on_category_change, args=(sid,))
        if s["category"] == CUSTOM:
            s["custom"] = st.text_input("Search for…", value=s["custom"], key=f"custom_{sid}",
                                        placeholder="e.g. vegan brunch, vinyl shop, rooftop")
        err = None
        if st.button("✨ Get AI pick", key=f"rec_{sid}", width="stretch"):
            with st.spinner("Asking the guide…"):
                err = get_recommendations(i)
            ss.result = None if not err else ss.result
        if err:
            st.warning(err)
        with st.expander("🔎 Or search a specific place"):
            q = st.text_input("Place name", key=f"find_{sid}")
            if st.button("Search", key=f"findbtn_{sid}") and q.strip():
                anc = anchor_for(i)
                res, e2 = places_search(q.strip(), anc["lat"] if anc else None, anc["lng"] if anc else None, n=5)
                if e2:
                    st.error(e2)
                elif not res:
                    st.warning("No match found.")
                else:
                    for p in res:
                        p["reason"] = ""
                    s["alts"], s["pick"] = res, res[0]
                    ss[f"alt_{sid}"] = res[0]["id"]
                    ss.result = None
        if s["pick"]:
            place_card(s["pick"])
            if len(s["alts"]) > 1:
                st.selectbox(
                    "Other popular options", [p["id"] for p in s["alts"]], key=f"alt_{sid}",
                    format_func=lambda pid, alts=s["alts"]: next(
                        f"{p['name']}" + (f"  ⭐{p['rating']:.1f}" if p.get("rating") else "")
                        for p in alts if p["id"] == pid),
                    on_change=on_alt_change, args=(sid,),
                )
        st.number_input("Time to spend here (min)", min_value=10, max_value=300, step=5, key=f"visit_{sid}")
        with st.popover("⋯ Reorder / remove"):
            st.button("▲ Move up", key=f"up_{sid}", on_click=move_stop, args=(i, -1), disabled=i == 0)
            st.button("▼ Move down", key=f"dn_{sid}", on_click=move_stop, args=(i, 1),
                      disabled=i == len(ss.stops) - 1)
            st.button("✕ Remove stop", key=f"rm_{sid}", on_click=remove_stop, args=(i,),
                      disabled=len(ss.stops) == 1)


def render_stops() -> None:
    st.subheader("2 · Pick up to 5 stops")
    if ss.flash:
        st.warning(ss.flash)
        ss.flash = None
    for i, s in enumerate(list(ss.stops)):
        render_stop(i, s)
    st.button(f"➕ Add stop ({len(ss.stops)}/{MAX_STOPS})", on_click=add_stop, key="add_stop")


# --------------------------------------------------------------------------- #
# Build + results
# --------------------------------------------------------------------------- #
def validate():
    if not ss.start:
        return "Set a starting point first (step 1)."
    for i, s in enumerate(ss.stops, 1):
        if not s["pick"]:
            return f"Stop {i} has no place yet - tap “Get AI pick” or search a place."
    return None


def build_route(optimize: bool) -> None:
    t0 = time.perf_counter()
    stops = list(ss.stops)
    note = None
    if optimize and len(stops) > 1:
        pts = [ss.start] + [s["pick"] for s in stops]
        mk = json.dumps(route_sig(pts))
        if mk not in ss.matrix_cache:
            ss.matrix_cache[mk] = fetch_matrix(KEY_MAPS, pts)
        m = ss.matrix_cache[mk]
        best = best_order(m) if m else None
        if best:
            new = [stops[j - 1] for j in best[1]]
            if [s["id"] for s in new] != [s["id"] for s in stops]:
                note = ("Stops were reordered to minimize total public transit time: "
                        + " → ".join(s["pick"]["name"] for s in new))
                ss.stops = new
                stops = new
        else:
            note = "Could not optimize the order (no transit data for some pairs) - your order was kept."
    points = [ss.start] + [s["pick"] for s in stops]
    ensure_legs(points)
    ss.result = {
        "sig": route_sig(points), "stop_ids": [s["id"] for s in stops], "note": note,
        "elapsed": time.perf_counter() - t0,
    }
    ss.guide = None
    for s in stops:
        ss.pop(f"done_{s['id']}", None)


def render_map(points, legs) -> None:
    paths = []
    for l in legs:
        enc = next((l[m].get("polyline") for m in ("TRANSIT", "DRIVE") if l[m].get("ok") and l[m].get("polyline")), None)
        if enc:
            paths.append({"path": decode_polyline(enc)})
        else:
            paths.append({"path": [[l["from"]["lng"], l["from"]["lat"]], [l["to"]["lng"], l["to"]["lat"]]]})
    df = pd.DataFrame([
        {"lat": p["lat"], "lon": p["lng"], "label": f"{i if i else 'Start'} · {p['label']}"}
        for i, p in enumerate(points)
    ])
    layers = [
        pdk.Layer("PathLayer", data=paths, get_path="path", get_color=[26, 115, 232], width_min_pixels=4),
        pdk.Layer("ScatterplotLayer", data=df, get_position="[lon, lat]", get_fill_color=[234, 67, 53],
                  get_radius=60, radius_min_pixels=8, pickable=True),
    ]
    try:
        view = pdk.data_utils.compute_view(df[["lon", "lat"]], 0.9)
    except Exception:
        view = pdk.ViewState(latitude=float(df.lat.mean()), longitude=float(df.lon.mean()), zoom=12)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{label}"}, map_style=None))


def render_route_tab(res, points, legs) -> None:
    t_sec, t_m = mode_total(legs, "TRANSIT")
    d_sec, d_m = mode_total(legs, "DRIVE")
    fare_txt, fare_sub = fare_summary(legs)
    st.markdown(
        "<div class='rg-metrics'>"
        f"<div class='rg-metric'><div class='l'>🚌 Public transit</div><div class='v'>{esc(fmt_dur(t_sec))}</div>"
        f"<div class='s'>{esc(fmt_km(t_m)) if t_sec is not None else 'A leg has no transit route'}</div></div>"
        f"<div class='rg-metric'><div class='l'>🚗 Driving</div><div class='v'>{esc(fmt_dur(d_sec))}</div>"
        f"<div class='s'>{esc(fmt_km(d_m)) if d_sec is not None else 'Driving route unavailable'}</div></div>"
        f"<div class='rg-metric'><div class='l'>💳 Est. transit fare</div><div class='v'>{esc(fare_txt)}</div>"
        f"<div class='s'>{esc(fare_sub)}</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"ℹ️ {DISCLAIMER}")
    if res.get("note"):
        st.info(res["note"])

    rows = []
    for i, l in enumerate(legs, 1):
        t, d = l["TRANSIT"], l["DRIVE"]
        rows.append({
            "Leg": i,
            "From": l["from"]["label"],
            "To": l["to"]["label"],
            "Transit": fmt_dur(t["seconds"]) if t["ok"] else f"⚠️ {t.get('error', 'n/a')}",
            "Fare": (f"{t['fare']['amount']:,.2f} {t['fare']['currency']}" if t["ok"] and t.get("fare")
                     else "fare not available"),
            "Lines": ", ".join(t.get("lines", [])) if t["ok"] else "",
            "Driving": fmt_dur(d["seconds"]) if d["ok"] else f"⚠️ {d.get('error', 'n/a')}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if any(l[m].get("transient") for l in legs for m in ("TRANSIT", "DRIVE")):
        st.warning("Some requests failed temporarily. Tap “Build my route” again to retry.")
    st.caption(f"Calculated in {res['elapsed']:.1f} s · repeated legs are served from the session cache.")

    render_map(points, legs)

    st.markdown("#### Open in Google Maps")
    mode_label = st.radio("Travel mode", list(MODES), horizontal=True, key="link_mode")
    mode = MODES[mode_label]
    url = maps_url(points[-1], points[0], points[1:-1] or None, mode)
    st.link_button("🗺️ Open full route in Google Maps", url, type="primary", width="stretch")
    if mode == "transit" and len(points) > 2:
        st.caption("Google Maps does not support intermediate stops in public transit mode. "
                   "Open the route leg by leg instead:")
    with st.expander("Leg-by-leg links", expanded=(mode == "transit" and len(points) > 2)):
        for i, l in enumerate(legs, 1):
            st.link_button(f"Leg {i}: {l['from']['label']} → {l['to']['label']}",
                           maps_url(l["to"], l["from"], None, mode), width="stretch")


def render_plan_tab(res, points, legs) -> None:
    c1, c2 = st.columns(2)
    start_t = c1.time_input("Start time", dtime(9, 0), key="plan_start")
    plan_mode_label = c2.radio("Plan travel by", ["Public transit", "Driving"], horizontal=True, key="plan_mode")
    mode = "TRANSIT" if plan_mode_label == "Public transit" else "DRIVE"
    visits = [int(ss.get(f"visit_{sid}", 45)) for sid in res["stop_ids"]]
    tl = build_timeline(points, legs, mode, start_t, visits)
    if tl is None:
        st.warning(f"A leg has no {plan_mode_label.lower()} route, so the daily plan cannot be calculated "
                   "(partial totals are never shown). Replace that stop or switch the travel mode.")
        return

    if st.button("✨ Generate my AI day guide", key="guide_btn", type="primary"):
        with st.spinner("Writing your day guide…"):
            stops_ctx = []
            for r in tl:
                p = r["place"]
                stops_ctx.append({
                    "index": r["i"], "name": p["name"], "address": p["address"], "type": p.get("type"),
                    "rating": p.get("rating"), "price": p.get("price"),
                    "arrive": r["arrive"].strftime("%H:%M"), "leave": r["leave"].strftime("%H:%M"),
                    "minutes_here": r["visit"], "travel_minutes_to_get_here": round(r["travel"] / 60),
                })
            data, err = gemini(
                "Create a short day guide for this itinerary.\n"
                f"Start: {points[0]['label']} at {start_t.strftime('%H:%M')} (travel by {plan_mode_label}).\n"
                f"Traveller preferences: {json.dumps(prefs())}\nStops: {json.dumps(stops_ctx)}\n"
                'Return JSON: {"summary":"<2 sentences>","stops":[{"index":1,"tip":"<max 30 words: what to do/try here>"}],'
                '"tips":["<max 4 short practical tips for the whole day>"]}. '
                "Do not invent opening hours, prices or facts you are not given; suggest checking Google Maps for hours.",
                "You are RouteGuide, a friendly, concise personal day-planning assistant.", temperature=0.6,
            )
        if err:
            st.error(err)
        else:
            ss.guide = data

    guide = ss.guide or {}
    tips = {int(x.get("index", 0)): x.get("tip", "") for x in guide.get("stops", []) if isinstance(x, dict)}
    if guide.get("summary"):
        st.info(guide["summary"])
    st.markdown(
        f"<div class='rg-tl'><span class='t'>{start_t.strftime('%H:%M')}</span> · Depart from "
        f"{esc(points[0]['label'])}</div>", unsafe_allow_html=True)
    for r in tl:
        p = r["place"]
        tip = f"<div class='tip'>💡 {esc(tips[r['i']])}</div>" if tips.get(r["i"]) else ""
        st.markdown(
            f"<div class='rg-tl'><span class='t'>{r['arrive'].strftime('%H:%M')} – {r['leave'].strftime('%H:%M')}</span>"
            f" · <b>{r['i']}. {esc(p['name'])}</b><br>"
            f"<span style='opacity:.75;font-size:.85rem'>🚶 {fmt_dur(r['travel'])} travel · {r['visit']} min here"
            f"{' · ⭐ ' + format(p['rating'], '.1f') if p.get('rating') else ''}</span>{tip}</div>",
            unsafe_allow_html=True)
    st.success(f"🏁 Back to free time around **{tl[-1]['leave'].strftime('%H:%M')}** · "
               f"total day length {fmt_dur((tl[-1]['leave'] - datetime.combine(date.today(), start_t)).total_seconds())}")
    for t in guide.get("tips", []) or []:
        st.markdown(f"- {t}")
    st.caption("Waiting time and opening hours are not modelled - check each place on Google Maps.")


def render_trip_tab(res, points, legs) -> None:
    ids = res["stop_ids"]
    done = [bool(ss.get(f"done_{sid}", False)) for sid in ids]
    st.progress(sum(done) / len(done), text=f"{sum(done)} of {len(done)} stops visited")
    nxt = next((i for i, d in enumerate(done) if not d), None)
    mode = MODES[ss.get("link_mode", "Public transit")]
    if nxt is None:
        st.success("🎉 Trip complete - enjoy the rest of your day!")
    else:
        p, leg = points[nxt + 1], legs[nxt]
        eta = leg["TRANSIT"] if leg["TRANSIT"]["ok"] else leg["DRIVE"]
        eta_txt = f" · about {fmt_dur(eta['seconds'])} from the previous stop" if eta["ok"] else ""
        st.markdown(
            f"<div class='rg-next'><b>Next up: {nxt + 1}. {esc(p['name'])}</b><br>"
            f"<span style='opacity:.8'>{esc(p['address'])}{esc(eta_txt)}</span></div>", unsafe_allow_html=True)
        # no origin -> Google Maps navigates from the traveller's live position
        st.link_button("🧭 Navigate to the next stop", maps_url(p, None, None, mode), type="primary", width="stretch")
    for i, sid in enumerate(ids):
        st.checkbox(f"{i + 1}. {points[i + 1]['name']}", key=f"done_{sid}")

    st.markdown("#### Ask your guide")
    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("e.g. “Is there a rainy-day alternative?” or “Where can I eat near stop 3?”")
        sent = st.form_submit_button("Ask")
    if sent and q.strip():
        ctx = {
            "start": points[0]["label"],
            "stops": [{"n": i + 1, "name": p["name"], "address": p["address"], "visited": done[i]}
                      for i, p in enumerate(points[1:])],
            "next_stop": None if nxt is None else points[nxt + 1]["name"],
            "preferences": prefs(),
        }
        with st.spinner("Thinking…"):
            ans, err = gemini(
                f"Trip context: {json.dumps(ctx)}\nTraveller question: {q.strip()}\n"
                "Answer in at most 120 words. Do not invent opening hours or prices; say so if unsure.",
                "You are RouteGuide, a friendly on-the-go travel assistant.", as_json=False, temperature=0.6)
        ss.qa.insert(0, (q.strip(), ans if ans else f"⚠️ {err}"))
        ss.qa = ss.qa[:5]
    for q, a in ss.qa:
        st.markdown(f"**You:** {q}")
        st.markdown(a)


def render_build() -> None:
    st.subheader("3 · Build your route")
    optimize = st.toggle("Optimize stop order (minimizes public transit time)", value=True, key="optimize")
    if st.button("🚀 Build my route", type="primary", width="stretch"):
        problem = validate()
        if problem:
            st.warning(problem)
        else:
            with st.spinner("Calculating routes…"):
                build_route(optimize)
            st.rerun()

    res = ss.result
    if not res:
        return
    points = [ss.start] + [find_stop(sid)["pick"] for sid in res["stop_ids"] if find_stop(sid) and find_stop(sid)["pick"]]
    if len(points) != len(res["stop_ids"]) + 1 or route_sig(points) != res["sig"]:
        st.warning("Your itinerary changed - tap “Build my route” to update the results.")
        return
    legs = get_legs(points)
    st.divider()
    t_route, t_plan, t_trip = st.tabs(["🗺️ Route", "📅 Day plan", "🧭 Trip mode"])
    with t_route:
        render_route_tab(res, points, legs)
    with t_plan:
        render_plan_tab(res, points, legs)
    with t_trip:
        render_trip_tab(res, points, legs)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()
    st.markdown(
        "<div class='rg-hero'><h1>🧭 RouteGuide</h1>"
        "<p>Your AI-powered multi-stop itinerary &amp; transit planner.</p></div>",
        unsafe_allow_html=True,
    )
    if not KEY_MAPS:
        st.error("`GOOGLE_MAPS_API_KEY` is missing. Add it to Streamlit secrets (see README).")
        st.code('GOOGLE_MAPS_API_KEY = "your-key"\nGEMINI_API_KEY = "your-key"', language="toml")
        st.stop()
    if not KEY_GEM:
        st.warning("`GEMINI_API_KEY` is missing - recommendations fall back to popularity ranking "
                   "and AI day guides are disabled.")
    init_state()
    render_personalization()
    render_start()
    render_stops()
    render_build()
    st.divider()
    st.caption("RouteGuide · Data from Google Maps Platform · AI by Google Gemini · "
               "Nothing is stored: your session lives only in this browser tab.")
    if st.button("↺ Start over"):
        ss.clear()
        st.rerun()


main()
