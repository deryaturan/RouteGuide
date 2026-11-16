"""
RouteGuide (RG) - AI-Powered Smart Multi-Stop Itinerary & Transit Planner
System Analysis and Design - Fall 2026-2027 - Derya Turan (2304109020)

Karşıladığı gereksinimler (rapor ile eşleşme):
  FR-01 / OBJ-01 / BR-02 : Tarayıcı konumu, adres arama (Nominatim), kayıtlı yerler (fallback)
  FR-02 / OBJ-02         : Gemini + Google Places (New) ile kategori bazlı öneri
  FR-03 / OBJ-03 / BR-01 : En fazla 5 durak ekle / sil / yeniden sırala
  FR-04                  : İkili (A->B, B->C) bacak bazlı çok duraklı transit hesabı
  FR-05 / OBJ-04-05      : Transit vs Araç süresi + tahmini ücret yan yana (BR-03 uyarısı)
  Rota optimizasyonu     : Distance Matrix + tüm permütasyonlar (<=5 durak => <=120 sıra)
  Harita                 : Directions polyline'ları ile arayüz içinde harita
  NFR-01                 : st.cache_data + harici API çağrı sayacı
  NFR-02                 : Mobil öncelikli, tek sütun düzen
  NFR-03                 : Hesaplama süresi ölçümü (hedef < 3 sn)
  NFR-04                 : Veritabanı yok, yalnızca st.session_state (durumsuz oturum)
  Hata/istisna akışları  : Bölüm 6.4

Gerekli anahtarlar (Streamlit Cloud > Settings > Secrets):
  GOOGLE_MAPS_API_KEY = "..."
  GEMINI_API_KEY      = "..."
  GEMINI_MODEL        = "gemini-2.5-flash"   # isteğe bağlı
"""

import itertools
import json
import math
import os
import time

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

try:  # FR-01: tarayıcı konumu (izin verilmezse uygulama çökmemeli - BR-02)
    from streamlit_geolocation import streamlit_geolocation

    HAS_GEO = True
except Exception:  # paket yoksa da manuel fallback çalışır
    HAS_GEO = False

# --------------------------------------------------------------------------
# Sabitler
# --------------------------------------------------------------------------
MAX_STOPS = 5  # BR-01
INF = 10**7
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "RouteGuide-SAD-Student-Project/1.0 (Izmir Democracy University)"

CATEGORIES = {
    "🍽️ Restoran": "restaurant",
    "☕ Kafe": "cafe",
    "🏛️ Müze": "museum",
    "🌳 Park / Doğa": "park",
    "🏺 Tarihi yer": "historical landmark",
    "🛍️ Alışveriş": "shopping mall",
    "🍰 Tatlı / Pastane": "dessert shop",
    "🎭 Sanat / Kültür": "art gallery or cultural center",
    "✍️ Diğer (kendim yazayım)": None,
}

PRESET_SPOTS = {  # R-03: hazır kayıtlı yerler
    "Şehir Merkezi (Konak Saat Kulesi)": {"lat": 38.41885, "lng": 27.12872},
    "Ege Üniversitesi Metro İstasyonu": {"lat": 38.45530, "lng": 27.22740},
}

PRICE_MAP = {
    "PRICE_LEVEL_FREE": "Ücretsiz",
    "PRICE_LEVEL_INEXPENSIVE": "₺",
    "PRICE_LEVEL_MODERATE": "₺₺",
    "PRICE_LEVEL_EXPENSIVE": "₺₺₺",
    "PRICE_LEVEL_VERY_EXPENSIVE": "₺₺₺₺",
}


class ApiError(Exception):
    """Harici API hatası (zaman aşımı, kota, yetki vb.)."""


# --------------------------------------------------------------------------
# Sayfa ayarı ve stil (NFR-02: 375px mobil uyumlu, yatay kaydırma yok)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="RouteGuide",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    .block-container {padding: 1.2rem 1rem 3rem 1rem; max-width: 760px;}
    button {white-space: normal !important;}
    html, body {overflow-x: hidden;}
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Oturum durumu (NFR-04: kalıcı depolama yok)
# --------------------------------------------------------------------------
def init_state():
    defaults = {
        "start": None,
        "stops": [],
        "saved": {"Ev": None, "Otel": None},
        "result": None,
        "discovered": None,
        "flash": [],
        "api_calls": 0,
        "timings": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def flash(kind, msg):
    st.session_state.flash.append((kind, msg))


def show_flash():
    for kind, msg in st.session_state.flash:
        getattr(st, kind)(msg)
    st.session_state.flash = []


def invalidate():
    st.session_state.result = None


# --------------------------------------------------------------------------
# Anahtarlar
# --------------------------------------------------------------------------
def secret(name, default=""):
    try:
        val = st.secrets.get(name)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(name, default)


# --------------------------------------------------------------------------
# Harici API yardımcıları
# --------------------------------------------------------------------------
def _request(method, url, **kwargs):
    # Önbellek ıskalarında çalışır => gerçek harici çağrı sayısı (NFR-01 kanıtı)
    st.session_state["api_calls"] = st.session_state.get("api_calls", 0) + 1
    try:
        return requests.request(method, url, timeout=20, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Ağ hatası veya zaman aşımı: {exc}")


def _maps_json(url, params):
    r = _request("GET", url, params=params)
    try:
        data = r.json()
    except ValueError:
        raise ApiError(f"Geçersiz yanıt (HTTP {r.status_code}).")
    status = data.get("status")
    if status in ("OK", "ZERO_RESULTS", "NOT_FOUND"):
        return data
    raise ApiError(
        f"Google Maps API hatası: {status} {data.get('error_message', '')}".strip()
    )


@st.cache_data(ttl=3600, show_spinner=False)
def geocode(query, country_codes):
    """Nominatim (OpenStreetMap) ile adres/yer arama."""
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 6,
        "accept-language": "tr",
    }
    if country_codes:
        params["countrycodes"] = country_codes
    r = _request("GET", NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT})
    if r.status_code != 200:
        raise ApiError(f"Nominatim hatası (HTTP {r.status_code}).")
    out = []
    for item in r.json():
        disp = item.get("display_name", "")
        out.append(
            {
                "name": item.get("name") or disp.split(",")[0],
                "address": disp,
                "lat": float(item["lat"]),
                "lng": float(item["lon"]),
                "place_id": f"osm-{item.get('place_id')}",
            }
        )
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def places_search(lat, lng, text_query, radius, maps_key):
    """Google Places API (New) - Text Search."""
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": maps_key,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,places.location,"
            "places.rating,places.userRatingCount,places.priceLevel,"
            "places.primaryTypeDisplayName,places.googleMapsUri"
        ),
    }
    body = {
        "textQuery": text_query,
        "languageCode": "tr",
        "maxResultCount": 12,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(min(max(radius, 100), 50000)),
            }
        },
    }
    r = _request("POST", url, headers=headers, json=body)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", "")
        except ValueError:
            msg = ""
        raise ApiError(f"Places API hatası (HTTP {r.status_code}): {msg}")
    places = []
    for p in r.json().get("places", []):
        loc = p.get("location", {})
        places.append(
            {
                "name": p.get("displayName", {}).get("text", "?"),
                "address": p.get("formattedAddress", ""),
                "lat": loc.get("latitude"),
                "lng": loc.get("longitude"),
                "place_id": p.get("id"),
                "rating": p.get("rating"),
                "n_ratings": p.get("userRatingCount"),
                "price": PRICE_MAP.get(p.get("priceLevel", ""), ""),
                "type": p.get("primaryTypeDisplayName", {}).get("text", ""),
                "uri": p.get("googleMapsUri", ""),
            }
        )
    return [p for p in places if p["lat"] is not None]


@st.cache_data(ttl=1800, show_spinner=False)
def gemini_json(prompt, gem_key, model):
    """Gemini'den yalnızca JSON döndürmesini ister ve ayrıştırır."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    r = _request("POST", url, params={"key": gem_key}, json=body)
    if r.status_code != 200:
        raise ApiError(f"Gemini API hatası (HTTP {r.status_code}).")
    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        return json.loads(text)
    except (KeyError, IndexError, ValueError):
        raise ApiError("Gemini yanıtı ayrıştırılamadı.")


@st.cache_data(ttl=900, show_spinner=False)
def distance_matrix(coords, mode, maps_key):
    """Tüm noktalar arası süre matrisi (saniye). Erişilemeyenler INF."""
    pts = "|".join(f"{a},{b}" for a, b in coords)
    params = {"origins": pts, "destinations": pts, "mode": mode, "key": maps_key, "language": "tr"}
    if mode == "transit":
        params["departure_time"] = int(time.time())
    data = _maps_json("https://maps.googleapis.com/maps/api/distancematrix/json", params)
    matrix = []
    for row in data.get("rows", []):
        r = []
        for el in row.get("elements", []):
            r.append(el["duration"]["value"] if el.get("status") == "OK" else INF)
        matrix.append(r)
    n = len(coords)
    if len(matrix) != n or any(len(r) != n for r in matrix):
        raise ApiError("Mesafe matrisi beklenen boyutta dönmedi.")
    return matrix


@st.cache_data(ttl=900, show_spinner=False)
def directions(olat, olng, dlat, dlng, mode, maps_key):
    """Tek bacak (A->B) rota. Rota yoksa None."""
    params = {
        "origin": f"{olat},{olng}",
        "destination": f"{dlat},{dlng}",
        "mode": mode,
        "language": "tr",
        "region": "tr",
        "key": maps_key,
    }
    if mode == "transit":
        params["departure_time"] = int(time.time())
    data = _maps_json("https://maps.googleapis.com/maps/api/directions/json", params)
    return parse_route(data)


def parse_route(data):
    routes = data.get("routes") or []
    if not routes:
        return None
    route = routes[0]
    leg = route["legs"][0]
    lines = []
    n_transit = 0
    for s in leg.get("steps", []):
        if s.get("travel_mode") == "TRANSIT":
            n_transit += 1
            line = s.get("transit_details", {}).get("line", {})
            veh = line.get("vehicle", {}).get("name", "")
            name = line.get("short_name") or line.get("name") or "?"
            lines.append(f"{veh} {name}".strip())
    fare = route.get("fare")
    return {
        "seconds": leg["duration"]["value"],
        "meters": leg["distance"]["value"],
        "polyline": route.get("overview_polyline", {}).get("points", ""),
        "lines": lines,
        "n_transit": n_transit,
        "api_fare": (float(fare["value"]), fare.get("currency", "TRY")) if fare else None,
    }


# --------------------------------------------------------------------------
# Saf (API'siz) yardımcı fonksiyonlar - testlenebilir
# --------------------------------------------------------------------------
def decode_polyline(encoded):
    coords, index, lat, lng = [], 0, 0, 0
    while index < len(encoded):
        for is_lat in (True, False):
            shift, result = 0, 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_lat:
                lat += delta
            else:
                lng += delta
        coords.append((lat / 1e5, lng / 1e5))
    return coords


def best_order(matrix, n_stops, return_home=False):
    """Başlangıç=0, duraklar=1..n. Toplam süreyi minimize eden sıra (<=120 permütasyon)."""
    best, best_cost = list(range(1, n_stops + 1)), None
    for perm in itertools.permutations(range(1, n_stops + 1)):
        cost = matrix[0][perm[0]]
        for a, b in zip(perm, perm[1:]):
            cost += matrix[a][b]
        if return_home:
            cost += matrix[perm[-1]][0]
        if best_cost is None or cost < best_cost:
            best, best_cost = list(perm), cost
    return best


def baseline_fare(n_boardings, base, transfer_ratio):
    """R-02: API ücret döndürmezse temel tarife modeli (tahmin)."""
    if n_boardings <= 0:
        return 0.0
    return round(base + (n_boardings - 1) * base * transfer_ratio, 2)


def can_add_stop(stops):
    """BR-01: en fazla 5 durak."""
    return len(stops) < MAX_STOPS


def fmt_dur(sec):
    if sec is None:
        return "—"
    m = int(round(sec / 60))
    h, m = divmod(m, 60)
    return f"{h} sa {m:02d} dk" if h else f"{m} dk"


def fmt_dist(meters):
    return "—" if meters is None else f"{meters / 1000:.1f} km"


def place_exists(place, stops):
    for s in stops:
        if s.get("place_id") and s.get("place_id") == place.get("place_id"):
            return True
        if abs(s["lat"] - place["lat"]) < 1e-5 and abs(s["lng"] - place["lng"]) < 1e-5:
            return True
    return False


# --------------------------------------------------------------------------
# Durum değiştiren callback'ler
# --------------------------------------------------------------------------
def set_start(place):
    st.session_state.start = {
        "name": place["name"],
        "address": place.get("address", ""),
        "lat": place["lat"],
        "lng": place["lng"],
        "place_id": place.get("place_id"),
    }
    invalidate()
    flash("success", f"Başlangıç noktası ayarlandı: {place['name']}")


def save_spot(label):
    s = st.session_state.start
    if not s:
        flash("warning", "Önce bir başlangıç noktası seçmelisin.")
        return
    st.session_state.saved[label] = dict(s)
    flash("success", f"'{label}' olarak kaydedildi (yalnızca bu oturumda).")


def add_stop(place):
    stops = st.session_state.stops
    if not can_add_stop(stops):  # BR-01
        flash("error", f"BR-01: En fazla {MAX_STOPS} durak ekleyebilirsin. Önce bir durağı sil.")
        return
    if place_exists(place, stops):
        flash("warning", f"'{place['name']}' zaten rotada.")
        return
    stops.append(
        {
            "name": place["name"],
            "address": place.get("address", ""),
            "lat": place["lat"],
            "lng": place["lng"],
            "place_id": place.get("place_id"),
        }
    )
    invalidate()
    flash("success", f"Durak eklendi: {place['name']} ({len(stops)}/{MAX_STOPS})")


def remove_stop(i):
    st.session_state.stops.pop(i)
    invalidate()


def move_stop(i, delta):
    stops = st.session_state.stops
    j = i + delta
    if 0 <= j < len(stops):
        stops[i], stops[j] = stops[j], stops[i]
        invalidate()


def clear_all():
    for k in ("start", "stops", "result", "discovered"):
        st.session_state[k] = None if k in ("start", "result", "discovered") else []
    st.session_state.saved = {"Ev": None, "Otel": None}


# --------------------------------------------------------------------------
# FR-02: Gemini + Places ile kategori bazlı öneri
# --------------------------------------------------------------------------
def run_discovery(center, category_label, query_en, prefs, radius, maps_key, gem_key, model):
    notes = []
    text_query, min_rating = query_en, 0.0

    # 1) Doğal dil tercihlerini arama sorgusuna çevir (Gemini)
    if gem_key and prefs.strip():
        try:
            parsed = gemini_json(
                "A traveler wants to find places in the category '"
                + category_label
                + "'. Their preferences (any language): '"
                + prefs.strip()
                + "'. Return ONLY JSON: {\"text_query\": \"a concise English Google Places "
                "text search query that reflects the category and preferences\", "
                "\"min_rating\": a number between 0 and 4.7}.",
                gem_key,
                model,
            )
            text_query = str(parsed.get("text_query") or query_en)
            min_rating = float(parsed.get("min_rating") or 0)
        except (ApiError, ValueError, TypeError) as exc:
            notes.append(f"Tercihler Gemini ile işlenemedi, varsayılan arama kullanıldı ({exc}).")
    elif prefs.strip() and not gem_key:
        notes.append("Gemini anahtarı yok: tercih metni yorumlanmadı, standart arama yapıldı.")

    # 2) Google Places araması
    places = places_search(center["lat"], center["lng"], text_query, radius, maps_key)
    if not places:
        return [], notes, text_query
    filtered = [p for p in places if (p["rating"] or 0) >= min_rating] or places

    # 3) Gemini ile bağlama duyarlı sıralama + gerekçe
    ranked = None
    if gem_key:
        try:
            listing = [
                {
                    "index": i,
                    "name": p["name"],
                    "type": p["type"],
                    "rating": p["rating"],
                    "ratings_count": p["n_ratings"],
                    "price": p["price"],
                    "address": p["address"],
                }
                for i, p in enumerate(filtered)
            ]
            out = gemini_json(
                "You are a local travel guide. Category: "
                + category_label
                + ". Traveler preferences: '"
                + (prefs.strip() or "none")
                + "'. Candidate venues (JSON): "
                + json.dumps(listing, ensure_ascii=False)
                + ". Pick the best up to 6 venues, ordered by relevance. Return ONLY JSON: "
                '{"picks": [{"index": <int>, "reason": "<one short sentence in Turkish>"}]}.',
                gem_key,
                model,
            )
            ranked = []
            for pick in out.get("picks", []):
                idx = int(pick["index"])
                if 0 <= idx < len(filtered):
                    item = dict(filtered[idx])
                    item["reason"] = str(pick.get("reason", ""))
                    ranked.append(item)
        except (ApiError, ValueError, TypeError, KeyError) as exc:
            notes.append(f"Gemini sıralaması kullanılamadı, puana göre sıralandı ({exc}).")
    if not ranked:
        ranked = sorted(
            filtered,
            key=lambda p: (p["rating"] or 0) * math.log10((p["n_ratings"] or 0) + 10),
            reverse=True,
        )[:6]
        ranked = [dict(p, reason="") for p in ranked]
    return ranked, notes, text_query


# --------------------------------------------------------------------------
# FR-04 / FR-05: Çok duraklı hesap (ikili bacaklar) + optimizasyon
# --------------------------------------------------------------------------
def compute_trip(start, stops, optimize, criterion, return_home, maps_key, fare_cfg):
    t0 = time.perf_counter()
    order = list(stops)
    optimized = False
    if optimize and len(stops) > 1:
        coords = tuple((p["lat"], p["lng"]) for p in [start] + stops)
        matrix = distance_matrix(coords, criterion, maps_key)
        perm = best_order(matrix, len(stops), return_home)
        order = [stops[i - 1] for i in perm]
        optimized = perm != list(range(1, len(stops) + 1))

    seq = [start] + order + ([start] if return_home else [])
    legs = []
    for a, b in zip(seq, seq[1:]):
        tr = directions(a["lat"], a["lng"], b["lat"], b["lng"], "transit", maps_key)
        dr = directions(a["lat"], a["lng"], b["lat"], b["lng"], "driving", maps_key)
        fare, fare_src, cur = None, "—", "TRY"
        if tr is not None:
            if tr["api_fare"]:
                fare, cur = tr["api_fare"]
                fare_src = "Google API"
            else:
                fare = baseline_fare(tr["n_transit"], fare_cfg["base"], fare_cfg["ratio"])
                fare_src = "Temel tarife (tahmin)"
        legs.append(
            {
                "from": a["name"],
                "to": b["name"],
                "transit": tr,
                "driving": dr,
                "fare": fare,
                "fare_src": fare_src,
                "currency": cur,
            }
        )

    t_vals = [l["transit"]["seconds"] for l in legs if l["transit"]]
    d_vals = [l["driving"]["seconds"] for l in legs if l["driving"]]
    fares = [l["fare"] for l in legs if l["fare"] is not None]
    result = {
        "start": start,
        "order": order,
        "legs": legs,
        "optimized": optimized,
        "t_transit": sum(t_vals) if t_vals else None,
        "t_driving": sum(d_vals) if d_vals else None,
        "missing_transit": sum(1 for l in legs if l["transit"] is None),
        "missing_driving": sum(1 for l in legs if l["driving"] is None),
        "fare_total": sum(fares) if fares else 0.0,
        "currency": legs[0]["currency"] if legs else "TRY",
        "elapsed": time.perf_counter() - t0,
        "return_home": return_home,
    }
    return result


# --------------------------------------------------------------------------
# Harita (Directions polyline'ları ile arayüz içinde)
# --------------------------------------------------------------------------
def build_map(res, mode_key):
    paths = []
    color = [30, 100, 220] if mode_key == "transit" else [230, 120, 20]
    for leg in res["legs"]:
        route = leg[mode_key]
        if route and route["polyline"]:
            pts = decode_polyline(route["polyline"])
            paths.append({"path": [[lng, lat] for lat, lng in pts], "color": color})
    marks = [{"label": "A: " + res["start"]["name"], "lat": res["start"]["lat"], "lng": res["start"]["lng"]}]
    for i, s in enumerate(res["order"], start=1):
        marks.append({"label": f"{i}: {s['name']}", "lat": s["lat"], "lng": s["lng"]})
    df = pd.DataFrame(marks)

    all_lat = [m["lat"] for m in marks] + [p[1] for d in paths for p in d["path"]]
    all_lng = [m["lng"] for m in marks] + [p[0] for d in paths for p in d["path"]]
    clat, clng = (min(all_lat) + max(all_lat)) / 2, (min(all_lng) + max(all_lng)) / 2
    span = max(
        max(all_lat) - min(all_lat),
        (max(all_lng) - min(all_lng)) * math.cos(math.radians(clat)),
        0.005,
    )
    zoom = max(9, min(15, math.log2(360 / span) - 1))

    layers = []
    if paths:
        layers.append(
            pdk.Layer("PathLayer", data=paths, get_path="path", get_color="color", width_min_pixels=4)
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[lng, lat]",
            get_fill_color=[200, 30, 30, 220],
            get_radius=60,
            radius_min_pixels=7,
            pickable=True,
        )
    )
    layers.append(
        pdk.Layer(
            "TextLayer",
            data=df,
            get_position="[lng, lat]",
            get_text="label",
            get_size=14,
            get_color=[20, 20, 20, 255],
            get_pixel_offset=[0, -20],
        )
    )
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=clat, longitude=clng, zoom=zoom),
        tooltip={"text": "{label}"},
    )


# --------------------------------------------------------------------------
# Doğrulama testleri (traceability: TC-01..TC-05 için çevrimdışı kontroller)
# --------------------------------------------------------------------------
def run_self_tests():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    pts = decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
    check("Polyline çözme", abs(pts[0][0] - 38.5) < 1e-4 and abs(pts[2][1] + 126.453) < 1e-3)
    check("BR-01: 5 durakta 6.'yı engelle", can_add_stop([0] * 4) and not can_add_stop([0] * 5))
    m = [[0, 10, 50, 20], [10, 0, 5, 40], [50, 5, 0, 8], [20, 40, 8, 0]]
    check("Optimizasyon (0->1->2->3 en kısa)", best_order(m, 3) == [1, 2, 3])
    m2 = [[0, 90, 5], [90, 0, 5], [5, 5, 0]]
    check("Optimizasyon (en yakın durak önce)", best_order(m2, 2)[0] == 2)
    check("Temel tarife: 0 biniş = 0", baseline_fare(0, 30, 0.5) == 0.0)
    check("Temel tarife: 2 biniş", baseline_fare(2, 30, 0.5) == 45.0)
    check("Süre biçimi", fmt_dur(3900) == "1 sa 05 dk" and fmt_dur(None) == "—")
    check(
        "Yinelenen durak engeli",
        place_exists({"lat": 1.0, "lng": 1.0, "place_id": "x"}, [{"lat": 1.0, "lng": 1.0, "place_id": "y"}]),
    )
    return results


# --------------------------------------------------------------------------
# Kenar çubuğu
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    maps_key = st.text_input(
        "Google Maps API anahtarı",
        value=secret("GOOGLE_MAPS_API_KEY"),
        type="password",
        help="Streamlit Cloud > Secrets içine GOOGLE_MAPS_API_KEY olarak da ekleyebilirsin.",
    )
    gem_key = st.text_input(
        "Gemini API anahtarı", value=secret("GEMINI_API_KEY"), type="password"
    )
    gem_model = st.text_input("Gemini modeli", value=secret("GEMINI_MODEL", "gemini-2.5-flash"))
    country = st.text_input("Adres arama ülke kodu", value="tr", help="Nominatim countrycodes (boş = tüm dünya)")
    st.divider()
    criterion = st.radio(
        "Optimizasyon kriteri",
        ["transit", "driving"],
        format_func=lambda x: "Toplu taşıma süresi" if x == "transit" else "Araç süresi",
    )
    st.subheader("Tarife varsayımı (R-02)")
    fare_base = st.number_input("Temel biniş ücreti (TL)", 0.0, 500.0, 30.0, 0.5,
                                help="API ücret döndürmezse kullanılan TAHMİNİ tarife. Kendi şehrinin güncel tarifesine göre güncelle.")
    fare_ratio = st.slider("Aktarma ücreti oranı", 0.0, 1.0, 0.5, 0.05)
    st.divider()
    st.metric("Harici API çağrısı (bu oturum)", st.session_state.api_calls)
    st.caption("Önbellek isabetlerinde sayaç artmaz (NFR-01).")
    if st.session_state.timings:
        st.caption("Hesaplama süreleri (sn): " + ", ".join(f"{t:.2f}" for t in st.session_state.timings[-6:]))
    with st.expander("🧪 Doğrulama testleri"):
        if st.button("Testleri çalıştır"):
            for name, ok in run_self_tests():
                st.write(("✅ " if ok else "❌ ") + name)
    st.button("🗑️ Oturumu sıfırla", on_click=clear_all)

# --------------------------------------------------------------------------
# Başlık
# --------------------------------------------------------------------------
st.title("🧭 RouteGuide")
st.caption("Yapay zekâ destekli çok duraklı gezi ve toplu taşıma planlayıcı")
show_flash()

if not maps_key:
    st.warning("Google Maps API anahtarı bulunamadı. Kenar çubuğundan (☰) gir veya Secrets'a ekle.")

c1, c2, c3 = st.columns(3)
c1.markdown("**📍 Başlangıç**  \n" + (st.session_state.start["name"] if st.session_state.start else "_seçilmedi_"))
c2.markdown(f"**🛑 Duraklar**  \n{len(st.session_state.stops)}/{MAX_STOPS}")
c3.markdown("**📊 Sonuç**  \n" + ("hazır" if st.session_state.result else "_yok_"))

tab_loc, tab_disc, tab_route, tab_res = st.tabs(["1 · Konum", "2 · Keşfet", "3 · Rota", "4 · Sonuç"])

# ---------------------------- 1) Konum (FR-01) ----------------------------
with tab_loc:
    st.subheader("Başlangıç noktası")
    sub_gps, sub_search, sub_saved = st.tabs(["📍 Konumum", "🔎 Adres ara", "⭐ Kayıtlı yerler"])

    with sub_gps:
        if HAS_GEO:
            st.caption("Aşağıdaki düğmeye basıp tarayıcı izin penceresinde 'İzin ver' de.")
            loc = streamlit_geolocation()
            if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
                st.success(f"Konum alındı: {loc['latitude']:.5f}, {loc['longitude']:.5f}")
                st.button(
                    "Bu konumu başlangıç yap",
                    on_click=set_start,
                    args=({"name": "Mevcut konumum", "lat": loc["latitude"], "lng": loc["longitude"],
                           "place_id": "gps"},),
                )
            else:
                st.info("Konum henüz alınmadı veya izin verilmedi. Sorun değil: 'Adres ara' veya "
                        "'Kayıtlı yerler' sekmesini kullanabilirsin (BR-02).")
        else:
            st.info("Tarayıcı konum bileşeni yüklenemedi; 'Adres ara' veya 'Kayıtlı yerler' ile devam et.")

    with sub_search:
        q = st.text_input("Adres veya yer ara", key="start_q", placeholder="Örn. Ege Üniversitesi Metro İstasyonu")
        if st.button("Ara", key="start_btn"):
            if not q.strip():
                st.warning("Lütfen aranacak bir yer yaz.")
            else:
                try:
                    st.session_state["start_res"] = geocode(q.strip(), country.strip())
                except ApiError as exc:
                    st.error(f"Arama başarısız: {exc} Birkaç saniye sonra tekrar dene.")
                    st.session_state["start_res"] = None
        res = st.session_state.get("start_res")
        if res is not None and len(res) == 0:
            st.info("Sonuç bulunamadı. Farklı yazımla dene.")
        if res:
            i = st.selectbox("Sonuç", range(len(res)), format_func=lambda k: res[k]["address"][:90], key="start_sel")
            st.button("Başlangıç yap", key="start_set", on_click=set_start, args=(res[i],))

    with sub_saved:
        for name, spot in PRESET_SPOTS.items():
            st.button(f"📌 {name}", key=f"preset_{name}", on_click=set_start,
                      args=({"name": name, "lat": spot["lat"], "lng": spot["lng"], "place_id": f"preset-{name}"},))
        st.markdown("**Kendi yerlerin** (yalnızca bu oturumda tutulur)")
        for label, spot in st.session_state.saved.items():
            cc1, cc2 = st.columns(2)
            if spot:
                cc1.button(f"🏠 {label}: {spot['name'][:25]}", key=f"use_{label}", on_click=set_start, args=(spot,))
            else:
                cc1.caption(f"{label}: kayıtlı değil")
            cc2.button(f"Başlangıcı '{label}' olarak kaydet", key=f"save_{label}", on_click=save_spot, args=(label,))

    if st.session_state.start:
        s = st.session_state.start
        st.success(f"**Seçili başlangıç:** {s['name']} ({s['lat']:.4f}, {s['lng']:.4f})")

# ---------------------------- 2) Keşfet (FR-02) ----------------------------
with tab_disc:
    st.subheader("Kategoriye göre mekan önerileri")
    cat = st.selectbox("Kategori", list(CATEGORIES.keys()))
    custom = ""
    if CATEGORIES[cat] is None:
        custom = st.text_input("Kategori / aranan şey", placeholder="Örn. deniz ürünleri, kitapçı, bisiklet kiralama")
    prefs = st.text_area(
        "Tercihlerin (isteğe bağlı)",
        placeholder="Örn. ucuz ve sakin, vegan seçenekli, deniz manzaralı",
        height=80,
    )
    radius = st.slider("Arama yarıçapı (m)", 500, 15000, 3000, 250)

    if st.button("✨ Önerileri getir", type="primary"):
        if not st.session_state.start:
            st.warning("Önce 1 · Konum sekmesinden bir başlangıç noktası seç.")
        elif not maps_key:
            st.error("Google Maps API anahtarı gerekli.")
        elif CATEGORIES[cat] is None and not custom.strip():
            st.warning("Lütfen aranacak kategoriyi yaz.")
        else:
            query_en = CATEGORIES[cat] or custom.strip()
            label = cat if CATEGORIES[cat] else custom.strip()
            with st.spinner("Mekanlar aranıyor ve yapay zekâ ile sıralanıyor..."):
                try:
                    items, notes, used_q = run_discovery(
                        st.session_state.start, label, query_en, prefs, radius,
                        maps_key, gem_key, gem_model,
                    )
                    st.session_state.discovered = {"items": items, "notes": notes, "label": label, "query": used_q}
                except ApiError as exc:
                    st.session_state.discovered = None
                    st.error(f"Öneriler alınamadı: {exc} Anahtar/kota durumunu kontrol edip tekrar dene.")

    disc = st.session_state.discovered
    if disc:
        for n in disc["notes"]:
            st.info(n)
        if not disc["items"]:
            st.info("Sonuç bulunamadı. Kategoriyi veya yarıçapı değiştirmeyi ya da 3 · Rota sekmesinden manuel arama yapmayı dene.")
        for i, p in enumerate(disc["items"]):
            with st.container(border=True):
                st.markdown(f"**{p['name']}**  \n{p['type']}")
                meta = []
                if p["rating"]:
                    meta.append(f"⭐ {p['rating']} ({p['n_ratings'] or 0})")
                if p["price"]:
                    meta.append(p["price"])
                if meta:
                    st.caption(" · ".join(meta))
                st.caption(p["address"])
                if p.get("reason"):
                    st.markdown(f"🤖 _{p['reason']}_")
                if p.get("uri"):
                    st.markdown(f"[Google Maps'te aç]({p['uri']})")
                already = place_exists(p, st.session_state.stops)
                st.button("✓ Rotada" if already else "➕ Durağa ekle", key=f"add_disc_{i}",
                          disabled=already, on_click=add_stop, args=(p,))

# ---------------------------- 3) Rota (FR-03/04) ----------------------------
with tab_route:
    st.subheader(f"Gezi planı ({len(st.session_state.stops)}/{MAX_STOPS} durak)")

    if not st.session_state.stops:
        st.info("Henüz durak yok. 2 · Keşfet sekmesinden öneri ekle veya aşağıdan manuel ara.")
    for i, s in enumerate(st.session_state.stops):
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {s['name']}**")
            st.caption(s.get("address", "")[:100])
            b1, b2, b3 = st.columns(3)
            b1.button("⬆️", key=f"up_{i}", on_click=move_stop, args=(i, -1), disabled=i == 0)
            b2.button("⬇️", key=f"down_{i}", on_click=move_stop, args=(i, 1),
                      disabled=i == len(st.session_state.stops) - 1)
            b3.button("🗑️", key=f"del_{i}", on_click=remove_stop, args=(i,))

    with st.expander("🔎 Manuel durak ara"):
        sq = st.text_input("Yer adı / adres", key="stop_q")
        if st.button("Ara", key="stop_btn"):
            if not sq.strip():
                st.warning("Lütfen aranacak bir yer yaz.")
            else:
                try:
                    st.session_state["stop_res"] = geocode(sq.strip(), country.strip())
                except ApiError as exc:
                    st.error(f"Arama başarısız: {exc}")
                    st.session_state["stop_res"] = None
        sres = st.session_state.get("stop_res")
        if sres is not None and len(sres) == 0:
            st.info("Sonuç bulunamadı.")
        if sres:
            k = st.selectbox("Sonuç", range(len(sres)), format_func=lambda j: sres[j]["address"][:90], key="stop_sel")
            st.button("➕ Durağa ekle", key="stop_add", on_click=add_stop, args=(sres[k],))

    st.divider()
    optimize = st.checkbox("🤖 Durak sırasını otomatik optimize et", value=True,
                           help="Seçilen kritere göre toplam süreyi en aza indiren sırayı bulur.")
    return_home = st.checkbox("Başlangıç noktasına geri dön", value=False)

    if st.button("🚀 Rotayı hesapla", type="primary"):
        if not st.session_state.start:
            st.warning("Hesaplama için başlangıç noktası eksik (1 · Konum).")
        elif not st.session_state.stops:
            st.warning("Hesaplama için en az bir durak seçmelisin.")
        elif not maps_key:
            st.error("Google Maps API anahtarı gerekli.")
        else:
            try:
                with st.spinner("Bacaklar hesaplanıyor..."):
                    res = compute_trip(
                        st.session_state.start, st.session_state.stops, optimize, criterion,
                        return_home, maps_key, {"base": fare_base, "ratio": fare_ratio},
                    )
                st.session_state.result = res
                st.session_state.timings.append(res["elapsed"])
                if res["optimized"]:
                    st.session_state.stops = list(res["order"])
                st.success("Rota hazır! 4 · Sonuç sekmesine geç.")
            except ApiError as exc:
                # 6.4: kota/zaman aşımı -> kısmi toplam sunma, yeniden dene mesajı
                st.session_state.result = None
                st.error(f"Rota hesaplanamadı: {exc} Kısmi toplam gösterilmedi; lütfen tekrar dene.")

# ---------------------------- 4) Sonuç (FR-05) ----------------------------
with tab_res:
    res = st.session_state.result
    if not res:
        st.info("Henüz sonuç yok. 3 · Rota sekmesinde 'Rotayı hesapla'ya bas.")
    else:
        st.subheader("Rota özeti")
        if res["optimized"]:
            st.success("Durak sırası toplam süreyi azaltacak şekilde yeniden düzenlendi: "
                       + " → ".join([res["start"]["name"]] + [s["name"] for s in res["order"]]))

        m1, m2, m3 = st.columns(3)
        prefix = "≥ " if res["missing_transit"] else ""
        m1.metric("🚌 Toplu taşıma", prefix + fmt_dur(res["t_transit"]))
        m2.metric("🚗 Özel araç", ("≥ " if res["missing_driving"] else "") + fmt_dur(res["t_driving"]))
        m3.metric("💳 Tahmini ücret", f"{res['fare_total']:.2f} {res['currency']}")

        if res["t_transit"] and res["t_driving"]:
            diff = res["t_transit"] - res["t_driving"]
            st.caption(("Toplu taşıma, araca göre " + fmt_dur(abs(diff)) + (" daha uzun." if diff > 0 else " daha kısa.")))

        if res["missing_transit"]:
            st.warning(f"{res['missing_transit']} bacak için toplu taşıma rotası bulunamadı; o bacaklarda araç süresi "
                       "gösteriliyor ve toplu taşıma toplamı eksiktir. İlgili durağı değiştirebilir veya silebilirsin.")
        # BR-03
        st.info("ℹ️ Gösterilen ücretler **tahmindir**; gerçek ücretler belediye güncellemelerine ve bilet türüne "
                "göre değişebilir. Operatörler arası dinamik aktarma indirimleri hesaplanamamaktadır.")

        rows = []
        for i, leg in enumerate(res["legs"], start=1):
            tr, dr = leg["transit"], leg["driving"]
            rows.append(
                {
                    "Bacak": f"{i}. {leg['from']} → {leg['to']}",
                    "Toplu taşıma": fmt_dur(tr["seconds"]) if tr else "Rota yok ⚠️",
                    "Araç": fmt_dur(dr["seconds"]) if dr else "Rota yok ⚠️",
                    "Mesafe": fmt_dist((dr or tr or {}).get("meters")),
                    "Ücret (tahmin)": f"{leg['fare']:.2f} {leg['currency']}" if leg["fare"] is not None else "—",
                    "Ücret kaynağı": leg["fare_src"],
                    "Hatlar": ", ".join(tr["lines"]) if tr and tr["lines"] else ("Yürüyüş" if tr else "—"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Harita")
        mode_key = st.radio("Gösterilen rota", ["transit", "driving"], horizontal=True,
                            format_func=lambda x: "🚌 Toplu taşıma" if x == "transit" else "🚗 Araç")
        st.pydeck_chart(build_map(res, mode_key), use_container_width=True)

        ok = res["elapsed"] < 3
        st.caption(f"⏱️ Hesaplama süresi: {res['elapsed']:.2f} sn "
                   + ("(✅ hedef < 3 sn)" if ok else "(⚠️ hedef 3 sn aşıldı; ilk çağrılar önbelleksizdir, tekrar hesaplayınca düşer)"))
