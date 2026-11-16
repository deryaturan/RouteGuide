import math
import time

import requests
import streamlit as st

# Page setup for mobile responsiveness
st.set_page_config(page_title="Route - Smart Travel Guide", page_icon="📍", layout="centered")

st.title("📍 Route")
st.caption("Your Independent Local Travel & Navigation Assistant")

st.markdown("---")

# ---------------------------------------------------------
# 1. STARTING LOCATION SECTION
# ---------------------------------------------------------
st.subheader("1. Starting Point")

loc_option = st.radio(
    "How would you like to set your starting location?",
    ["Manual Address (City, District, Neighborhood)", "Use GPS Coordinates (Lat, Lon)"],
    horizontal=True,
)

start_coords = None
start_address_str = ""

if loc_option == "Manual Address (City, District, Neighborhood)":
    start_address_str = st.text_input(
        "Enter Detailed Location:",
        placeholder="e.g. Alsancak, Konak, Izmir",
    )
else:
    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat_in = st.number_input("Latitude:", value=38.4237, format="%.6f")
    with col_lon:
        lon_in = st.number_input("Longitude:", value=27.1428, format="%.6f")
    start_coords = (lat_in, lon_in)

# ---------------------------------------------------------
# 2. ITINERARY CUSTOMIZATION
# ---------------------------------------------------------
st.markdown("---")
st.subheader("2. Customize Your Itinerary Stops")

stop_count = st.slider("Select Number of Stops:", min_value=2, max_value=5, value=3)

CATEGORIES = [
    "🏛️ Historical Site",
    "🖼️ Museum",
    "☕ Popular Cafe",
    "🍽️ Popular Restaurant",
    "🌳 Park & Nature",
    "🛍️ Local Market / Shopping",
]

stops_config = []

for i in range(stop_count):
    st.write(f"**Stop {i+1} Configuration:**")
    c1, c2 = st.columns(2)
    with c1:
        cat = st.selectbox(f"Category for Stop {i+1}:", CATEGORIES, key=f"cat_{i}")
    with c2:
        dist = st.text_input(
            f"District / Area for Stop {i+1}:", placeholder="e.g. Konak", key=f"dist_{i}"
        )

    stops_config.append({"category": cat, "district": dist})


# ---------------------------------------------------------
# HELPER FUNCTIONS (DISTANCE & TRANSIT TABS)
# ---------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates approximate distance between two points in km"""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def estimate_transit_times(dist_km):
    """Estimates transit, walking, and driving durations"""
    walk_min = round((dist_km / 4.5) * 60)  # 4.5 km/h walking speed
    transit_min = round((dist_km / 20.0) * 60 + 5)  # 20 km/h avg bus/metro + wait time
    drive_min = round((dist_km / 30.0) * 60 + 2)  # 30 km/h city driving
    return walk_min, transit_min, drive_min


# ---------------------------------------------------------
# 3. ROUTE GENERATION LOGIC
# ---------------------------------------------------------
st.markdown("---")

if st.button("🚀 Generate My Custom Route", type="primary"):
    headers = {"User-Agent": "RouteApp_SystemAnalysis_Project/2.0"}
    url = "https://nominatim.openstreetmap.org/search"

    route_points = []

    with st.spinner("Fetching best local spots and calculating transit routes..."):
        # Resolve Starting Coordinates if manual
        if loc_option == "Manual Address (City, District, Neighborhood)":
            if not start_address_str.strip():
                st.warning("Please enter a starting location.")
                st.stop()

            p = {"q": start_address_str, "format": "json", "limit": 1}
            try:
                r = requests.get(url, params=p, headers=headers, timeout=5).json()
                if r:
                    start_coords = (float(r[0]["lat"]), float(r[0]["lon"]))
                    start_name = r[0].get("display_name", "").split(",")[0]
                else:
                    st.error("Could not locate your starting address. Please be more specific.")
                    st.stop()
            except Exception:
                st.error("Network error resolving starting point.")
                st.stop()
        else:
            start_name = "Your Custom Location"

        route_points.append(
            {
                "name": f"Starting Point: {start_name}",
                "lat": start_coords[0],
                "lon": start_coords[1],
                "type": "Start",
            }
        )

        # Resolve Each Stop
        for idx, cfg in enumerate(stops_config, start=1):
            raw_cat = cfg["category"].split(" ")[1]  # Extract word after emoji
            query = f"{raw_cat} {cfg['district']}".strip()

            params = {
                "q": query,
                "format": "json",
                "limit": 1,
                "accept-language": "en",
            }
            try:
                time.sleep(1)  # Nominatim usage policy: max 1 request per second
                res = requests.get(url, params=params, headers=headers, timeout=5).json()
                if res:
                    place = res[0]
                    p_name = place.get("display_name", "").split(",")[0]
                    route_points.append(
                        {
                            "name": f"Stop {idx}: {p_name} ({cfg['category']})",
                            "lat": float(place["lat"]),
                            "lon": float(place["lon"]),
                            "type": cfg["category"],
                        }
                    )
            except Exception:
                continue

        # ---------------------------------------------------------
        # DISPLAY RESULTS & TRANSIT TIMELINE
        # ---------------------------------------------------------
        if len(route_points) > 1:
            st.success("🎉 Route Successfully Created!")

            total_distance = 0.0
            total_transit_time = 0
            google_coords = []

            for i in range(len(route_points)):
                pt = route_points[i]
                google_coords.append(f"{pt['lat']},{pt['lon']}")

                st.subheader(f"📍 {pt['name']}")

                # Calculate legs between stops
                if i < len(route_points) - 1:
                    next_pt = route_points[i + 1]
                    leg_dist = haversine_distance(
                        pt["lat"], pt["lon"], next_pt["lat"], next_pt["lon"]
                    )
                    walk, transit, drive = estimate_transit_times(leg_dist)

                    total_distance += leg_dist
                    total_transit_time += transit

                    # Display Transit Metrics inside a styled container
                    with st.container():
                        st.info(
                            f"**Travel to next stop ({leg_dist:.2f} km):**\n\n"
                            f"🚌 **Public Transit:** ~{transit} min  |  "
                            f"🚶‍♂️ **Walking:** ~{walk} min  |  "
                            f"🚗 **Drive/Taxi:** ~{drive} min"
                        )
                        st.markdown("↓")

            # Final Summary Box
            st.markdown("---")
            st.write("### 📊 Route Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Distance", f"{total_distance:.1f} km")
            col2.metric("Total Est. Transit Time", f"~{total_transit_time} min")
            col3.metric("Total Stops", f"{len(route_points) - 1}")

            # Google Maps Link
            maps_url = "https://www.google.com/maps/dir/" + "/".join(google_coords)
            st.link_button("🗺️ Open Interactive Route in Google Maps", maps_url)

        else:
            st.warning(
                "No locations could be found for the specified criteria. "
                "Try refining your district or category names."
            )
