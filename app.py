"""
Pedestrian Fatality Explorer -- ITE 2026 poster companion app.

Every pedestrian-involved fatal crash in FARS 2016-2024 (60,022 mapped),
each explained by the SHAP top factors of an XGBoost involvement model
and translated into a plain-language narrative with a candidate
countermeasure.

Data contract: reads ONLY ped_map_data.parquet (built by 06_build_geopackage.py).
Retrain the model -> rerun pipeline -> replace the parquet -> app updates.
"""

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import FastMarkerCluster

st.set_page_config(page_title="Pedestrian Fatality Explorer",
                   layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ data
@st.cache_data
def load_data():
    df = pd.read_parquet("ped_map_data.parquet")
    df["ped_age"] = df.ped_age.astype("Int64")
    return df

df = load_data()

FACTOR_FAMILIES = {   # legend: color -> meaning
    "#c53030": "Crossing / junction context",
    "#dd6b20": "Road design (arterials)",
    "#6b46c1": "Time of day",
    "#2c5282": "Lighting",
    "#2f855a": "Weather / exposure",
    "#718096": "Other",
}

# ------------------------------------------------------------------ header
st.title("🚶 Pedestrian Fatality Explorer")
st.markdown(
    "**60,022 pedestrian deaths on U.S. roads (2016–2024), each one explained.** "
    "Dots are colored by the condition that most set that crash apart, according "
    "to an explainable machine-learning model (XGBoost + SHAP) trained on FARS data. "
    "Tap any dot to read what happened and what could prevent the next one."
)

mode = st.radio("Mode", ["🗺️ Explore the map", "🚶 Check a walking route"],
                horizontal=True, label_visibility="collapsed")

# ================================================================== explore
if mode == "🗺️ Explore the map":
    c1, c2, c3 = st.columns([1.2, 1, 1.6])
    with c1:
        states = ["Entire U.S. (clustered)"] + sorted(df.STATENAME.dropna().unique())
        state = st.selectbox("State", states, index=states.index("Alabama"))
    with c2:
        years = st.slider("Years", 2016, 2024, (2016, 2024))
    with c3:
        factors = st.multiselect("Filter by top factor (optional)",
                                 sorted(df.factor_label.unique()))

    view = df[(df.YEAR >= years[0]) & (df.YEAR <= years[1])]
    if factors:
        view = view[view.factor_label.isin(factors)]

    col_map, col_side = st.columns([2.6, 1])

    with col_map:
        if state == "Entire U.S. (clustered)":
            m = folium.Map(location=[39.5, -96.5], zoom_start=5, tiles="cartodbpositron", prefer_canvas=True)
            FastMarkerCluster(view[["LATITUDE", "LONGITUDE"]].values.tolist()).add_to(m)
            st.caption("Zoom in to break clusters apart. Pick a state for "
                       "individually explained crashes.")
        else:
            view = view[view.STATENAME == state]
            m = folium.Map(location=[view.LATITUDE.mean(), view.LONGITUDE.mean()],
                           zoom_start=7, tiles="cartodbpositron", prefer_canvas=True)
            if len(view) > 5000:
                FastMarkerCluster(view[["LATITUDE", "LONGITUDE"]].values.tolist()).add_to(m)
                st.warning(f"{len(view):,} points - clustered for speed. Narrow the "
                           "years or factors to see individually explained dots.")
                view_iter = view.iloc[:0]
            else:
                view_iter = view
            for r in view_iter.itertuples():
                folium.CircleMarker(
                    location=[r.LATITUDE, r.LONGITUDE],
                    radius=5, color=r.factor_color, fill=True, fill_opacity=0.85,
                    weight=1,
                    tooltip=f"{r.factor_label} ({r.YEAR})",
                    popup=folium.Popup(
                        f"<div style='width:260px;font-size:13px'>"
                        f"<b style='color:{r.factor_color}'>{r.factor_label}</b><br>"
                        f"{r.narrative}</div>", max_width=280),
                ).add_to(m)
        st_folium(m, height=560, use_container_width=True, returned_objects=[])

    with col_side:
        st.subheader(f"{len(view):,} deaths shown")
        st.markdown("**Legend (dot colors)**")
        for color, label in FACTOR_FAMILIES.items():
            st.markdown(
                f"<span style='color:{color};font-size:18px'>●</span> {label}",
                unsafe_allow_html=True)
        st.markdown("**Top factors here**")
        st.dataframe(view.factor_label.value_counts().head(8).rename("deaths"),
                     use_container_width=True)

# ================================================================== route
else:
    from backend_route import get_route_crashes

    st.markdown("Enter a walking (or driving) trip; the app shows every "
                "pedestrian death within the corridor and why it happened.")
    c1, c2, c3, c4 = st.columns([1.6, 1.6, 1, 1])
    with c1:
        origin = st.text_input("From", placeholder="e.g., Bryant-Denny Stadium, Tuscaloosa")
    with c2:
        destination = st.text_input("To", placeholder="e.g., Downtown Tuscaloosa")
    with c3:
        profile = st.selectbox("Mode", ["foot-walking", "driving-car"])
    with c4:
        buffer_m = st.selectbox("Corridor width (m)", [50, 100, 200, 400], index=1)

    if st.button("Find pedestrian deaths along this route", type="primary"):
        if not origin.strip() or not destination.strip():
            st.error("Please enter both locations.")
        else:
            with st.spinner("Routing and scanning corridor..."):
                st.session_state.route_res = get_route_crashes(
                    df, origin, destination, profile, buffer_m)

    res = st.session_state.get("route_res")
    if res:
        if "error" in res:
            st.error(res["error"])
            if "details" in res:
                st.caption(res["details"])
        else:
            crashes = pd.DataFrame(res["crashes"])
            st.subheader(f"{len(crashes)} pedestrian deaths within "
                         f"{buffer_m} m of this {res['distance_km']} km route")
            col_map, col_side = st.columns([2.6, 1])
            with col_map:
                rc = res["route_coords"]
                m = folium.Map(location=[rc[0][1], rc[0][0]], zoom_start=13,
                               tiles="cartodbpositron", prefer_canvas=True)
                folium.PolyLine([(lat, lon) for lon, lat in rc],
                                color="#3182ce", weight=4, opacity=0.8).add_to(m)
                for r in crashes.itertuples():
                    folium.CircleMarker(
                        location=[r.LATITUDE, r.LONGITUDE],
                        radius=6, color=r.factor_color, fill=True, fill_opacity=0.9,
                        weight=1, tooltip=f"{r.factor_label} ({r.YEAR})",
                        popup=folium.Popup(
                            f"<div style='width:260px;font-size:13px'>"
                            f"<b style='color:{r.factor_color}'>{r.factor_label}</b><br>"
                            f"{r.narrative}</div>", max_width=280),
                    ).add_to(m)
                st_folium(m, height=520, use_container_width=True, returned_objects=[])
            with col_side:
                if len(crashes):
                    st.markdown("**Corridor risk pattern**")
                    st.dataframe(crashes.factor_label.value_counts().rename("deaths"),
                                 use_container_width=True)
                    top = crashes.factor_label.value_counts().index[0]
                    st.info(f"Dominant factor on this corridor: **{top}**")
                else:
                    st.success("No recorded pedestrian deaths in this corridor "
                               "(2016–2024). That is not a guarantee of safety.")

st.caption("Data: NHTSA FARS 2016–2024 · Model: XGBoost + SHAP (context features "
           "only) · Narratives: LLM-assisted, precomputed · Built for the ITE 2026 "
           "Annual Meeting poster session. Historical records, not predictions; "
           "absence of dots ≠ safety.")
