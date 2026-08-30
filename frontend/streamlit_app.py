"""streamlit_app.py — Car Recommender UI (thin client)

Talks only to the FastAPI backend (car_recommender.api.main) over HTTP — no
model loading, no DataFrame, no Anthropic calls happen in this process.
Run: streamlit run frontend/streamlit_app.py

Two secrets, both optional and independent:
  APP_PASSWORD  — human-facing login gate for this Streamlit page.
  API_KEY       — sent as X-API-Key to the backend on every request; only
                  needed if the backend was started with API_PASSWORD set.
"""

import hmac
import os
import re
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent


def _load_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:  # noqa: BLE001, S110 - st.secrets raises a generic error when unconfigured; fall through on purpose
        pass
    val = os.environ.get(key, "")
    if val:
        return val
    env_file = BASE.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith((key + "=", key + " =")):
                return line.split("=", 1)[-1].strip().strip('"').strip("'")
    return default


API_BASE_URL = _load_secret("API_BASE_URL", "http://localhost:8000")
API_KEY = _load_secret("API_KEY")
APP_PASSWORD = _load_secret("APP_PASSWORD")

_MAX_INPUT_LEN = 1000
_MAX_TURNS = 20

_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return text.strip()[:_MAX_INPUT_LEN]


def _check_password() -> bool:
    if not APP_PASSWORD:
        return True  # no password configured = open (local dev)
    if st.session_state.get("authenticated"):
        return True
    with st.form("login"):
        st.subheader("Car Recommender")
        pwd = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if hmac.compare_digest(pwd.encode(), APP_PASSWORD.encode()):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


st.set_page_config(page_title="Car Recommender", layout="wide")

if not _check_password():
    st.stop()


@st.cache_resource
def get_api_client() -> httpx.Client:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    return httpx.Client(base_url=API_BASE_URL, headers=headers, timeout=30.0)


client = get_api_client()


def api_get(path: str) -> dict:
    resp = client.get(path)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: dict) -> httpx.Response:
    return client.post(path, json=payload)


@st.cache_data(ttl=300)
def load_filter_options() -> dict:
    return api_get("/cars/filters")


try:
    filters_meta = load_filter_options()
except httpx.HTTPError as e:
    st.error(f"Could not reach the API at {API_BASE_URL}: {e}")
    st.stop()

st.title("Car Recommender")
st.caption("Used car catalog 2001-2024 — results ranked by predicted buyer satisfaction.")

tab_browse, tab_advisor = st.tabs(["Browse", "Advisor"])

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")


def multiselect_or_all(label, options):
    chosen = st.sidebar.multiselect(label, options)
    return chosen if chosen else options


sel_bodytype = multiselect_or_all("Body Type", filters_meta["bodytypes"])
sel_make = multiselect_or_all("Make", filters_meta["makes"])
sel_fuel = multiselect_or_all("Fuel Type", filters_meta["fuel_types"])
sel_drive = multiselect_or_all("Drivetrain", filters_meta["drivetrains"])
sel_trans = multiselect_or_all("Transmission", filters_meta["transmissions"])

st.sidebar.divider()

price_min, price_max = int(filters_meta["price_min"]), int(filters_meta["price_max"])
col_a, col_b = st.sidebar.columns(2)
sel_min_price = col_a.number_input("Min Price ($)", min_value=price_min, max_value=price_max, value=price_min, step=500)
sel_price = col_b.number_input("Max Price ($)", min_value=price_min, max_value=price_max, value=price_max, step=500)

mpg_max = int(filters_meta["mpg_max"])
col_a, col_b = st.sidebar.columns(2)
sel_mpg = col_a.number_input("Min MPG", min_value=0, max_value=mpg_max, value=0)
sel_max_mpg = col_b.number_input("Max MPG", min_value=0, max_value=mpg_max, value=mpg_max)

hp_max = int(filters_meta["hp_max"])
col_a, col_b = st.sidebar.columns(2)
sel_hp = col_a.number_input("Min HP", min_value=0, max_value=hp_max, value=0)
sel_max_hp = col_b.number_input("Max HP", min_value=0, max_value=hp_max, value=hp_max)

year_min, year_max = int(filters_meta["year_min"]), int(filters_meta["year_max"])
single_year = st.sidebar.number_input("Exact Year (0 = any)", min_value=0, max_value=year_max, value=0)
if single_year == 0:
    col_a, col_b = st.sidebar.columns(2)
    sel_year_min = col_a.number_input("From Year", min_value=year_min, max_value=year_max, value=year_min)
    sel_year_max = col_b.number_input("To Year", min_value=year_min, max_value=year_max, value=year_max)
else:
    sel_year_min = sel_year_max = single_year

st.sidebar.divider()

pred_min = float(filters_meta["predicted_rating_min"])
pred_max = float(filters_meta["predicted_rating_max"])
col_a, col_b = st.sidebar.columns(2)
sel_pred_min = col_a.number_input("Min Predicted", min_value=pred_min, max_value=pred_max, value=pred_min, step=0.1, format="%.1f")
sel_pred_max = col_b.number_input("Max Predicted", min_value=pred_min, max_value=pred_max, value=pred_max, step=0.1, format="%.1f")

st.sidebar.divider()

sel_ev = st.sidebar.checkbox("Electric / EV only")
sel_luxury = st.sidebar.checkbox("Luxury brands only")

top_n = st.sidebar.number_input("Results to show", min_value=1, max_value=50, value=10)


def _active_filter_lines() -> list[str]:
    lines = []
    if len(sel_make) < len(filters_meta["makes"]):
        lines.append(f"Make: {', '.join(sel_make)}")
    if len(sel_bodytype) < len(filters_meta["bodytypes"]):
        lines.append(f"Body type: {', '.join(sel_bodytype)}")
    if len(sel_fuel) < len(filters_meta["fuel_types"]):
        lines.append(f"Fuel type: {', '.join(sel_fuel)}")
    if len(sel_drive) < len(filters_meta["drivetrains"]):
        lines.append(f"Drivetrain: {', '.join(sel_drive)}")
    if len(sel_trans) < len(filters_meta["transmissions"]):
        lines.append(f"Transmission: {', '.join(sel_trans)}")
    if sel_min_price > price_min or sel_price < price_max:
        lines.append(f"Price: ${sel_min_price:,.0f} – ${sel_price:,.0f}")
    if sel_mpg > 0 or sel_max_mpg < mpg_max:
        lines.append(f"MPG: {sel_mpg} – {sel_max_mpg}")
    if sel_hp > 0 or sel_max_hp < hp_max:
        lines.append(f"HP: {sel_hp} – {sel_max_hp}")
    if sel_year_min > year_min or sel_year_max < year_max:
        lines.append(f"Year: {sel_year_min} – {sel_year_max}")
    if sel_ev:
        lines.append("Electric / EV only")
    if sel_luxury:
        lines.append("Luxury brands only")
    return lines


active_filter_lines = _active_filter_lines()


def build_recommendation_request() -> dict:
    req = {
        "bodytype": sel_bodytype,
        "make": sel_make,
        "fuel_type": sel_fuel,
        "drivetrain": sel_drive,
        "transmission_type": sel_trans,
        "min_price": sel_min_price,
        "max_price": sel_price,
        "min_mpg_comb": sel_mpg,
        "max_mpg_comb": sel_max_mpg,
        "min_hp": sel_hp,
        "max_hp": sel_max_hp,
        "min_year": sel_year_min,
        "max_year": sel_year_max,
        "min_predicted_rating": sel_pred_min,
        "max_predicted_rating": sel_pred_max,
        "top_n": top_n,
    }
    if sel_ev:
        req["is_ev"] = True
    if sel_luxury:
        req["luxury"] = True
    return req


# ---------------------------------------------------------------------------
# Tab 1 — Browse
# ---------------------------------------------------------------------------
with tab_browse:
    resp = api_post("/recommendations", build_recommendation_request())
    if resp.status_code != 200:
        st.error(f"Search failed: {resp.text}")
        st.stop()

    payload = resp.json()
    cars = payload["cars"]
    total_matched = payload["total_matched"]

    st.subheader(f"{total_matched:,} cars match — top {len(cars)} unique models shown")

    if not cars:
        st.warning("No cars match the current filters. Try loosening your criteria.")
    else:
        df_display = pd.DataFrame(cars)
        display_cols = {
            "make": "Make", "model": "Model", "year": "Year", "trim": "Trim",
            "price": "Price ($)", "mpg_comb": "MPG", "hp": "HP",
            "drivetrain": "Drivetrain", "fuel_type": "Fuel",
            "consumer_overall_rating": "Actual Rating",
            "predicted_consumer_rating": "Predicted Rating",
        }
        out = df_display[list(display_cols)].rename(columns=display_cols)
        out["Price ($)"] = out["Price ($)"].apply(lambda x: f"${x:,.0f}")

        st.dataframe(
            out.style.background_gradient(
                subset=["Predicted Rating"], cmap="RdYlGn", vmin=3.5, vmax=5.0
            ).format({
                "Predicted Rating": "{:.2f}",
                "Actual Rating": lambda x: f"{x:.1f}" if pd.notna(x) else "—",
            }),
            use_container_width=True,
            height=min(50 + len(out) * 35, 600),
        )

        st.caption(
            "Actual Rating = aggregated buyer reviews (model-year level). "
            "Predicted Rating = Random Forest estimate from car specs + expert ratings. "
            "Prices shown are new-car MSRP — used market prices are typically 30-60% lower."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Predicted Rating", f"{df_display['predicted_consumer_rating'].mean():.2f}")
        col2.metric("Avg Price", f"${df_display['price'].mean():,.0f}")
        col3.metric("Avg MPG", f"{df_display['mpg_comb'].mean():.1f}")
        col4.metric("Avg HP", f"{df_display['hp'].mean():.0f}")

        st.divider()
        st.subheader("Car Detail")

        pick_options = [f"{int(c['year'])} {c['make']} {c['model']}" for c in cars]
        picked_label = st.selectbox("Select a car to inspect", options=["—"] + pick_options)

        if picked_label != "—":
            idx = pick_options.index(picked_label)
            car_id = cars[idx]["car_id"]
            detail = api_get(f"/cars/{car_id}")

            st.markdown(f"### {detail['year']} {detail['make']} {detail['model']} — {detail['trim'] or ''}")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"${detail['price']:,.0f}")
            c2.metric("MPG (combined)", f"{detail['mpg_comb']:.0f}")
            c3.metric("Horsepower", f"{detail['hp']:.0f}" + (" ~" if detail["hp_imputed"] else ""))
            c4.metric("Torque (lb-ft)", f"{detail['torque_lbft']:.0f}" + (" ~" if detail["torque_imputed"] else ""))
            c5.metric("Cargo (cu ft)", f"{detail['cargo_cuft']:.1f}" if detail["cargo_cuft"] else "—")

            if detail["hp_imputed"] or detail["torque_imputed"]:
                st.caption("~ = ML-imputed (raw data was missing)")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Drivetrain", detail["drivetrain"])
            c2.metric("Transmission", detail["transmission_type"])
            c3.metric("Fuel Type", detail["fuel_type"])
            c4.metric("Doors", str(detail["doors"]))
            c5.metric("Body Type", detail["bodytype"])

            st.divider()
            col_exp, col_con = st.columns(2)

            with col_exp:
                st.markdown("**Expert Ratings**")
                ratings = detail["expert_ratings"]
                for label, key in [
                    ("Value", "value"), ("Performance", "performance"), ("Quality", "quality"),
                    ("Comfort", "comfort"), ("Reliability", "reliability"), ("Styling", "styling"),
                ]:
                    st.write(f"{label}: **{ratings[key]:.1f}** / 5.0")

            with col_con:
                st.markdown("**Consumer Rating**")
                actual = detail["consumer_overall_rating"]
                count = detail["consumer_review_count"]
                st.write(f"Overall: **{actual:.1f}** / 5.0" if actual is not None else "No consumer rating")
                st.write(f"Based on {count:,} reviews" if count else "No consumer reviews")
                st.write(f"Predicted: **{detail['predicted_consumer_rating']:.2f}** / 5.0")

            st.divider()

            labels = {0: "Not Available", 1: "Optional", 2: "Standard"}
            st.markdown("**Features**")
            fc1, fc2, fc3 = st.columns(3)
            for col, group in zip([fc1, fc2, fc3], ["safety", "connectivity", "convenience"]):
                col.markdown(f"*{group.capitalize()}*")
                for feat, val in detail["features"][group].items():
                    col.write(f"{labels[val]} — {feat}")

            with st.expander("Show more specs"):
                s1, s2, s3, s4 = st.columns(4)

                s1.markdown("**Fuel Economy**")
                s1.write(f"City:     {detail['mpg_city']:.0f} MPG")
                s1.write(f"Highway:  {detail['mpg_hwy']:.0f} MPG")
                s1.write(f"Combined: {detail['mpg_comb']:.0f} MPG")

                s2.markdown("**Performance**")
                s2.write(f"Horsepower:  {detail['hp']:.0f} hp" + (" *(imputed)*" if detail["hp_imputed"] else ""))
                s2.write(f"Torque:      {detail['torque_lbft']:.0f} lb-ft" + (" *(imputed)*" if detail["torque_imputed"] else ""))

                s3.markdown("**Classification**")
                s3.write(f"Electric:  {'Yes' if detail['is_ev'] else 'No'}")
                s3.write(f"Luxury:    {'Yes' if detail['luxury'] else 'No'}")
                s3.write(f"Car ID:    {detail['car_id']}")

                s4.markdown("**Engine**")
                s4.write(detail["engine"] or "—")


# ---------------------------------------------------------------------------
# Tab 2 — Advisor
# ---------------------------------------------------------------------------
with tab_advisor:
    st.caption(
        "Describe what you are looking for in plain language. "
        "The advisor will search the catalog and explain the results."
    )

    if active_filter_lines:
        st.info(
            "Sidebar filters active — the advisor will only recommend from this filtered pool. "
            "Clear filters in the sidebar to search the full catalog."
        )

    if "advisor_history" not in st.session_state:
        st.session_state.advisor_history = []  # opaque, server-owned — round-tripped each turn
        st.session_state.advisor_display = []  # [(role, text), ...] for rendering only

    for role, text in st.session_state.advisor_display:
        with st.chat_message(role):
            st.markdown(text)

    user_input = st.chat_input("Tell me what you are looking for...")

    if user_input:
        user_input = _sanitize(user_input)
        if not user_input:
            st.warning("Please enter a valid message.")
        elif len(st.session_state.advisor_display) >= _MAX_TURNS * 2:
            st.warning("Conversation limit reached. Please clear and start a new one.")
        else:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.advisor_display.append(("user", user_input))

            with st.spinner("Searching catalog..."):
                resp = api_post("/advisor/chat", {
                    "message": user_input,
                    "history": st.session_state.advisor_history,
                    "active_filters": active_filter_lines,
                })

            if resp.status_code == 200:
                body = resp.json()
                st.session_state.advisor_history = body["history"]
                with st.chat_message("assistant"):
                    st.markdown(body["reply"])
                st.session_state.advisor_display.append(("assistant", body["reply"]))
            elif resp.status_code == 429:
                st.error("Too many requests. Please wait a moment and try again.")
            elif resp.status_code == 503:
                st.error("Advisor is not configured on the server (missing ANTHROPIC_API_KEY).")
            else:
                detail = resp.json().get("detail", resp.text) if resp.content else resp.text
                st.error(f"The advisor could not process your request: {detail}")

    if st.session_state.get("advisor_display") and st.button("Clear conversation"):
        st.session_state.advisor_history = []
        st.session_state.advisor_display = []
        st.rerun()
