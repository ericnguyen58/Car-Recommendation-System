"""
app.py — Car Recommender UI
Run: streamlit run app.py
"""

import hmac
import json
import logging
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE       = Path(__file__).parent
MODELS_DIR = BASE / 'models'
DATA_PATH  = BASE / 'data' / 'final' / 'model_ready.UVCRS'
LOG_DIR    = BASE / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Audit logging  — no API keys, no secrets, no full conversation content
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename=LOG_DIR / 'advisor.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Secrets  — priority: st.secrets → env var → .env file
# ---------------------------------------------------------------------------
def _load_secret(key: str, default: str = '') -> str:
    try:
        return st.secrets[key]
    except Exception:
        pass
    val = os.environ.get(key, '')
    if val:
        return val
    env_file = BASE / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(key + '=') or line.startswith(key + ' ='):
                return line.split('=', 1)[-1].strip().strip('"').strip("'")
    return default

# ---------------------------------------------------------------------------
# Password gate  — set APP_PASSWORD in .env / st.secrets to enable
# ---------------------------------------------------------------------------
def _check_password() -> bool:
    expected = _load_secret('APP_PASSWORD', '')
    if not expected:
        return True                          # no password configured = open (local dev)
    if st.session_state.get('authenticated'):
        return True
    with st.form('login'):
        st.subheader('Car Recommender')
        pwd = st.text_input('Password', type='password')
        if st.form_submit_button('Login'):
            if hmac.compare_digest(pwd.encode(), expected.encode()):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error('Incorrect password.')
                _log.warning('Failed login attempt.')
    return False

# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------
_MAX_INPUT_LEN = 1000

def _sanitize(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)                             # strip HTML tags
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # control chars
    return text.strip()[:_MAX_INPUT_LEN]

# ---------------------------------------------------------------------------
# Rate limiter  — per session, resets after _RATE_WINDOW seconds
# ---------------------------------------------------------------------------
_RATE_LIMIT  = 30    # max requests
_RATE_WINDOW = 3600  # seconds (1 hour)

def _within_rate_limit() -> bool:
    now        = time.time()
    timestamps = [t for t in st.session_state.get('req_timestamps', [])
                  if now - t < _RATE_WINDOW]
    if len(timestamps) >= _RATE_LIMIT:
        return False
    timestamps.append(now)
    st.session_state.req_timestamps = timestamps
    return True

# ---------------------------------------------------------------------------
# Page config + auth gate
# ---------------------------------------------------------------------------
st.set_page_config(page_title='Car Recommender', layout='wide')

if not _check_password():
    st.stop()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ['make', 'bodytype', 'Fuel Type', 'Drivetrain', 'Transmission Type']

NUMERIC_FILTERS = {
    'max_price':      ('price',      '<='),
    'min_price':      ('price',      '>='),
    'min_mpg_comb':   ('mpg_comb',   '>='),
    'max_mpg_comb':   ('mpg_comb',   '<='),
    'min_hp':         ('hp',         '>='),
    'max_hp':         ('hp',         '<='),
    'min_year':       ('year',       '>='),
    'max_year':       ('year',       '<='),
    'min_cargo_cuft': ('cargo_cuft', '>='),
}

_NUMERIC_BOUNDS = {
    'max_price':      (0,    500_000),
    'min_price':      (0,    500_000),
    'min_mpg_comb':   (0,    200),
    'max_mpg_comb':   (0,    200),
    'min_hp':         (0,    2000),
    'max_hp':         (0,    2000),
    'min_year':       (2001, 2024),
    'max_year':       (2001, 2024),
    'min_cargo_cuft': (0,    500),
}

# ---------------------------------------------------------------------------
# Load once (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_everything():
    rf      = joblib.load(MODELS_DIR / 'rf_recommender.joblib')
    le_dict = joblib.load(MODELS_DIR / 'label_encoders.joblib')
    with open(MODELS_DIR / 'feature_list.json') as f:
        feat_list = json.load(f)

    df = pd.read_csv(DATA_PATH)
    for col in CATEGORICAL_FEATURES:
        le    = le_dict[col]
        known = set(le.classes_)
        df[col + '_enc'] = df[col].astype(str).apply(
            lambda v, le=le, known=known: le.transform([v])[0] if v in known else -1
        )
    df['predicted_consumer_rating'] = rf.predict(df[feat_list]).round(2)
    return df

df = load_everything()

# Categorical allowlists built after df loads
_KNOWN_CATEGORICAL = {
    col: set(df[col].dropna().astype(str).str.lower().unique())
    for col in CATEGORICAL_FEATURES
}

# ---------------------------------------------------------------------------
# Advisor helpers
# ---------------------------------------------------------------------------
def _validate_tool_input(raw: dict) -> dict:
    """Sanitize and bounds-check every field from Claude before it touches the DataFrame."""
    safe = {}

    for key, df_col in [('make', 'make'), ('bodytype', 'bodytype'), ('Drivetrain', 'Drivetrain')]:
        if key in raw:
            vals = raw[key] if isinstance(raw[key], list) else [raw[key]]
            allowed = _KNOWN_CATEGORICAL[df_col]
            safe[key] = [str(v)[:64] for v in vals
                         if isinstance(v, str) and v.lower() in allowed]

    if 'fuel_type' in raw:
        vals = raw['fuel_type'] if isinstance(raw['fuel_type'], list) else [raw['fuel_type']]
        allowed = _KNOWN_CATEGORICAL['Fuel Type']
        safe['fuel_type'] = [str(v)[:64] for v in vals
                             if isinstance(v, str) and v.lower() in allowed]

    for key in ('is_ev', 'luxury'):
        if key in raw:
            safe[key] = bool(raw[key])

    for key, (lo, hi) in _NUMERIC_BOUNDS.items():
        if key in raw:
            try:
                safe[key] = max(lo, min(hi, float(raw[key])))
            except (TypeError, ValueError):
                pass

    if 'top_n' in raw:
        try:
            safe['top_n'] = max(1, min(20, int(raw['top_n'])))
        except (TypeError, ValueError):
            safe['top_n'] = 5

    return safe


def _apply_filters(prefs: dict, base_df: pd.DataFrame | None = None) -> pd.DataFrame:
    source = base_df if base_df is not None else df
    mask = pd.Series(True, index=source.index)
    if 'fuel_type' in prefs:
        prefs = dict(prefs)
        prefs['Fuel Type'] = prefs.pop('fuel_type')
    for key in CATEGORICAL_FEATURES:
        if key not in prefs:
            continue
        val  = prefs[key]
        vals = [v.lower() for v in (val if isinstance(val, list) else [val])]
        mask &= source[key].str.lower().isin(vals)
    if 'is_ev'  in prefs: mask &= source['is_ev']  == int(bool(prefs['is_ev']))
    if 'luxury' in prefs: mask &= source['luxury'] == int(bool(prefs['luxury']))
    for key, (col, op) in NUMERIC_FILTERS.items():
        if key not in prefs:
            continue
        try:
            val = float(prefs[key])
            mask &= (source[col] <= val) if op == '<=' else (source[col] >= val)
        except (TypeError, ValueError):
            pass
    return source[mask]


def _get_recommendations(prefs: dict, top_n: int = 5, base_df: pd.DataFrame | None = None) -> dict:
    results = _apply_filters(prefs, base_df=base_df)
    if results.empty:
        return {'count': 0, 'message': 'No cars matched those filters. Try relaxing one or more criteria.'}
    display_cols = [c for c in [
        'make', 'model', 'year', 'trim', 'bodytype', 'Fuel Type', 'Drivetrain',
        'price', 'mpg_comb', 'hp', 'rating_reliability', 'predicted_consumer_rating',
    ] if c in results.columns]
    top = (
        results[display_cols]
        .sort_values('predicted_consumer_rating', ascending=False)
        .drop_duplicates(subset=['make', 'model', 'year'])
        .head(top_n)
        .reset_index(drop=True)
    )
    top.index += 1
    records = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for _, row in top.iterrows()]
    return {
        'count':         len(records),
        'total_matched': int(len(results)),
        'cars':          records,
        'note':          'Prices shown are new-car MSRP — used market prices will be significantly lower (often 30-60% less).',
    }


def _execute_tool(name: str, tool_input: dict) -> str:
    if name != 'get_car_recommendations':
        _log.warning('Unknown tool requested: %s', name[:64])
        return json.dumps({'error': 'Unknown tool.'})
    validated = _validate_tool_input(dict(tool_input))
    top_n     = validated.pop('top_n', 5)
    base_df   = st.session_state.get('sidebar_filtered_df')
    result    = _get_recommendations(validated, top_n=top_n, base_df=base_df)
    _log.info('Tool call: filters=%s matched=%s', list(validated.keys()), result.get('total_matched', 0))
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Advisor constants (built after df is loaded)
# ---------------------------------------------------------------------------
_valid_bodytypes = ', '.join(sorted(df['bodytype'].dropna().unique()))
_valid_drives    = ', '.join(sorted(df['Drivetrain'].dropna().unique()))
_valid_fuels     = ', '.join(sorted(df['Fuel Type'].dropna().unique())[:12])

ADVISOR_SYSTEM = f"""You are a knowledgeable, friendly car buying advisor helping first-time buyers \
find a reliable used car. The catalog covers model years 2001-2024 across all major makes.

You have one tool: get_car_recommendations. Use it whenever the buyer asks for \
suggestions, wants to compare options, or mentions any preference that maps to a filter.

When presenting results:
- Highlight the top 2-3 picks and explain specifically why each fits the buyer's stated needs
- Always call out the reliability rating (out of 5.0) when available — this is the most \
important factor for used car buyers
- Mention fuel economy for budget-conscious buyers
- Remind buyers that listed prices are new-car MSRP, not used market value
- If a search returns zero results, diagnose which filter is likely too strict, \
suggest an adjustment, and call the tool again with relaxed criteria
- Keep explanations plain, concise, and free of jargon

Valid catalog values for reference:
  Body types  : {_valid_bodytypes}
  Drivetrains : {_valid_drives}
  Fuel types  : {_valid_fuels} (and more)
"""

ADVISOR_TOOLS = [
    {
        "name": "get_car_recommendations",
        "description": (
            "Search the car catalog and return the top-N cars ranked by predicted buyer "
            "satisfaction. Call this whenever the user wants recommendations or mentions "
            "any filtering criteria such as budget, body type, fuel economy, or brand."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "make": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more car makes to include. Omit to search all makes.",
                },
                "bodytype": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more body types: Sedan, SUV, Pickup, Hatchback, Coupe, Convertible, Wagon, Minivan, Van.",
                },
                "min_year":     {"type": "integer", "description": "Minimum model year."},
                "max_year":     {"type": "integer", "description": "Maximum model year."},
                "max_price":    {"type": "number",  "description": "Maximum MSRP in USD (new car price — quality proxy only)."},
                "min_price":    {"type": "number",  "description": "Minimum MSRP in USD."},
                "min_mpg_comb": {"type": "number",  "description": "Minimum combined MPG."},
                "min_hp":       {"type": "number",  "description": "Minimum horsepower."},
                "Drivetrain": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more drivetrains: FWD, AWD, RWD, 4WD, 2WD.",
                },
                "fuel_type": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more fuel types, e.g. ['Gasoline'], ['Hybrid'], ['Electric'].",
                },
                "is_ev":   {"type": "boolean", "description": "Set true to show only battery-electric vehicles."},
                "luxury":  {"type": "boolean", "description": "Set true to show only luxury/premium brand vehicles."},
                "top_n":   {"type": "integer", "description": "How many recommendations to return (1-20, default 5)."},
            },
            "required": [],
        },
    }
]

# ---------------------------------------------------------------------------
# Page header + tabs
# ---------------------------------------------------------------------------
st.title('Car Recommender')
st.caption('Used car catalog 2001-2024 — results ranked by predicted buyer satisfaction.')

tab_browse, tab_advisor = st.tabs(['Browse', 'Advisor'])

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header('Filters')

def multiselect_or_all(label, options):
    chosen = st.sidebar.multiselect(label, options)
    return chosen if chosen else options

bodytypes     = sorted(df['bodytype'].dropna().unique())
makes         = sorted(df['make'].dropna().unique())
fuel_types    = sorted(df['Fuel Type'].dropna().unique())
drivetrains   = sorted(df['Drivetrain'].dropna().unique())
transmissions = sorted(df['Transmission Type'].dropna().unique())

sel_bodytype  = multiselect_or_all('Body Type', bodytypes)
sel_make      = multiselect_or_all('Make', makes)
sel_fuel      = multiselect_or_all('Fuel Type', fuel_types)
sel_drive     = multiselect_or_all('Drivetrain', drivetrains)
sel_trans     = multiselect_or_all('Transmission', transmissions)

st.sidebar.divider()

price_min, price_max = int(df['price'].min()), int(df['price'].max())
col_a, col_b = st.sidebar.columns(2)
sel_min_price = col_a.number_input('Min Price ($)', min_value=price_min, max_value=price_max, value=price_min, step=500)
sel_price     = col_b.number_input('Max Price ($)', min_value=price_min, max_value=price_max, value=price_max, step=500)

mpg_max = int(df['mpg_comb'].max())
col_a, col_b = st.sidebar.columns(2)
sel_mpg     = col_a.number_input('Min MPG', min_value=0, max_value=mpg_max, value=0)
sel_max_mpg = col_b.number_input('Max MPG', min_value=0, max_value=mpg_max, value=mpg_max)

hp_max = int(df['hp'].max())
col_a, col_b = st.sidebar.columns(2)
sel_hp     = col_a.number_input('Min HP', min_value=0, max_value=hp_max, value=0)
sel_max_hp = col_b.number_input('Max HP', min_value=0, max_value=hp_max, value=hp_max)

year_min, year_max = int(df['year'].min()), int(df['year'].max())
single_year = st.sidebar.number_input('Exact Year (0 = any)', min_value=0, max_value=year_max, value=0)
if single_year == 0:
    col_a, col_b = st.sidebar.columns(2)
    sel_year_min = col_a.number_input('From Year', min_value=year_min, max_value=year_max, value=year_min)
    sel_year_max = col_b.number_input('To Year',   min_value=year_min, max_value=year_max, value=year_max)
else:
    sel_year_min = sel_year_max = single_year

st.sidebar.divider()

pred_min = float(df['predicted_consumer_rating'].min())
pred_max = float(df['predicted_consumer_rating'].max())
col_a, col_b = st.sidebar.columns(2)
sel_pred_min = col_a.number_input('Min Predicted', min_value=pred_min, max_value=pred_max, value=pred_min, step=0.1, format='%.1f')
sel_pred_max = col_b.number_input('Max Predicted', min_value=pred_min, max_value=pred_max, value=pred_max, step=0.1, format='%.1f')

st.sidebar.divider()

sel_ev     = st.sidebar.checkbox('Electric / EV only')
sel_luxury = st.sidebar.checkbox('Luxury brands only')

top_n = st.sidebar.number_input('Results to show', min_value=1, max_value=50, value=10)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
mask = (
    df['bodytype'].isin(sel_bodytype) &
    df['make'].isin(sel_make) &
    df['Fuel Type'].isin(sel_fuel) &
    df['Drivetrain'].isin(sel_drive) &
    df['Transmission Type'].isin(sel_trans) &
    (df['price']    >= sel_min_price) &
    (df['price']    <= sel_price) &
    (df['mpg_comb'] >= sel_mpg) &
    (df['mpg_comb'] <= sel_max_mpg) &
    (df['hp']       >= sel_hp) &
    (df['hp']       <= sel_max_hp) &
    (df['year']                      >= sel_year_min) &
    (df['year']                      <= sel_year_max) &
    (df['predicted_consumer_rating'] >= sel_pred_min) &
    (df['predicted_consumer_rating'] <= sel_pred_max)
)
if sel_ev:
    mask &= df['is_ev'] == 1
if sel_luxury:
    mask &= df['luxury'] == 1

sidebar_filtered_df = df[mask]
st.session_state['sidebar_filtered_df'] = sidebar_filtered_df

results = (
    sidebar_filtered_df
    .sort_values('predicted_consumer_rating', ascending=False)
    .drop_duplicates(subset=['make', 'model'])
    .head(top_n)
    .reset_index(drop=True)
)
results.index += 1

# ---------------------------------------------------------------------------
# Tab 1 — Browse
# ---------------------------------------------------------------------------
with tab_browse:
    st.subheader(f'{mask.sum():,} cars match — top {min(top_n, len(results))} unique models shown')

    if results.empty:
        st.warning('No cars match the current filters. Try loosening your criteria.')
    else:
        display_cols = {
            'make':                       'Make',
            'model':                      'Model',
            'year':                       'Year',
            'trim':                       'Trim',
            'price':                      'Price ($)',
            'mpg_comb':                   'MPG',
            'hp':                         'HP',
            'Drivetrain':                 'Drivetrain',
            'Fuel Type':                  'Fuel',
            'consumer_overall_rating':    'Actual Rating',
            'predicted_consumer_rating':  'Predicted Rating',
        }

        out = results[list(display_cols)].rename(columns=display_cols)
        out['Price ($)'] = out['Price ($)'].apply(lambda x: f'${x:,.0f}')

        st.dataframe(
            out.style.background_gradient(
                subset=['Predicted Rating'], cmap='RdYlGn', vmin=3.5, vmax=5.0
            ).format({
                'Predicted Rating': '{:.2f}',
                'Actual Rating':    lambda x: f'{x:.1f}' if pd.notna(x) else '—',
            }),
            use_container_width=True,
            height=min(50 + len(out) * 35, 600),
        )

        st.caption(
            'Actual Rating = aggregated buyer reviews (model-year level). '
            'Predicted Rating = Random Forest estimate from car specs + expert ratings. '
            'Prices shown are new-car MSRP — used market prices are typically 30-60% lower.'
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Avg Predicted Rating', f"{results['predicted_consumer_rating'].mean():.2f}")
        col2.metric('Avg Price',            f"${results['price'].mean():,.0f}")
        col3.metric('Avg MPG',              f"{results['mpg_comb'].mean():.1f}")
        col4.metric('Avg HP',               f"{results['hp'].mean():.0f}")

        st.divider()
        st.subheader('Car Detail')

        pick_options = results.apply(
            lambda r: f"{int(r['year'])} {r['make']} {r['model']} — {r['trim']}", axis=1
        ).tolist()

        picked_label = st.selectbox('Select a car to inspect', options=['—'] + pick_options)

        if picked_label != '—':
            idx    = pick_options.index(picked_label)
            car    = results.iloc[idx]
            car_id = car['car_id']

            full = df[df['car_id'] == car_id].iloc[0]

            st.markdown(f"### {int(full['year'])} {full['make']} {full['model']} — {full['trim']}")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('Price',          f"${full['price']:,.0f}")
            c2.metric('MPG (combined)', f"{full['mpg_comb']:.0f}")
            c3.metric('Horsepower',     f"{full['hp']:.0f}" + (' ~' if full['hp_missing'] else ''))
            c4.metric('Torque (lb-ft)', f"{full['torque_lbft']:.0f}" + (' ~' if full['torque_missing'] else ''))
            c5.metric('Cargo (cu ft)',  f"{full['cargo_cuft']:.1f}" if full['cargo_cuft'] else '—')

            if full['hp_missing'] or full['torque_missing']:
                st.caption('~ = ML-imputed (raw data was missing)')

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('Drivetrain',   full['Drivetrain'])
            c2.metric('Transmission', full['Transmission Type'])
            c3.metric('Fuel Type',    full['Fuel Type'])
            c4.metric('Doors',        f"{int(full['doors'])}")
            c5.metric('Body Type',    full['bodytype'])

            st.divider()

            col_exp, col_con = st.columns(2)

            with col_exp:
                st.markdown('**Expert Ratings**')
                for label, col in [
                    ('Value',       'rating_value'),
                    ('Performance', 'rating_performance'),
                    ('Quality',     'rating_quality'),
                    ('Comfort',     'rating_comfort'),
                    ('Reliability', 'rating_reliability'),
                    ('Styling',     'rating_styling'),
                ]:
                    val = full[col]
                    st.write(f'{label}: **{val:.1f}** / 5.0')

            with col_con:
                st.markdown('**Consumer Rating**')
                actual = full['consumer_overall_rating']
                count  = full['consumer_review_count']
                st.write(f'Overall: **{actual:.1f}** / 5.0')
                st.write(f'Based on {int(count):,} reviews' if count > 0 else 'No consumer reviews')
                st.write(f'Predicted: **{full["predicted_consumer_rating"]:.2f}** / 5.0')

            st.divider()

            FEATURE_COLS = {
                'Safety': [
                    'Child Seat Anchors', 'Child Door Locks', 'Traction Control',
                    'Stability Control', 'Hill Start Assist', 'Blind-Spot Alert',
                    'Collision Warning System',
                ],
                'Connectivity': [
                    'Bluetooth Wireless Technology', 'Hands Free Phone',
                    'Bluetooth Streaming Audio', 'Satellite Radio', 'Smartphone Interface',
                    'Navigation System', 'Voice Recognition System', 'Internet Access',
                    'Real-Time Traffic Information', 'Premium Radio',
                ],
                'Convenience': [
                    'Cruise Control', 'Remote Keyless Entry', 'Remote Engine Start',
                    'Power Windows', 'Power Outlet', 'Rear Window Defroster',
                    'Steering Wheel Controls', 'Tilt Steering Wheel',
                ],
            }
            LABELS = {0: 'Not Available', 1: 'Optional', 2: 'Standard'}

            st.markdown('**Features**')
            fc1, fc2, fc3 = st.columns(3)
            for col, (group, cols) in zip([fc1, fc2, fc3], FEATURE_COLS.items()):
                col.markdown(f'*{group}*')
                for feat in cols:
                    val = int(full[feat]) if pd.notna(full[feat]) else 0
                    col.write(f'{LABELS[val]} — {feat}')

            with st.expander('Show more specs'):
                s1, s2, s3, s4 = st.columns(4)

                s1.markdown('**Fuel Economy**')
                s1.write(f'City:     {full["mpg_city"]:.0f} MPG')
                s1.write(f'Highway:  {full["mpg_hwy"]:.0f} MPG')
                s1.write(f'Combined: {full["mpg_comb"]:.0f} MPG')

                s2.markdown('**Performance**')
                s2.write(f'Horsepower:  {full["hp"]:.0f} hp'
                         + (' *(imputed)*' if full['hp_missing'] else ''))
                s2.write(f'Torque:      {full["torque_lbft"]:.0f} lb-ft'
                         + (' *(imputed)*' if full['torque_missing'] else ''))

                s3.markdown('**Classification**')
                s3.write(f'Electric:  {"Yes" if full["is_ev"] else "No"}')
                s3.write(f'Luxury:    {"Yes" if full["luxury"] else "No"}')
                s3.write(f'Car ID:    {full["car_id"]}')
                if pd.notna(full.get('review_id')):
                    s3.write(f'Review ID: {full["review_id"]}')

                s4.markdown('**Engine**')
                engine = full.get('Engine')
                s4.write(engine if pd.notna(engine) else '—')


# ---------------------------------------------------------------------------
# Tab 2 — Advisor
# ---------------------------------------------------------------------------
_MAX_TURNS      = 20
_MAX_TOOL_LOOPS = 5

with tab_advisor:
    st.caption('Describe what you are looking for in plain language. The advisor will search the catalog and explain the results.')

    # --- Build a human-readable summary of active sidebar filters ---
    def _active_filter_lines() -> list[str]:
        lines = []
        if len(sel_make) < len(makes):
            lines.append(f"Make: {', '.join(sel_make)}")
        if len(sel_bodytype) < len(bodytypes):
            lines.append(f"Body type: {', '.join(sel_bodytype)}")
        if len(sel_fuel) < len(fuel_types):
            lines.append(f"Fuel type: {', '.join(sel_fuel)}")
        if len(sel_drive) < len(drivetrains):
            lines.append(f"Drivetrain: {', '.join(sel_drive)}")
        if len(sel_trans) < len(transmissions):
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
    sidebar_count = mask.sum()
    total_count   = len(df)

    if active_filter_lines:
        st.info(
            f'Sidebar filters active ({sidebar_count:,} of {total_count:,} cars) — '
            f'the advisor will only recommend from this filtered pool. '
            f'Clear filters in the sidebar to search the full catalog.'
        )

    # --- Dynamic system prompt: tell Claude exactly what is pre-filtered ---
    if active_filter_lines:
        _filter_context = (
            "\n\nThe user has already pre-set the following sidebar filters. "
            "Your tool is already restricted to cars matching these criteria — "
            "do NOT re-apply them as tool parameters (that would over-constrain the results). "
            "Instead, use the tool with only the ADDITIONAL filters the user mentions in chat. "
            "Always acknowledge the active sidebar filters when presenting recommendations "
            "so the user understands the scope.\n\nActive sidebar filters:\n"
            + "\n".join(f"  - {line}" for line in active_filter_lines)
        )
        _dynamic_system = ADVISOR_SYSTEM + _filter_context
    else:
        _dynamic_system = ADVISOR_SYSTEM

    if 'advisor_api_messages' not in st.session_state:
        st.session_state.advisor_api_messages = []
        st.session_state.advisor_display      = []

    for role, text in st.session_state.advisor_display:
        with st.chat_message(role):
            st.markdown(text)

    user_input = st.chat_input('Tell me what you are looking for...')

    if user_input:
        user_input = _sanitize(user_input)

        if not user_input:
            st.warning('Please enter a valid message.')
        elif len(st.session_state.advisor_display) >= _MAX_TURNS * 2:
            st.warning('Conversation limit reached. Please clear and start a new one.')
        elif not _within_rate_limit():
            st.warning('Too many requests. Please wait a moment before asking again.')
        else:
            api_key = _load_secret('ANTHROPIC_API_KEY')
            if not api_key or not api_key.startswith('sk-'):
                st.error('API key not configured. Add ANTHROPIC_API_KEY to your .env file.')
            else:
                with st.chat_message('user'):
                    st.markdown(user_input)
                st.session_state.advisor_display.append(('user', user_input))
                st.session_state.advisor_api_messages.append({'role': 'user', 'content': user_input})
                _log.info('User message received (len=%d)', len(user_input))

                try:
                    import anthropic as _anthropic
                    client   = _anthropic.Anthropic(api_key=api_key, max_retries=3)
                    api_msgs = list(st.session_state.advisor_api_messages)

                    with st.spinner('Searching catalog...'):
                        for _ in range(_MAX_TOOL_LOOPS):
                            response = client.messages.create(
                                model='claude-haiku-4-5',
                                max_tokens=2048,
                                system=_dynamic_system,
                                tools=ADVISOR_TOOLS,
                                messages=api_msgs,
                            )
                            api_msgs.append({'role': 'assistant', 'content': response.content})

                            if response.stop_reason != 'tool_use':
                                break

                            tool_results = []
                            for block in response.content:
                                if block.type == 'tool_use':
                                    result = _execute_tool(block.name, dict(block.input))
                                    tool_results.append({
                                        'type':        'tool_result',
                                        'tool_use_id': block.id,
                                        'content':     result,
                                    })
                            api_msgs.append({'role': 'user', 'content': tool_results})

                    reply = next((b.text for b in response.content if b.type == 'text'), '')
                    st.session_state.advisor_api_messages = api_msgs
                    _log.info('Advisor reply sent (len=%d)', len(reply))

                    with st.chat_message('assistant'):
                        st.markdown(reply)
                    st.session_state.advisor_display.append(('assistant', reply))

                except _anthropic.RateLimitError:
                    st.error('Rate limit reached. Please wait a moment and try again.')
                    _log.warning('Anthropic rate limit hit.')
                except _anthropic.APIStatusError as e:
                    if e.status_code >= 500:
                        st.error('The advisor service is temporarily unavailable. Please try again.')
                    else:
                        st.error('The advisor could not process your request. Please try again.')
                    _log.error('APIStatusError status=%d', e.status_code)
                except Exception:
                    st.error('An unexpected error occurred. Please try again.')
                    _log.exception('Unexpected advisor error.')

    if st.session_state.get('advisor_display'):
        if st.button('Clear conversation'):
            st.session_state.advisor_api_messages = []
            st.session_state.advisor_display      = []
            st.rerun()
