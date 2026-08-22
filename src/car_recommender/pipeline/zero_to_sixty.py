"""
zero_to_sixty.py  –  Physics-based 0-60 mph estimator for vehicles with hp > 180

Formula
-------
    Time = max(a × X, b + c + d) × e × f × g × l + h

    X   = (½ × mass_kg × v²) / power_watts     ← minimum theoretical 0-60 (KE/power)
    a   = secondary-effects multiplier           ← calibrated per BODYTYPE GROUP from data
    b   = drivetrain absolute grip threshold     ← traction-limited minimum seconds
    c   = tire-type absolute grip threshold      ← from VEHICLE-CLASS tire table
    d   = road-condition absolute grip penalty   ← from ROAD CONDITIONS table
    e   = engine-type proportional factor        ← empirically derived from 7 k+ observed times
    f   = tire-type proportional factor          ← from VEHICLE-CLASS tire table
    g   = road-condition proportional factor     ← from ROAD CONDITIONS table
    l   = segment launch-management factor       ← luxury / EV / body-class table
    h   = gearbox shift-time absolute penalty    ← cumulative seconds lost to gear changes


Key improvements over v1
------------------------
    • a calibrated separately per bodytype group (Sports/Sedan/Hatch/Wagon/SUV/Truck/Van)
    • c and f inferred from vehicle class (bodytype + premium tier), not locked to zero
    • l factor captures luxury-brand launch control, EV instant-torque tuning,
      and penalty for body types that historically under-perform their power-to-weight
    • Road-condition lookup table (d, g) lets users re-run for non-ideal scenarios
    • SUSPECT_HP_OR_WEIGHT flag on rows where |est − obs| > 2 s

Ideal test environment (used by default)
-----------------------------------------
    Surface  : Flat, sealed dry-tarmac drag strip  (μ ≈ 1.0)
    Weather  : Calm, 20 °C, sea-level
    Tires    : OEM standard fitment (inferred by vehicle class)
    Driver   : Professional launch technique, engine at operating temp

Usage
-----
    python -m car_recommender.pipeline.zero_to_sixty

Output
------
    data/final/cars/zero_to_sixty_est.csv   (22 columns, hp > 180 rows only)
"""

import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ── Paths ────────────────────────────────────────────────────────────────────
from car_recommender.core.paths import PROJECT_ROOT

ROOT  = PROJECT_ROOT
STATS = ROOT / "data/final/cars/car_stats.csv"
FEATS = ROOT / "data/final/cars/car_features.csv"
OUT   = ROOT / "data/final/cars/zero_to_sixty_est.csv"

# ── Physical constants ────────────────────────────────────────────────────────
V_60MPH_MPS = 60.0 * 0.44704          # 26.8224 m/s
LBS_TO_KG   = 0.453592
HP_TO_WATTS = 745.7
KE_COEFF    = 0.5 * V_60MPH_MPS**2    # 359.720 J/kg

# ── Data-quality outlier bounds (null-out before imputation) ─────────────────
WEIGHT_MIN_LBS = 1_500.0
WEIGHT_MAX_LBS = 10_000.0
HP_MAX_LBS     = 1_500.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FORMULA PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── BODYTYPE GROUPING ─────────────────────────────────────────────────────────
# Used to select the right calibrated `a`, tire params, and segment factor.
BODYTYPE_GROUP = {
    "Coupe":       "Sports",
    "Convertible": "Sports",
    "Sedan":       "Sedan",
    "Hatchback":   "Hatch",
    "Wagon":       "Wagon",
    "SUV":         "SUV",
    "Pickup":      "Truck",
    "Minivan":     "Van",
    "Van":         "Van",
}

# ── SEGMENT CLASSIFICATION ────────────────────────────────────────────────────
# "Premium" = luxury brand OR EV/PHEV fuel type.
# This drives both the tire lookup and the l factor.
LUXURY_MAKES = {
    "Acura", "Alfa-Romeo", "Audi", "Bentley", "BMW", "Bugatti Rimac",
    "Cadillac", "Ferrari", "Genesis", "Infiniti", "Jaguar", "Karma",
    "Koenigsegg", "Lamborghini", "Land-Rover", "Lexus", "Lincoln",
    "Lucid", "Maserati", "McLaren Automotive", "Mercedes-Benz",
    "Polestar", "Porsche", "Rivian", "Rolls-Royce", "Tesla", "Volvo",
}
EV_FUELS = {"Electric", "Plug-in Hybrid"}

# ── a: BODYTYPE-GROUP SECONDARY-EFFECTS MULTIPLIER ───────────────────────────
# Calibrated below via scipy from observed 0-60 data for each group.
# Captures: drivetrain losses, rolling resistance, aerodynamic drag during
# acceleration, and the average shape of each class's power-delivery curve.
# Physics meaning: effective_time ≈ a × X_theoretical × engine_penalty
#
# Sports (a ≈ 1.878): lower CG, better aero, shorter gear ratios → less loss
# Hatch  (a ≈ 1.733): hot-hatch tuning; calibrated on performance hatchbacks
# Sedan  (a ≈ 1.948): general passenger reference
# SUV    (a ≈ 1.920): slightly better than sedan due to larger contact patch
# Wagon  (a ≈ 1.934): close to sedan
# Truck/Van: use global fallback (too few clean calibration points)
A_BY_GROUP = {}   # populated at runtime by calibrate_a_by_group()
A_GLOBAL   = 1.916  # pre-computed fallback

# ── b: DRIVETRAIN ABSOLUTE GRIP-THRESHOLD PENALTY (seconds) ──────────────────
# Represents the traction-limited LOWER BOUND on the base time regardless
# of how much power the vehicle has.  Applies through the max() function.
#
#  AWD: best multi-axle traction; both front and rear axles resist spin → 0.0 s
#  4WD: mechanical transfer case; slight delay activating all wheels → 0.20 s
#  FWD: torque steer + all braking/traction on front axle → 0.35 s
#  RWD: rear-wheel oversteer/wheelspin under hard launch → 0.45 s
#  2WD: dataset label meaning unspecified two-driven wheels → same as FWD
B_DRIVETRAIN = {
    "AWD": 0.00,
    "4WD": 0.20,
    "FWD": 0.35,
    "RWD": 0.45,
    "2WD": 0.35,
}

# ── ROAD CONDITIONS TABLE (d, g) ─────────────────────────────────────────────
# Universal lookup: choose a scenario for d and g.
# d (absolute grip threshold addition, seconds)
# g (proportional time stretch due to reduced surface friction)
#
# Physics: wet roads lower the friction coefficient μ, which both raises the
# minimum traction-limited time (d > 0) and proportionally extends the whole
# run because the tyres cannot put power down as efficiently (g > 1).
ROAD_CONDITIONS = {
    #  key            label                              d      g
    "ideal_test":  ("Dry flat sealed tarmac, drag strip", 0.00, 1.00),
    "good_road":   ("Dry public road, slight gradient",   0.05, 1.04),
    "damp":        ("Damp surface, light moisture",        0.15, 1.09),
    "wet":         ("Wet road, standing water patches",    0.35, 1.18),
    "cold_dry":    ("Cold dry (<5 °C), hard tires",       0.20, 1.10),
    "cold_wet":    ("Cold wet surface",                    0.55, 1.28),
    "gravel":      ("Loose gravel / dirt",                 0.70, 1.45),
}
ROAD_SCENARIO = "ideal_test"   # ← change this to simulate different conditions

# ── TIRE TABLE (c, f) by (bodytype_group, is_premium) ────────────────────────
# c (absolute grip threshold, seconds): additional traction-limited floor from
#    tyre compound / construction; UHP tyres ≈ 0.0 s, all-terrain ≈ 0.35 s
# f (proportional time factor): rolling-resistance + grip-efficiency penalty;
#    UHP = 1.00 (reference), load-rated all-terrain truck tyres ≈ 1.10
#
# Since individual tyre data is absent, we infer from what class of tyre
# each vehicle segment is typically factory-fitted with:
#
#  Sports Premium   → Ultra-high-performance summer (Michelin PS4S, Pirelli P Zero)
#  Sports Non-Prem  → Performance all-season or basic summer
#  Sedan Premium    → Run-flat touring (BMW standard) or performance all-season
#  Sedan Non-Prem   → Standard all-season (Michelin Defender, Continental TrueContact)
#  Hatch Premium    → All-season performance or touring
#  Hatch Non-Prem   → Standard all-season
#  Wagon Premium    → All-season performance
#  Wagon Non-Prem   → Standard all-season
#  SUV Premium      → All-season performance (larger, higher rolling resistance)
#  SUV Non-Prem     → Standard all-season or all-terrain
#  Truck Premium    → Load-rated all-terrain (heavy carcass, high rolling resist.)
#  Truck Non-Prem   → All-terrain / load-rated HD
#  Van              → Standard touring / load-rated
#  Unknown          → Default to sedan equivalent
#
#  key: (bodytype_group, is_premium)  →  (c_seconds, f_factor)
TIRE_PARAMS = {
    ("Sports", True):   (0.00, 1.00),  # UHP summer: sticky, zero extra threshold
    ("Sports", False):  (0.10, 1.02),  # Performance all-season: minor grip lag

    ("Sedan",  True):   (0.15, 1.03),  # Run-flat touring: stiffer sidewall, more RR
    ("Sedan",  False):  (0.20, 1.04),  # Standard all-season

    ("Hatch",  True):   (0.12, 1.03),  # All-season performance
    ("Hatch",  False):  (0.20, 1.04),  # Standard all-season

    ("Wagon",  True):   (0.12, 1.03),  # All-season performance (Audi A6 Avant, etc.)
    ("Wagon",  False):  (0.20, 1.04),  # Standard all-season

    ("SUV",    True):   (0.18, 1.05),  # All-season performance (larger diameter)
    ("SUV",    False):  (0.28, 1.07),  # Standard / light all-terrain

    ("Truck",  True):   (0.28, 1.08),  # Load-rated all-terrain (luxury trucks)
    ("Truck",  False):  (0.35, 1.10),  # Heavy all-terrain / load-rated HD

    ("Van",    True):   (0.25, 1.07),  # Touring / commercial
    ("Van",    False):  (0.30, 1.08),  # Standard commercial / load-rated

    ("Unknown",True):   (0.15, 1.03),  # Default to sedan premium
    ("Unknown",False):  (0.20, 1.04),  # Default to sedan standard
}

# ── e: ENGINE-TYPE PROPORTIONAL FACTOR ───────────────────────────────────────
# Derived empirically from ratio of (actual_t60 − h) / (a × X) per engine type,
# after fixing a to the Gas_NA reference calibration.
#
# Values < 1.0: engine delivers more EFFECTIVE torque relative to its peak-HP
#               rating than a Gas_NA engine (better low-RPM force, instant torque).
# Values > 1.0: engine delivers LESS effectively relative to peak HP (power curve
#               peaks late, parasitic losses, or lower energy-delivery rate).
#
# Key findings:
#  Diesel (NA/Turbo): e=0.86 — flat torque curve peaks at very low RPM; the entire
#    0-60 run is powered near peak torque regardless of RPM.  Peak HP understates
#    average force delivered.
#  Electric: e=0.88 — full torque at 0 RPM; only battery current limits delivery.
#  Gas_Turbo: e=0.97 — modern twin-scroll / e-turbo spools in <0.5 s; effective
#    lag over a 5–7 s run is small; rated HP nearly reflects full-run average.
#  Gas_NA: e=1.00 — reference; power only peaks near redline, so average effective
#    power over the run is lower than the rated peak.
#  Gas_SC: e=1.11 — instant boost but parasitic belt-drive loss reduces net wheel
#    power below what the HP rating suggests.
E_ENGINE = {
    "Diesel_NA":       0.86,
    "Diesel_Turbo":    0.86,
    "Diesel_SC":       0.86,
    "Electric":        0.88,
    "Plug-in Hybrid":  0.90,
    "Hybrid":          0.93,
    "Gas_Turbo":       0.97,
    "Gas_NA":          1.00,  # ← calibration reference
    "Natural Gas":     1.04,
    "Flex-Fuel_Turbo": 1.06,
    "Flex-Fuel_NA":    1.08,
    "Gas_SC":          1.11,
}

# ── l: SEGMENT LAUNCH-MANAGEMENT FACTOR ──────────────────────────────────────
# Captures the SYSTEM-LEVEL effect of launch sophistication on actual elapsed
# time, INDEPENDENT of engine type (already in e) and tyre choice (already in c/f).
#
# Premium (luxury brand or EV):
#   Advantages: electronic launch control minimises wheelspin, active torque
#     vectoring maximises grip utilisation, adaptive gearbox maps optimise shifts.
#   Disadvantages: luxury features add weight (run-flat spares, heavy glass,
#     soundproofing) not always reflected in curb_weight imputation; comfort-biased
#     suspension tune may sacrifice launch grip.
#
# Key calibration findings (empirical median of actual/predicted per cell):
#   Sports+Premium  = 1.069 → luxury sports cars are ~7% slower than raw hp/weight
#     implies; heavier body kits, run-flat tyres, and comfort suspension outweigh
#     launch-control benefits for many BMWs / Audis tested at moderate power levels
#   Hatch+Premium   = 1.115 → Mini Cooper S and similar: FWD, torque steer, heavier
#     than hot-hatch baseline → systematically outperforms only moderate
#   Wagon+Premium   = 0.988 → premium wagons (Audi RS6, Volvo V90) punch above weight
#   SUV+Premium     = 1.004 → launch control nearly neutralises weight penalty → ≈1.00
#   Sedan+Premium   = 1.001 → same; AMG / M launch control essentially optimal
#   Van+Non-Prem    = 1.150 → heavy, FWD, zero performance tuning → time penalty
#
# Values are rounded from empirical medians and anchored by physical reasoning.
# key: (bodytype_group, is_premium)  →  l
SEGMENT_FACTOR = {
    ("Sports",  True):  1.06,   # heavy luxury sports + run-flat tires > launch control benefit
    ("Sports",  False): 1.00,   # sporty non-luxury: driver-dependent, reference

    ("Sedan",   True):  1.00,   # AMG / M launch control = well-calibrated; empirical ≈ 1.001
    ("Sedan",   False): 1.01,   # minor torque management gap vs luxury; empirical ≈ 1.013

    ("Hatch",   True):  1.10,   # FWD torque steer + comfort bias + run-flats; empirical ≈ 1.115
    ("Hatch",   False): 1.04,   # non-luxury hatch: FWD penalty partly absorbed by b

    ("Wagon",   True):  0.98,   # RS6 / V90 Polestar: punches above weight; empirical ≈ 0.988
    ("Wagon",   False): 1.01,

    ("SUV",     True):  1.00,   # Cayenne / Model X launch control → neutral; empirical ≈ 1.004
    ("SUV",     False): 1.02,   # standard SUV: slightly worse launch efficiency

    ("Truck",   True):  1.08,   # heavy luxury pickup despite launch software; empirical ≈ 1.126
    ("Truck",   False): 1.00,   # diesel trucks well-calibrated by e factor; empirical ≈ 0.983

    ("Van",     True):  1.05,   # premium van (rare)
    ("Van",     False): 1.12,   # heavy, FWD, no performance intent; empirical ≈ 1.150

    ("Unknown", True):  1.01,
    ("Unknown", False): 1.01,
}

# ── h: GEARBOX SHIFT-TIME ABSOLUTE PENALTY (seconds) ─────────────────────────
# Cumulative lost time from 2-3 gear changes during a typical 0-60 run.
# DCT shifts are near-seamless; torque-converter automatics pause for torque
# converter lock-up; manuals require driver input + clutch engagement.
H_GEARBOX = {
    "DCT":       0.10,   # dual-clutch: electro-hydraulic actuation ≈ 50 ms/shift
    "Automatic": 0.25,   # torque-converter + shift pause ≈ 100-150 ms × 2 shifts
    "Manual":    0.40,   # driver reaction (200 ms) + clutch (200 ms) × ~2 shifts
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLASSIFIERS & HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_TURBO_RE = re.compile(
    r"turbo|twin.?turbo|ecoboost|powerstroke|duramax|cummins|biturbo|tdi|hdi|crd|bluemotion",
    re.IGNORECASE,
)
_SC_RE  = re.compile(r"supercharg|kompressor|s/c",    re.IGNORECASE)
_DSL_RE = re.compile(r"diesel|dsl|tdi|hdi|crd|bluemotion", re.IGNORECASE)


def classify_engine(fuel: str, engine_str: str) -> str:
    fuel = str(fuel).strip()
    eng  = str(engine_str).strip()
    if fuel == "Electric":       return "Electric"
    if fuel == "Plug-in Hybrid": return "Plug-in Hybrid"
    if fuel == "Hybrid":         return "Hybrid"
    if fuel == "Natural Gas":    return "Natural Gas"
    t  = bool(_TURBO_RE.search(eng))
    sc = bool(_SC_RE.search(eng))
    d  = bool(_DSL_RE.search(eng)) or fuel == "Diesel"
    if d:
        return "Diesel_SC" if sc else ("Diesel_Turbo" if t else "Diesel_NA")
    if fuel == "Flex-Fuel":
        return "Flex-Fuel_Turbo" if (t or sc) else "Flex-Fuel_NA"
    return "Gas_SC" if sc else ("Gas_Turbo" if t else "Gas_NA")


def classify_gearbox(trans: str, dct_flag) -> str:
    if pd.notna(dct_flag) and float(dct_flag) >= 2.0:
        return "DCT"
    return "Manual" if str(trans).strip() == "Manual" else "Automatic"


def impute_by_groups(df: pd.DataFrame, col: str, keys: list, agg: str = "mean") -> pd.Series:
    """Fill NaN via group aggregation, cascading through smaller key subsets."""
    result = df[col].copy()
    for width in range(len(keys), 0, -1):
        for combo in combinations(keys, width):
            grp = list(combo)
            if agg == "mean":
                fills = df.groupby(grp, dropna=False)[col].transform("mean")
            else:
                def _mode(s):
                    m = s.mode(dropna=True)
                    return m.iloc[0] if len(m) else np.nan
                fills = df.groupby(grp, dropna=False)[col].transform(_mode)
            mask = result.isna() & fills.notna()
            result[mask] = fills[mask]
            if result.isna().sum() == 0:
                return result
    result.fillna(df[col].mean() if agg == "mean" else df[col].mode().iloc[0], inplace=True)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FORMULA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_x(mass_kg: np.ndarray, power_w: np.ndarray) -> np.ndarray:
    """Minimum theoretical 0-60 s: KE / peak power."""
    return (KE_COEFF * mass_kg) / power_w


def apply_formula(
    a:   np.ndarray,   # per-car a values (from bodytype group)
    X:   np.ndarray,
    b:   np.ndarray,
    c:   np.ndarray,
    d:   float,
    e:   np.ndarray,
    f:   np.ndarray,
    g:   float,
    l:   np.ndarray,
    h:   np.ndarray,
) -> np.ndarray:
    """Full 10-parameter formula."""
    base = np.maximum(a * X, b + c + d)
    return base * e * f * g * l + h


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CALIBRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calibrate_a_by_group(df: pd.DataFrame) -> dict:
    """
    Calibrate `a` independently per bodytype group.

    Calibration subset: Gas or Flex-Fuel (ICE reference), Automatic transmission,
    known observed 0-60, reasonable range [2, 14] s.
    Includes ALL drivetrains (not just AWD) so the b term is active.
    l is set to 1.0 during calibration (absorbed into a for this pass);
    the separate l table corrects residuals afterwards.
    """
    cal = df[
        df["Fuel Type"].isin(["Gas", "Flex-Fuel"])
        & (df["gearbox"] == "Automatic")
        & df["0 - 60"].notna()
        & df["0 - 60"].between(2.0, 14.0)
        & df["X"].notna()
        & df["e_val"].notna()
    ].copy()

    results = {}
    groups  = ["Sports", "Sedan", "Hatch", "Wagon", "SUV", "Truck", "Van", "Unknown"]
    MIN_CAL = 40   # minimum calibration rows per group

    # Compute global a first (used as fallback)
    X_g, b_g, c_g, e_g, h_g, y_g = (
        cal["X"].values, cal["b"].values, cal["c"].values,
        cal["e_val"].values, cal["h_val"].values, cal["0 - 60"].values,
    )
    def rmse_global(a):
        pred = np.maximum(a * X_g, b_g + c_g) * e_g + h_g
        return np.sqrt(np.mean((pred - y_g) ** 2))
    global_a = minimize_scalar(rmse_global, bounds=(1.0, 4.0), method="bounded").x

    print(f"\n  Global a (fallback for small groups) = {global_a:.4f}")

    for grp in groups:
        sub = cal[cal["bt_group"] == grp]
        if len(sub) < MIN_CAL:
            results[grp] = global_a
            print(f"  {grp:10s}: n={len(sub):4d}  → fallback  a={global_a:.4f}")
            continue
        X_s, b_s, c_s, e_s, h_s, y_s = (
            sub["X"].values, sub["b"].values, sub["c"].values,
            sub["e_val"].values, sub["h_val"].values, sub["0 - 60"].values,
        )
        def rmse_grp(a, X=X_s, b=b_s, c=c_s, e=e_s, h=h_s, y=y_s):
            pred = np.maximum(a * X, b + c) * e + h
            return np.sqrt(np.mean((pred - y) ** 2))
        a_grp = minimize_scalar(rmse_grp, bounds=(1.0, 4.0), method="bounded").x
        results[grp] = a_grp
        print(f"  {grp:10s}: n={len(sub):4d}  a={a_grp:.4f}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 66)
    print("  0-60 mph Estimator  –  hp > 180  |  full 10-parameter formula")
    print("=" * 66)

    # ── 1. Road conditions ────────────────────────────────────────────────────
    label, d_road, g_road = ROAD_CONDITIONS[ROAD_SCENARIO]
    print(f"\nRoad scenario  : '{ROAD_SCENARIO}'  →  {label}")
    print(f"  d={d_road} s  g={g_road}")

    # ── 2. Load ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
    df   = pd.read_csv(STATS)
    feat = pd.read_csv(FEATS, usecols=["car_id", "Dual-Clutch Automatic Transmission"])
    df   = df.merge(feat, on="car_id", how="left")
    print(f"      {len(df):,} total rows")

    # ── 3. Outlier cleaning ───────────────────────────────────────────────────
    print("\n[2/7] Cleaning data-entry outliers …")
    bad_wt = df["curb_weight_lbs"].notna() & (
        (df["curb_weight_lbs"] < WEIGHT_MIN_LBS) | (df["curb_weight_lbs"] > WEIGHT_MAX_LBS)
    )
    bad_hp = df["hp"].notna() & (df["hp"] > HP_MAX_LBS)
    print(f"      curb_weight outliers nulled: {bad_wt.sum()}  "
          f"(< {WEIGHT_MIN_LBS:.0f} or > {WEIGHT_MAX_LBS:.0f} lbs)")
    print(f"      hp outliers nulled         : {bad_hp.sum()}  "
          f"(> {HP_MAX_LBS:.0f} hp)")
    df.loc[bad_wt, "curb_weight_lbs"] = np.nan
    df.loc[bad_hp, "hp"]              = np.nan

    # ── 4. Impute ─────────────────────────────────────────────────────────────
    print("\n[3/7] Imputing missing inputs …")
    for col, keys, agg in [
        ("hp",               ["make", "bodytype", "year"], "mean"),
        ("curb_weight_lbs",  ["make", "bodytype", "year"], "mean"),
        ("Drivetrain",       ["make", "bodytype"],          "mode"),
        ("Transmission Type",["make", "bodytype"],          "mode"),
    ]:
        before = df[col].isna().sum()
        df[col] = impute_by_groups(df, col, keys, agg)
        print(f"      {col:25s}  null {before:5,} → {df[col].isna().sum():5,}")

    # ── 5. Feature engineering ────────────────────────────────────────────────
    print("\n[4/7] Engineering formula inputs …")

    # Classification
    df["engine_type"] = [
        classify_engine(f, e)
        for f, e in zip(df["Fuel Type"], df["Engine"].fillna(""))
    ]
    df["gearbox"] = [
        classify_gearbox(t, d)
        for t, d in zip(
            df["Transmission Type"].fillna("Automatic"),
            df["Dual-Clutch Automatic Transmission"],
        )
    ]
    df["bt_group"]   = df["bodytype"].map(BODYTYPE_GROUP).fillna("Unknown")
    df["is_luxury"]  = df["make"].isin(LUXURY_MAKES)
    df["is_ev"]      = df["Fuel Type"].isin(EV_FUELS)
    df["is_premium"] = df["is_luxury"] | df["is_ev"]

    # Drivetrain b
    df["b"] = df["Drivetrain"].map(B_DRIVETRAIN).fillna(B_DRIVETRAIN["FWD"])

    # Tire c and f  (bodytype group × premium tier)
    tire_c = df.apply(
        lambda r: TIRE_PARAMS.get((r["bt_group"], r["is_premium"]),
                                   TIRE_PARAMS[("Unknown", r["is_premium"])])[0],
        axis=1,
    )
    tire_f = df.apply(
        lambda r: TIRE_PARAMS.get((r["bt_group"], r["is_premium"]),
                                   TIRE_PARAMS[("Unknown", r["is_premium"])])[1],
        axis=1,
    )
    df["c"], df["f"] = tire_c, tire_f

    # Engine e, gearbox h, segment l
    df["e_val"] = df["engine_type"].map(E_ENGINE).fillna(1.0)
    df["h_val"] = df["gearbox"].map(H_GEARBOX).fillna(H_GEARBOX["Automatic"])
    df["l_val"] = df.apply(
        lambda r: SEGMENT_FACTOR.get(
            (r["bt_group"], r["is_premium"]),
            SEGMENT_FACTOR[("Unknown", r["is_premium"])],
        ),
        axis=1,
    )

    # X
    df["mass_kg"] = df["curb_weight_lbs"] * LBS_TO_KG
    df["power_w"] = df["hp"] * HP_TO_WATTS
    df["X"]       = compute_x(df["mass_kg"].values, df["power_w"].values)

    print(f"      Bodytype groups : {df['bt_group'].value_counts().to_dict()}")
    print(f"      Premium tier    : {df['is_premium'].value_counts().to_dict()}")
    print(f"      Engine types    : {df['engine_type'].value_counts().to_dict()}")
    print(f"      Gearbox types   : {df['gearbox'].value_counts().to_dict()}")

    # ── 6. Calibrate a per bodytype group ─────────────────────────────────────
    print("\n[5/7] Calibrating `a` per bodytype group …")
    a_map = calibrate_a_by_group(df)
    df["a_val"] = df["bt_group"].map(a_map)

    # ── 7. Apply formula ──────────────────────────────────────────────────────
    print("\n[6/7] Applying formula …")
    ok = df[["X", "b", "c", "e_val", "f", "l_val", "h_val", "a_val"]].notna().all(axis=1)
    df["t60_est"] = np.nan
    df.loc[ok, "t60_est"] = apply_formula(
        a=df.loc[ok, "a_val"].values,
        X=df.loc[ok, "X"].values,
        b=df.loc[ok, "b"].values,
        c=df.loc[ok, "c"].values,
        d=d_road,
        e=df.loc[ok, "e_val"].values,
        f=df.loc[ok, "f"].values,
        g=g_road,
        l=df.loc[ok, "l_val"].values,
        h=df.loc[ok, "h_val"].values,
    ).round(2)

    # Filter hp > 180
    high_hp = df[df["hp"] > 180].copy()
    print(f"      hp > 180 rows    : {len(high_hp):,}")
    print(f"      Estimates ready  : {high_hp['t60_est'].notna().sum():,}")

    # Data quality flag
    high_hp["data_quality_flag"] = ""
    both     = high_hp["0 - 60"].notna() & high_hp["t60_est"].notna()
    suspect  = both & ((high_hp["t60_est"] - high_hp["0 - 60"]).abs() > 2.0)
    high_hp.loc[suspect, "data_quality_flag"] = "SUSPECT_HP_OR_WEIGHT"
    print(f"      SUSPECT rows     : {suspect.sum()} "
          f"(|est − obs| > 2 s; likely imputation artifact)")

    # ── 8. Validation ─────────────────────────────────────────────────────────
    print("\n[6b] Validation (hp > 180, known 0-60) …")
    val      = high_hp[high_hp["0 - 60"].notna() & high_hp["t60_est"].notna()].copy()
    res      = val["t60_est"] - val["0 - 60"]
    mae      = res.abs().mean()
    rmse_val = np.sqrt((res**2).mean())
    bias     = res.mean()
    w05      = (res.abs() <= 0.5).mean() * 100
    w10      = (res.abs() <= 1.0).mean() * 100

    print(f"      Rows         : {len(val):,}")
    print(f"      RMSE         : {rmse_val:.3f} s")
    print(f"      MAE          : {mae:.3f} s")
    print(f"      Bias         : {bias:+.3f} s  (positive = predicts too slow)")
    print(f"      Within ±0.5s : {w05:.1f}%")
    print(f"      Within ±1.0s : {w10:.1f}%")

    print("\n      ─ by bodytype group ─")
    bt_v = val.groupby("bt_group").apply(
        lambda g: pd.Series({
            "n":     len(g),
            "RMSE":  round(np.sqrt(((g["t60_est"] - g["0 - 60"])**2).mean()), 3),
            "MAE":   round((g["t60_est"] - g["0 - 60"]).abs().mean(), 3),
            "bias":  round((g["t60_est"] - g["0 - 60"]).mean(), 3),
            "±0.5s": round(((g["t60_est"] - g["0 - 60"]).abs() <= 0.5).mean() * 100, 1),
            "±1.0s": round(((g["t60_est"] - g["0 - 60"]).abs() <= 1.0).mean() * 100, 1),
        }),
        include_groups=False,
    ).sort_values("RMSE")
    print(bt_v.to_string())

    print("\n      ─ by drivetrain ─")
    drv_v = val.groupby("Drivetrain").apply(
        lambda g: pd.Series({
            "n":     len(g),
            "RMSE":  round(np.sqrt(((g["t60_est"] - g["0 - 60"])**2).mean()), 3),
            "bias":  round((g["t60_est"] - g["0 - 60"]).mean(), 3),
            "±0.5s": round(((g["t60_est"] - g["0 - 60"]).abs() <= 0.5).mean() * 100, 1),
        }),
        include_groups=False,
    ).sort_values("RMSE")
    print(drv_v.to_string())

    print("\n      ─ by engine type ─")
    eng_v = val.groupby("engine_type").apply(
        lambda g: pd.Series({
            "n":     len(g),
            "RMSE":  round(np.sqrt(((g["t60_est"] - g["0 - 60"])**2).mean()), 3),
            "bias":  round((g["t60_est"] - g["0 - 60"]).mean(), 3),
            "±0.5s": round(((g["t60_est"] - g["0 - 60"]).abs() <= 0.5).mean() * 100, 1),
        }),
        include_groups=False,
    ).sort_values("RMSE")
    print(eng_v.to_string())

    print("\n      ─ by is_premium ─")
    prem_v = val.groupby("is_premium").apply(
        lambda g: pd.Series({
            "n":     len(g),
            "RMSE":  round(np.sqrt(((g["t60_est"] - g["0 - 60"])**2).mean()), 3),
            "bias":  round((g["t60_est"] - g["0 - 60"]).mean(), 3),
            "±0.5s": round(((g["t60_est"] - g["0 - 60"]).abs() <= 0.5).mean() * 100, 1),
            "±1.0s": round(((g["t60_est"] - g["0 - 60"]).abs() <= 1.0).mean() * 100, 1),
        }),
        include_groups=False,
    )
    prem_v.index = ["Non-Premium", "Premium"]
    print(prem_v.to_string())

    # ── 9. Save ───────────────────────────────────────────────────────────────
    print("\n[7/7] Writing output …")
    keep = [
        "car_id", "make", "model", "year", "trim", "bodytype", "bt_group",
        "hp", "curb_weight_lbs", "Fuel Type", "Engine",
        "Drivetrain", "Transmission Type",
        "engine_type", "gearbox", "is_premium",
        "a_val", "X", "b", "c", "e_val", "f", "l_val", "h_val",
        "t60_est", "0 - 60", "price", "data_quality_flag",
    ]
    out = (
        high_hp[keep]
        .rename(columns={
            "a_val":            "a_bodytype",
            "X":                "X_theoretical_s",
            "b":                "b_drivetrain_s",
            "c":                "c_tire_s",
            "e_val":            "e_engine",
            "f":                "f_tire",
            "l_val":            "l_segment",
            "h_val":            "h_gearbox_s",
            "t60_est":          "t60_estimated_s",
            "0 - 60":           "t60_observed_s",
            "Fuel Type":        "fuel_type",
            "Transmission Type":"transmission",
        })
        .sort_values("t60_estimated_s", na_position="last")
        .reset_index(drop=True)
    )
    out.to_csv(OUT, index=False)
    print(f"      Saved → {OUT.relative_to(ROOT)}  ({len(out):,} rows)")
    print(f"      Road scenario applied: '{ROAD_SCENARIO}'  (d={d_road}, g={g_road})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Distribution of t60_estimated_s (hp > 180) ──────────────────")
    print(out["t60_estimated_s"].describe().round(2).to_string())

    print("\n── Top 20 fastest estimated (SUSPECT rows excluded) ─────────────")
    clean = out[out["data_quality_flag"] != "SUSPECT_HP_OR_WEIGHT"]
    top20 = clean[["make", "model", "year", "trim", "hp", "bt_group",
                    "is_premium", "Drivetrain", "t60_estimated_s",
                    "t60_observed_s"]].head(20)
    print(top20.to_string(index=False))

    print("\n── Formula parameters (ALL 10) ─────────────────────────────────")
    print(f"  a  (bodytype group)          : {a_map}")
    print(f"  b  (drivetrain grip, s)      : {B_DRIVETRAIN}")
    print("  c  (tire grip, s)            : from TIRE_PARAMS[bt_group, is_premium]")
    print(f"  d  (road grip, s)            : {d_road}  [{ROAD_SCENARIO}]")
    print("  e  (engine type)             :")
    for k, v in sorted(E_ENGINE.items(), key=lambda x: x[1]):
        print(f"       {k:22s}: {v:.2f}")
    print("  f  (tire proportional)       : from TIRE_PARAMS[bt_group, is_premium]")
    print(f"  g  (road proportional)       : {g_road}  [{ROAD_SCENARIO}]")
    print("  l  (segment launch factor)   : from SEGMENT_FACTOR[bt_group, is_premium]")
    print(f"  h  (gearbox shift time, s)   : {H_GEARBOX}")
    print()


if __name__ == "__main__":
    main()
