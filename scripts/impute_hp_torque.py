"""
Manual hp / torque imputation for rare, luxury, and EV-only vehicles.

Sources: Edmunds, UltimateSpecs, automobile-catalog, Wikipedia, manufacturer sites.
Run after standardize.py — patches data/final/cars/car_stats.UVCRS in place.

Strategy:
  - Exact specs looked up online for super-luxury / rare models
  - kW → hp conversion (×1.341) for EV-only entries where only motor wattage is known
  - Torque left null where manufacturers don't publish lb-ft (e.g. Lucid Air)
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CSV  = BASE / 'data' / 'final' / 'cars' / 'car_stats.UVCRS'

cars = pd.read_csv(CSV)
before_hp     = cars['hp'].isna().sum()
before_torque = cars['torque_lbft'].isna().sum()

# ---------------------------------------------------------------------------
# Helper — apply a patch to rows matching a boolean mask
# ---------------------------------------------------------------------------
def patch(mask, hp=None, torque=None):
    if hp is not None:
        cars.loc[mask & cars['hp'].isna(), 'hp'] = float(hp)
    if torque is not None:
        cars.loc[mask & cars['torque_lbft'].isna(), 'torque_lbft'] = float(torque)

def m(make, model, years=None, trim_contains=None):
    """Build a boolean mask by make + model + optional years / trim substring."""
    mask = (cars['make'] == make) & (cars['model'] == model)
    if years:
        mask &= cars['year'].isin(years if hasattr(years, '__iter__') else [years])
    if trim_contains:
        mask &= cars['trim'].str.contains(trim_contains, case=False, na=False)
    return mask

# ===========================================================================
# BENTLEY
# ===========================================================================
# Continental W12 (6.0L, 12-cyl) — standard W12 trim (Gen 2 onward: 567 hp / 516 lb-ft)
# Source: UltimateSpecs, automobile-catalog
for model in ['Continental Flying Spur', 'Continental GT', 'Continental GTC',
               'Continental GT Convertible']:
    patch(m('Bentley', model, range(2011, 2016)), hp=567, torque=516)

# Continental GT Speed — higher-output W12 Speed (626 hp / 605 lb-ft)
patch(m('Bentley', 'Continental GT Speed Convertible', [2014]), hp=626, torque=605)

# Continental Supersports — 621 hp W12 (factory overboost tune)
for model in ['Continental Supersports', 'Continental Supersports Convertible']:
    patch(m('Bentley', model, range(2011, 2014)), hp=621, torque=590)

# Flying Spur (3rd-gen body, 2014–2015) — W12 616 hp
patch(m('Bentley', 'Flying Spur', [2014, 2015]), hp=617, torque=590)

# Bentayga V8 PHEV 2020–2021 — 3.0 V6 + electric, 443 hp combined / 516 lb-ft
# Source: Wikipedia (Bentley Bentayga), Edmunds
patch(m('Bentley', 'Bentayga', [2020, 2021]), hp=443, torque=516)

# Bentayga Hybrid 2023 refresh — 456 hp / 516 lb-ft
patch(m('Bentley', 'Bentayga Hybrid', [2023]), hp=456, torque=516)

# Flying Spur Hybrid 2022–2024 — 2.9 V6 PHEV, 536 hp / 553 lb-ft
patch(m('Bentley', 'Flying Spur Hybrid', [2022, 2023, 2024]), hp=536, torque=553)

# ===========================================================================
# FERRARI  (combined system output for hybrids/PHEVs)
# ===========================================================================
# LaFerrari Hybrid 2014 — 6.3 V12 + HY-KERS, 950 hp / 664 lb-ft
# Source: Wikipedia (LaFerrari)
patch(m('Ferrari', 'LaFerrari Hybrid', [2014]), hp=950, torque=664)
patch(m('Ferrari', 'LaFerrari Aperta', [2017]), hp=950, torque=664)

# SF90 Stradale / Spider 2021–2024 — 3.9 V8 TT + 3×electric, 986 hp / 590 lb-ft
# Source: Wikipedia (Ferrari SF90 Stradale)
for model in ['SF90 Stradale Coupe', 'SF90 Stradale', 'SF90 Spider']:
    patch(m('Ferrari', model, range(2021, 2025)), hp=986, torque=590)

# 296 GTB / Spider 2022–2024 — 2.9 V6 PHEV, 819 hp / 546 lb-ft
# Source: Ferrari.com
for model in ['296 GTB', '296 GTB Spider']:
    patch(m('Ferrari', model, range(2022, 2025)), hp=819, torque=546)

# ===========================================================================
# McLAREN  (combined system)
# ===========================================================================
# P1 2014–2015 — 3.8 V8 TT + electric, 903 hp / 664 lb-ft
# Source: Wikipedia (McLaren P1)
patch(m('McLaren Automotive', 'P1', [2014, 2015]), hp=903, torque=664)

# Speedtail 2020 — 4.0 V8 TT hybrid, 1,036 hp / 848 lb-ft
# Source: automobile-catalog
patch(m('McLaren Automotive', 'Speedtail', [2020]), hp=1036, torque=848)

# Artura 2023 — 3.0 V6 PHEV, 671 hp / 531 lb-ft
# Source: McLaren / Edmunds
patch(m('McLaren Automotive', 'Artura', [2023]), hp=671, torque=531)

# ===========================================================================
# KOENIGSEGG
# ===========================================================================
# Regera 2019–2020 — 5.0 V8 TT + 3×electric, 1,500 hp / 1,475 lb-ft
# Source: Koenigsegg.com
patch(m('Koenigsegg', 'Regera', [2019, 2020]), hp=1500, torque=1475)

# ===========================================================================
# LAMBORGHINI
# ===========================================================================
# Revuelto 2024 — 6.5 V12 + 3×electric, 1,001 hp combined / 535 lb-ft (V12 crank)
# Source: Wikipedia (Lamborghini Revuelto)
patch(m('Lamborghini', 'Revuelto', [2024]), hp=1001, torque=535)

# ===========================================================================
# ROLLS-ROYCE
# ===========================================================================
# Spectre 2024 — dual electric (190 + 360 kW), official 584 hp / 664 lb-ft
# Source: MotorAuthority, Edmunds
for model in ['Spectre (22 inch wheels)', 'Spectre (23 inch wheels)',
              'Spectre Black Badge (22 inch wheels)', 'Spectre Black Badge (23 inch wheels)']:
    patch(m('Rolls-Royce', model, [2024]), hp=584, torque=664)

# ===========================================================================
# POLESTAR
# ===========================================================================
# Polestar 1 2020–2021 — 2.0L ICE + 3×electric, 619 hp / 738 lb-ft combined
# Source: Wikipedia (Polestar 1)
patch(m('Polestar', '1', [2020, 2021]), hp=619, torque=738)

# ===========================================================================
# MASERATI
# ===========================================================================
# GranTurismo MC / Sport 2014 — 4.7 V8 NA, 450 hp (kept) / 384 lb-ft torque only
# Source: automobile-catalog
patch(m('Maserati', 'GranTurismo', [2014], trim_contains='MC'),    torque=384)
patch(m('Maserati', 'GranTurismo', [2014], trim_contains='Sport'), torque=384)

# Quattroporte GTS 2014 — 3.8 V8 TT, 530 hp (kept) / 524 lb-ft torque only
# Source: UltimateSpecs
patch(m('Maserati', 'Quattroporte', [2014], trim_contains='GTS'), torque=524)

# Quattroporte S Q4 2014 — 3.0 V6 TT, 410 hp (kept) / 406 lb-ft torque only
patch(m('Maserati', 'Quattroporte', [2014], trim_contains='S Q4'), torque=406)

# Grecale GT 2023–2024 — 2.0L mild hybrid, 296 hp / 332 lb-ft
# Source: UltimateSpecs, cleanfleetreport
patch(m('Maserati', 'Grecale GT', [2023, 2024]), hp=296, torque=332)

# Grecale Modena 2023–2024 — 2.0L mild hybrid (higher tune), 325 hp / 332 lb-ft
patch(m('Maserati', 'Grecale Modena', [2023, 2024]), hp=325, torque=332)

# GranTurismo Folgore 2024 — 3×275 kW electric, 751 hp rated / 996 lb-ft
# Source: InsideEVs, Maserati.com
patch(m('Maserati', 'GranTurismo', [2024], trim_contains='Folgore'), hp=751, torque=996)
patch(m('Maserati', 'Grancabrio Folgore', [2024]), hp=751, torque=996)

# ===========================================================================
# KARMA
# ===========================================================================
# Revero 2018–2019 — 2.0L 4-cyl PHEV + 2×150 kW, 403 hp / 981 lb-ft combined
# Source: CarBuzz, Wikipedia (Karma Revero)
patch(m('Karma', 'Revero', [2018, 2019]), hp=403, torque=981)

# Revero GT 2020 — 1.5L 3-cyl PHEV + 2×175 kW, 536 hp / 550 lb-ft
patch(m('Karma', 'Revero GT (21-inch wheels)', [2020]), hp=536, torque=550)

# GS-6 2021 — same PHEV drivetrain as Revero GT
for trim in ['GS-6 (21-inch wheels)', 'GS-6 (22-inch wheels)']:
    patch(m('Karma', trim, [2021]), hp=536, torque=550)

# ===========================================================================
# FISKER
# ===========================================================================
# Fisker Karma 2012 — 2.0L PHEV + 2×150 kW, 403 hp / 981 lb-ft
# Source: Wikipedia (Fisker Karma)
patch(m('Fisker', 'Karma', [2012]), hp=403, torque=981)

# Ocean Extreme One 2023 — dual motor, 563 hp / 543 lb-ft
# Source: EVspecs
patch(m('Fisker', 'Ocean Extreme One', [2023]), hp=563, torque=543)

# Ocean Sport 2024 — single motor 207 kW → 278 hp / 279 lb-ft
for trim in ['Ocean Sport 20in', 'Ocean Sport 22in']:
    patch(m('Fisker', trim, [2024]), hp=278, torque=279)

# ===========================================================================
# LUCID AIR  (hp only — Lucid does not publish torque in lb-ft)
# ===========================================================================
# Source: Lucid Motors press releases, Edmunds
LUCID_HP = {
    ('Air Dream Edition Performance', 2022): 1111,
    ('Air Dream Edition Range',       2022): 1111,
    ('Air Grand Touring Performance', 2022): 1011,
    ('Air Grand Touring',             2022): 800,
    ('Air Pure',                      2022): 480,
    ('Air Touring',                   2022): 620,
    ('Air Grand Touring Performance', 2023): 1050,
    ('Air Grand Touring',             2023): 800,
    ('Air Pure',                      2023): 430,
    ('Air Sapphire',                  2023): 1234,
    ('Air Touring',                   2023): 620,
}
for (trim_name, yr), hp_val in LUCID_HP.items():
    patch(m('Lucid', 'Air', [yr], trim_contains=trim_name), hp=hp_val)

# ===========================================================================
# RIVIAN
# ===========================================================================
# R1T 2022 — quad motor, 835 hp / 908 lb-ft
# Source: Wikipedia (Rivian R1T)
patch(m('Rivian', 'R1T', [2022]), hp=835, torque=908)

# ===========================================================================
# ALFA-ROMEO
# ===========================================================================
# Stelvio Quadrifoglio 2018 — 2.9 V6 Biturbo, 505 hp / 443 lb-ft
# Source: Alfa Romeo
patch(m('Alfa-Romeo', 'Stelvio', [2018], trim_contains='Quadrifoglio'), hp=505, torque=443)

# Tonale eAWD 2024 — 1.3L PHEV + 89 kW, 272 hp / 347 lb-ft combined
# Source: Alfa Romeo
patch(m('Alfa-Romeo', 'Tonale', [2024], trim_contains='eAWD'), hp=272, torque=347)

# ===========================================================================
# VINFAST
# ===========================================================================
# VF 8 Eco 2023 — 2×130 kW = 260 kW → 348 hp / 368 lb-ft
# Source: KBB, TrueCar
patch(m('Vinfast', 'VF 8 Eco', [2023]), hp=348, torque=368)

# VF 8 Plus 2023 — 2×150 kW = 300 kW → 402 hp / 457 lb-ft
patch(m('Vinfast', 'VF 8 Plus', [2023]), hp=402, torque=457)

# VF 9 Eco 2024 — 300 kW → 402 hp / 457 lb-ft
patch(m('Vinfast', 'VF 9 Eco', [2024]), hp=402, torque=457)

# VF 9 Plus 2024 — 260 kW → 349 hp / 457 lb-ft
patch(m('Vinfast', 'VF 9 Plus', [2024]), hp=349, torque=457)

# ===========================================================================
# BUGATTI RIMAC
# ===========================================================================
# Nevera 2024 — 4×electric, 1,914 hp / 1,740 lb-ft
# Source: Rimac Automobili
patch(m('Bugatti Rimac', 'Nevera', [2024]), hp=1914, torque=1740)

# ===========================================================================
# EV CONVERSIONS  (kW → hp from Engine string; torque from manufacturer data)
# ===========================================================================

# smart fortwo electric drive (kW ratings from Engine column)
# Source: smart USA spec sheets
patch(m('smart', 'fortwo electric drive cabriolet',   [2011])
    | m('smart', 'fortwo electric drive coupe',       [2011]),
    hp=41, torque=88)                              # 30 kW

patch(m('smart', 'fortwo electric drive convertible', [2013, 2014, 2015, 2016])
    | m('smart', 'fortwo electric drive coupe',       [2013, 2014, 2015, 2016]),
    hp=74, torque=96)                              # 55 kW

patch(m('smart', 'fortwo electric drive convertible', [2017, 2018])
    | m('smart', 'fortwo electric drive coupe',       [2017, 2018])
    | m('smart', 'EQ fortwo (convertible)',            [2019])
    | m('smart', 'EQ fortwo (coupe)',                  [2019]),
    hp=81, torque=118)                             # 60 kW

# BYD e6 2012–2020 — 75 kW PMSM, ~100 hp / 148 lb-ft (200 Nm)
# Source: BYD tech sheets
for yr in range(2012, 2021):
    patch(m('BYD', 'e6', [yr]), hp=100, torque=148)

# Azure Dynamics Transit Connect Electric 2012 — 52 kW UQM, 70 hp / 162 lb-ft
# Source: UQM motor datasheet
patch(m('Azure Dynamics', 'Transit Connect Electric Van/Wagon', [2012]), hp=70, torque=162)

# CODA Automotive CODA 2012–2013 — 100 kW UQM, 134 hp / 163 lb-ft
# Source: CODA spec sheet
for yr in [2012, 2013]:
    patch(m('CODA Automotive', 'CODA', [yr]), hp=134, torque=163)

# Kandi K27 2021 — 20 kW, 27 hp / 70 lb-ft
# Source: Kandi spec sheet
patch(m('Kandi', 'K27', [2021]), hp=27, torque=70)

# Lordstown Endurance 2023 — 4×96 kW = 384 kW → 515 hp
# Torque: hub-motor architecture, not published as a combined lb-ft figure
patch(m('Lordstown', 'Endurance', [2023]), hp=515)

# Mahindra TR40 2011 — 2.2L diesel, 120 hp / 218 lb-ft (295 Nm)
# Source: Mahindra Genio specs
patch(m('Mahindra', 'TR40', [2011]), hp=120, torque=218)

# London Taxi 2003 — 2.4L diesel TX2, 94 hp / 163 lb-ft (221 Nm)
# Source: LTC TX2 technical data
patch(m('London Taxi', 'London Taxi', [2003]), hp=94, torque=163)

# ===========================================================================
# SUMMARY
# ===========================================================================
after_hp     = cars['hp'].isna().sum()
after_torque = cars['torque_lbft'].isna().sum()

print(f'hp      filled: {before_hp - after_hp:>4}  (remaining null: {after_hp:,})')
print(f'torque  filled: {before_torque - after_torque:>4}  (remaining null: {after_torque:,})')

cars.to_csv(CSV, index=False)
print(f'\nSaved → {CSV}')
