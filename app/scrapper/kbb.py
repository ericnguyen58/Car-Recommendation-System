"""
Module: kbb_car_crawler.py

Description:
------------
This module handles automated web crawling of vehicle data from
Kelley Blue Book (KBB). Its sole responsibility is to collect raw
car style and specification data using Selenium.

Responsibilities:
-----------------
- Load KBB vehicle pages
- Navigate style categories (tabs)
- Extract style-level specifications and pricing data
- Return results as structured pandas DataFrames

Non-Responsibilities:
--------------------
- Business logic
- Data validation or cleaning
- Persistence (database / files)
- Analytics or machine learning

Design Principle:
----------------
Single Responsibility — this module is strictly for data crawling.
"""

# Custom Chrome driver setup (configured elsewhere with options, anti-bot, etc.)
from app.core.driver import setup_chrome_driver as scd

# Selenium utilities for locating elements and waiting
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Data processing
import pandas as pd
import time

# Dictionary of car brands and their models (URL slugs used by KBB)
_CARS = {
    "acura": [
        "tlx",
        "rlx"
    ],
    "alfa-romeo": [
        "giulia"
    ],
    "aston-martin": [
        "rapide"
    ],
    "audi": [
        "a3",
        "a4",
        "a6",
        "a8",
        "s3",
        "rs-3",
        "s4",
        "r-4",
        "s6",
        "rs-6",
        "s8"
    ],
    "bentley": [
        "flying-spur"
    ],
    "bmw": [
        "2-series-gran-coupe",
        "3-series",
        "5-series",
        "7-series",
        "m3",
        "m5"
    ],
    "cadillac": [
        "ct4",
        "ct5",
        "ct6",
        "xts"
    ],
    "genesis": [
        "g70",
        "g80",
        "g90"
    ],
    "infiniti": [
        "q50",
        "q70"
    ],
    "jaguar": [
        "xe",
        "xf",
        "xj"
    ],
    "lexus": [
        "is",
        "es",
        "gs",
        "ls"
    ],
    "lincoln": [
        "mkz",
        "continental"
    ],
    "maserati": [
        "ghibli",
        "quattroporte"
    ],
    "mercedes-benz": [
        "a-class",
        "c-class",
        "e-class",
        "s-class",
        "cla",
        "cls",
        "amg-c-class",
        "amg-e-class",
        "amg-s-class"
    ],
    "porsche": [
        "panamera"
    ],
    "rolls-royce": [
        "ghost",
        "phantom"
    ],
    "tesla": [
        "model-3",
        "model-s"
    ],
    "volvo": [
        "s60",
        "s90"
    ]
}
CARS = {
    "audi": [
        "a3",      # exists for many years
        "a4",      # very stable
        "rs-3"     # ONLY test with year >= 2017
    ],
    "bmw": [
        "3-series" # very stable
    ],
    "lexus": [
        "es"       # very stable
    ]
}

YEARS = [2015, 2020]
#To do list
# Edit text so it is more stable -done
# Add more cars
# Crawl  spec, consumer revie, safety
# Try to access all styles of a car -done



def get_style_tabs(styles_section):
    """
    Retrieve all style category tabs (e.g., Sedan, Coupe, Wagon).

    Why:
    - Some pages contain tabs, others do not
    - Returning [None] allows unified handling downstream
    """
    tabs = styles_section.find_elements(
        By.XPATH,
        ".//button[@role='tab' or @aria-selected]"
    )
    return tabs if tabs else [None]

def activate_tab(driver, wait, styles_section, tab, idx):
    """
    Click a style category tab and wait for the content to update.

    Why:
    - KBB dynamically updates the DOM
    - We compare innerHTML before and after clicking to detect real changes
    """
    if tab is None:
        return None  # no category concept

    category = tab.text.strip() or f"category_{idx}"
    print(f"➡️ Clicking tab: {category}")

    previous_html = styles_section.get_attribute("innerHTML")
    driver.execute_script("arguments[0].click();", tab)

    wait.until(
        lambda d: d.find_element(By.ID, "styles")
        .get_attribute("innerHTML") != previous_html
    )

    wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@id='styles']//a[@title]")
        )
    )

    return category

def get_style_cards(driver):
    """
    Retrieve all style cards currently visible in the Styles section.
    """
    styles_section = driver.find_element(By.ID, "styles")
    return styles_section.find_elements(
        By.XPATH,
        ".//a[@title and contains(@href, '/')]"
    )

# Page loading & navigation helpers
def load_styles_section(driver, wait, url):
    """
    Load the Styles section on a Kelley Blue Book (KBB) car page.

    NOTE:
    - Some model-year combinations do not exist on KBB
      (e.g. Audi RS3 2015).
    - When the page does not exist, the 'styles' section
      will never appear and this function will timeout.
    """

    driver.get(url)

    return wait.until(
        EC.presence_of_element_located((By.ID, "styles"))
    )


# Data extraction helpers
def parse_style_card(card):
    """
    Parse a single style card into a structured dictionary.

    Returns:
    - Dictionary of extracted attributes
    - None if the card does not contain enough information
    """
    texts = card.text.split("\n")
    if len(texts) < 6:
        return None

    cargo_cu_ft, torque_lb_ft = extract_cargo_and_torque(texts)

    return {
        "Style": safe_get(texts, 0),
        "Price": safe_get(texts, 1),
        "MPG": safe_get(texts, 2),
        "Horsepower": safe_get(texts, 3),
        "Engine": safe_get(texts, 4),
        "CargoRoom_cu_ft": cargo_cu_ft,
        "Torque_lb_ft": torque_lb_ft,
        "0-60": safe_get(texts, -4),
        "Top Speed": safe_get(texts, -3),
        "Curb Weight": safe_get(texts, -2),
    }

def extract_cargo_and_torque(texts):
    """
    Extract cargo volume and torque values from style text lines.
    """
    cargo_cu_ft = None
    torque_lb_ft = None

    for t in texts:
        t = t.strip()
        if "cu ft" in t.lower():
            cargo_cu_ft = t
        elif "lb-ft" in t.lower():
            torque_lb_ft = t

    return cargo_cu_ft or "NA", torque_lb_ft or "NA"

def safe_get(arr, idx):
    """
    Safely access an index in a list.

    Prevents IndexError when cards have missing fields.
    """
    return arr[idx] if idx < len(arr) else None

def infer_category_from_style(style_name):
    """
    Infer vehicle category from the style name if needed.
    """
    for keyword in ["Sedan", "Wagon", "Coupe", "Convertible", "Hatchback"]:
        if keyword.lower() in style_name.lower():
            return keyword
    return None

# Main scraping logic
def scrape_kbb_styles(driver, wait, url):
    """
    Scrape all style variants for a given car model and year.

    Returns:
    - Pandas DataFrame containing style-level data
    """
    data = []

    try:
        # Attempt to load the Styles section
        styles_section = load_styles_section(driver, wait, url)

    except TimeoutException:
        # Most common reason for timeout:
        # - The model-year page does not exist on KBB
        # - Example: Audi RS3 2015
        # This is expected behavior and should be skipped.
        print(f"⏭️ Skipping unavailable model-year page: {url}")
        return pd.DataFrame(data)  # or return empty DataFrame if your pipeline expects it

    print(f"Visiting: {url}")

    styles_section = load_styles_section(driver, wait, url)
    tabs = get_style_tabs(styles_section)

    for idx, tab in enumerate(tabs):
        category = activate_tab(driver, wait, styles_section, tab, idx)

        style_cards = get_style_cards(driver)

        for card in style_cards:
            row = parse_style_card(card)
            if row:
                data.append(row)

    print(f"🛞 Found {len(style_cards)} styles.")
    return pd.DataFrame(data)

def kbb_worker():
    """
    Orchestrates scraping across brands, models, and years.

    Returns:
    - Combined DataFrame of all scraped vehicles
    """
    all_data = []

    driver = scd()
    wait = WebDriverWait(driver, 15)

    try:
        for brand, models in CARS.items():
            for model in models:
                for yr in YEARS:
                    url = f"https://www.kbb.com/{brand}/{model}/{yr}/"

                    df = scrape_kbb_styles(driver, wait, url)

                    if not df.empty:
                        df["Brand"] = brand.capitalize()
                        df["Model"] = model.capitalize()
                        df["Year"] = yr
                        all_data.append(df)

    finally:
        driver.quit()

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    

if __name__ == "__main__":
    df = kbb_worker()
    print(df)
    