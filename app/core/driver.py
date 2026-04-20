from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def setup_chrome_driver():

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_window_size(1920, 1080)

        # Fake navigator
        driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
        """)
        return driver
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChromeDriver: {e}")

if __name__ == "__main__":
    start_time = time.time()
    driver = setup_chrome_driver()
    driver.get("https://www.kbb.com/bmw/3-series/2016/")
    print("Page title is:", driver.title)
    driver.quit()
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Process time: {elapsed:.2f} seconds")