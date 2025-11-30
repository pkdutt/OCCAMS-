import json
import time
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
)
from bs4 import BeautifulSoup


# ---------- CONFIG ----------
START_URL = "https://www.occamsadvisory.com/"     # <- change this
OUTPUT_JSON = "scraped_content.json"
MAX_PAGES = 200                       # safety limit

# ---------- SELENIUM SETUP ----------
def get_driver():
    options = Options()
    # options.add_argument("--headless=new")   # comment out if you want to see browser
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(20)
    return driver


# ---------- HELPERS ----------
def get_domain(url: str) -> str:
    return urlparse(url).netloc


def is_same_domain(url: str, root_domain: str) -> bool:
    return get_domain(url) == root_domain or get_domain(url).endswith("." + root_domain)


def extract_visible_text(html: str) -> str:
    """
    Uses BeautifulSoup to pull visible text, ignoring script/style/noscript.
    """
    soup = BeautifulSoup(html, "html.parser")

    # remove things that aren't content
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # get text with minimal extra whitespace
    text = soup.get_text(separator="\n")
    # clean up whitespace a bit
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines
    return "\n".join(lines)


def extract_links(driver, current_url: str, root_domain: str) -> set:
    """
    Uses Selenium to find all <a> tags, normalizes URLs,
    filters same-domain links and ignores hashtag links.
    """
    links = set()
    try:
        a_tags = driver.find_elements(By.TAG_NAME, "a")
    except NoSuchElementException:
        return links

    for a in a_tags:
        href = a.get_attribute("href")
        if not href:
            continue

        # ❌ Ignore anchor-only links or links containing '#'
        if "#" in href:
            continue

        # Normalize relative URLs
        href = urljoin(current_url, href)

        # Only crawl HTTP/HTTPS links
        if href.startswith("http://") or href.startswith("https://"):
            # Same domain filter
            if is_same_domain(href, root_domain):
                links.add(href)

    return links


# ---------- MAIN CRAWLER ----------
def crawl(start_url: str, max_pages: int = 100) -> dict:
    driver = get_driver()
    scraped_data = {}          # url -> text
    visited = set()
    to_visit = [start_url]
    root_domain = get_domain(start_url)

    try:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop()
            if url in visited:
                continue

            print(f"[{len(visited)+1}/{max_pages}] Visiting: {url}")
            visited.add(url)

            # Load page
            try:
                driver.get(url)
                time.sleep(2)  # small delay; adjust if needed
            except (TimeoutException, WebDriverException) as e:
                print(f"Failed to load {url}: {e}")
                continue

            # Extract text
            page_source = driver.page_source
            text = extract_visible_text(page_source)
            scraped_data[url] = text

            # Extract links and enqueue
            links = extract_links(driver, url, root_domain)
            for link in links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

    finally:
        driver.quit()

    return scraped_data


if __name__ == "__main__":
    data = crawl(START_URL, MAX_PAGES)

    # Save to JSON: { "url": "content" }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(data)} pages to {OUTPUT_JSON}")
