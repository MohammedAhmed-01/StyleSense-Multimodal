from playwright.sync_api import sync_playwright
import time


def get_all_products():
    url = "https://www.nike.com/w/mens-hoodies-and-pullovers-6riveznik1"
    product_links = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        same_count_rounds = 0
        max_same_count_rounds = 20  # زيادة عدد الجولات
        scroll_rounds = 0
        max_scroll_rounds = 50  # زيادة عدد مرات السكروول

        while True:
            old_count = len(product_links)
            scroll_rounds += 1

            # Scroll to bottom - استخدام JavaScript
            for _ in range(10):
                page.evaluate("window.scrollBy(0, window.innerHeight * 1.5)")
                time.sleep(2)

            # لو فيه زرار تحميل/عرض المزيد اضغطه
            possible_buttons = [
                "button:has-text('Load More')",
                "button:has-text('Show More')",
                "button:has-text('View More')",
                "button:has-text('More')",
                "button[data-test='load-more']",
                "button[class*='load']",
            ]

            for selector in possible_buttons:
                try:
                    if page.locator(selector).count() > 0:
                        btn = page.locator(selector).first
                        if btn.is_visible():
                            btn.click()
                            time.sleep(3)
                except:
                    pass

            # اجمع اللينكات
            links = page.locator('a[href*="/t/"]').all()
            for l in links:
                try:
                    href = l.get_attribute("href")
                    if href and "/t/" in href:
                        clean_href = href.split("?")[0]

                        if clean_href.startswith("http"):
                            full = clean_href
                        else:
                            full = "https://www.nike.com" + clean_href

                        product_links.add(full)
                except:
                    pass

            new_count = len(product_links)
            print(f"Round {scroll_rounds}: Collected {new_count} products")

            # شرط التوقف
            if new_count == old_count:
                same_count_rounds += 1
            else:
                same_count_rounds = 0

            # توقف عند 1300 منتج او بعد 20 جولة متتاليه بدون زيادة او بعد 50 سكروول
            if new_count >= 1300 or same_count_rounds >= max_same_count_rounds or scroll_rounds >= max_scroll_rounds:
                break

        browser.close()

    return list(product_links)


if __name__ == "__main__":
    links = get_all_products()

    print("\nTOTAL PRODUCTS:", len(links))
    
    # حفظ في ملف
    with open("hoodies_sweatshirts_links.txt", "w", encoding="utf-8") as f:
        for link in links:
            f.write(link + "\n")
    print(f"Saved {len(links)} links to hoodies_sweatshirts_links.txt")

