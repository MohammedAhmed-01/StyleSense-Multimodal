import scrapy
from fashion_scraper.items import FashionScraperItem


class PantsSpider(scrapy.Spider):
    name = "pants_tights"
    allowed_domains = ["nike.com", "www.nike.com"]

    def start_requests(self):
        with open("pants_tights_links.txt", "r", encoding="utf-8") as f:
            links = [line.strip() for line in f if line.strip()]

        self.logger.info(f"Loaded {len(links)} pants links")

        for index, link in enumerate(links, start=1):
            yield scrapy.Request(
                url=link,
                callback=self.parse_product,
                meta={
                    "label": "pants_trousers",
                    "index": index
                }
            )

    def parse_product(self, response):
        item = FashionScraperItem()

        title = response.css("title::text").get(default="").strip()

        description = response.css(
            'meta[name="description"]::attr(content)'
        ).get(default="").strip()

        image_url = response.css(
            'meta[property="og:image"]::attr(content)'
        ).get(default="").strip()

        if image_url.startswith("http://"):
            image_url = image_url.replace("http://", "https://")

        label = response.meta["label"]
        index = response.meta["index"]

        item["title"] = title
        item["description"] = description
        item["product_url"] = response.url.split("?")[0]
        item["label"] = label
        item["image_urls"] = [image_url] if image_url else []
        item["image_name"] = f"{label}/item_{index}.jpg"

        yield item