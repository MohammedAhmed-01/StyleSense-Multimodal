import scrapy
from fashion_scraper.items import FashionScraperItem


class HoodiesSweatshirtsSpider(scrapy.Spider):
    name = "hoodies_sweatshirts"
    allowed_domains = ["nike.com", "www.nike.com"]

    def start_requests(self):
        with open("hoodies_sweatshirts_links.txt", "r", encoding="utf-8") as f:
            links = [line.strip() for line in f if line.strip()]

        self.logger.info(f"Loaded {len(links)} product links from hoodies_sweatshirts_links.txt")

        for index, link in enumerate(links, start=1):
            yield scrapy.Request(
                url=link,
                callback=self.parse_product,
                meta={
                    "label": "hoodies_sweatshirts",
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

        index = response.meta["index"]
        label = response.meta["label"]

        item["title"] = title
        item["description"] = description
        item["product_url"] = response.url.split("?")[0]
        item["label"] = label
        item["image_urls"] = [image_url] if image_url else []
        item["image_name"] = f"{label}/hoodie_{index}.jpg"

        yield item