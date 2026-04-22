import scrapy


class FashionScraperItem(scrapy.Item):
    title = scrapy.Field()
    description = scrapy.Field()
    product_url = scrapy.Field()
    label = scrapy.Field()

    image_urls = scrapy.Field()
    images = scrapy.Field()
    image_name = scrapy.Field()