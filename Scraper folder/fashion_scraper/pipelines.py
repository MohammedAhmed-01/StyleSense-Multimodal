import hashlib
from scrapy.pipelines.images import ImagesPipeline
from scrapy import Request


class FashionImagesPipeline(ImagesPipeline):

    def get_media_requests(self, item, info):
        for image_url in item.get("image_urls", []):
            yield Request(
                url=image_url,
                meta={"item": item}
            )

    def file_path(self, request, response=None, info=None, *, item=None):
        if item is None:
            item = request.meta.get("item", {})

        label = item.get("label", "unknown")
        image_guid = hashlib.md5(request.url.encode()).hexdigest()
        filename = f"{label}/{image_guid}.jpg"
        return filename

    def item_completed(self, results, item, info):
        if results:
            item["image_name"] = results[0][1]["path"]
        else:
            item["image_name"] = ""
        return item