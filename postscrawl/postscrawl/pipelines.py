# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import json

class PostscrawlPipeline:
    def process_item(self, item, spider):
        item = json.dumps(item, sort_keys=True)
        print(item)
        return item
