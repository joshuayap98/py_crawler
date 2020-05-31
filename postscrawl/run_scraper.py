import scrapy
import os
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from postscrawl.postscrawl.spiders.AAPL_posts_spider import AAPLPostSpider

class Scraper(object):
    def __init__(self):
        settings_file_path = 'postscrawl.postscrawl.settings'
        os.environ.setdefault('SCRAPY_SETTINGS_MODULE', settings_file_path)
        self.process = CrawlerProcess(get_project_settings())
        self.spider = AAPLPostSpider


    def start_AAPL_posts_crawl(self):
        self.process.crawl(self.spider)
        self.process.start()