import scrapy
import requests
from .exceptions import Exceptions
class StaticVariable:

    BLACKLIST_URLs = ['investors.com']

class AAPLPostSpider(scrapy.Spider):
    name = "AAPL_posts"
    allowed_domains = ['finance.yahoo.com', 'thestreet.com', 'marketwatch.com', 'investopedia.com']
    start_urls = ['https://finance.yahoo.com/quote/AAPL/news']

    def parse(self, response):
        print('%s[START]%s' % ('-'*50, '-'*50))
        for post in response.css('li.js-stream-content'):
            next_url = self.url_check(post.css('h3 a').attrib['href'])
            if next_url:
                # Create temp item dictionary for callback function to populate
                item = {}
                yield scrapy.Request(next_url, callback=self.parse_detail, meta={'item': item})
        

                
    def url_check(self, url):
        if any(blacklist_url in url for blacklist_url in StaticVariable.BLACKLIST_URLs):
            print(Exceptions.BLACKLISTED + ': ' + url)
            return None
        try:
            requests.get(url)
            return url
        except:
            print('[EXCEPTION] %s : %s' % (Exceptions.INVALID, url))
            return 'https://' + self.allowed_domains[0] + url

    """
    [summary]
    Yield item object to PostscrawlPipeline for data collection and storage
    """
    def parse_detail(self, response):
        item = response.meta['item']
        item['source'] = response.request.url
        item['title'] = response.css('title::text').get()
        item['date'] = response.css('time::text')[0].get() if not None else 'Date unspecified'
        item['body'] = response.css('p::text').getall()
        yield item