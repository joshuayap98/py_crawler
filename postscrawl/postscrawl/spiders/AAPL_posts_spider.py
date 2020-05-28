import scrapy
import requests
from .exceptions import Exceptions
class StaticVariable:

    BLACKLIST_URLs = ['investors.com']

class AAPLPostSpider(scrapy.Spider):
    name = "AAPL_posts"
    allowed_domains = ['finance.yahoo.com']
    start_urls = [
        'https://finance.yahoo.com/quote/AAPL/news'
    ]
    posts_array = []

    def parse(self, response):
        print('---[START] Crawl process has started on %s---' % self.start_urls[0])
        for post in response.css('li.js-stream-content'):
            next_url = self.url_check(post.css('h3 a').attrib['href'])
            if next_url is not None:
                try:
                    print('[LOG] Attempt next_url: %s' % next_url)
                    post = scrapy.Request(next_url, self.parse_detail)
                    self.posts_array.append(post)
                    print('[LOG] Attempted next_url, added: \n%s.\nposts_array length: %d' 
                    % ( next_url, len(self.posts_array)))
                except:
                    print('[EXCEPTION] %s' % Exceptions.INVALID)
        print('---[END] Crawl process has ended on %s---' % self.start_urls[0])

                
    def url_check(self, url):
        if any(blacklist_url in url for blacklist_url in StaticVariable.BLACKLIST_URLs):
            print(Exceptions.BLACKLISTED + ': ' + url)
            return None
        try:
            print('[LOG] GET Request to url: %s' % url)
            requests.get(url)
            return url
        except:
            print('[EXCEPTION] %s : %s' % (Exceptions.INVALID, url))
            return 'https://' + self.allowed_domains[0] + url

    """
    [summary]
    return a post object.
    """
    def parse_detail(self, response):
        return {
            'title': response.css('title::text').get(),
            'time': response.css('time::text')[0].get() if not None else 'Date unspecified',
            'source': response.request.url
        }