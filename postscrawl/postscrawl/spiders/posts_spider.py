import scrapy
class StaticVariable:
    
    BASE_URL = 'https://finance.yahoo.com'
    BLACKLIST_URLs = ['investors.com']

class PostsSpider(scrapy.Spider):
    name = "AAPL_posts"
    allowed_domains = ['finance.yahoo.com']
    start_urls = [
        'https://finance.yahoo.com/quote/AAPL/news'
    ]

    def parse(self, response):
        for post in response.css('li.js-stream-content'):
            next_url = self.url_check(post.css('h3 a').attrib['href'])
            if next_url:
                yield scrapy.Request(next_url, callback=self.parse_detail)

    """function to check valid url.
    append to StaticVariable -> BLACKLIST_URLs to block out websites
    """
    def url_check(self, url):
        if not url.startswith('https'):
            return StaticVariable.BASE_URL + url
        elif any(blacklist_url in url for blacklist_url in StaticVariable.BLACKLIST_URLs):
            return None

    """function to formulate date object.
    """
    def parse_detail(self, response):
        yield {
            'title': response.css('title::text').get(),
            'time': response.css('time::text')[0].get() if not None else 'Date unspecified'
        }