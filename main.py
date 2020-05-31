from postscrawl.run_scraper import Scraper
from postscrawl.classifier.classifier import Classifier
def main():
   scraper = Scraper()
   scraper.start_AAPL_posts_crawl()


if __name__ == '__main__':
    main()