from postscrawl.run_scraper import Scraper
from postscrawl.classifier.classifier import Classifier

def main():
   # scraper = Scraper()
   # scraper.start_AAPL_posts_crawl()
   classifier=Classifier()
   dataset=classifier.read_csv_data('./postscrawl/classifier/data/Combined_News_DJIA.csv')
   # train_dataset = dataset[dataset['Date'] < '2015-01-01']
   # test_dataset = dataset[dataset['Date'] > '2014-12-31']
   train_dataset = classifier.concat_dataset(dataset)
   classifier.generate_term_matrix(dataset, train_dataset)


   """
    [summary] Implementation 1.0
    """
"""
   data_model=classifier.train_model(train_dataset)
   classifier.prediction_result(test_dataset, data_model)
"""

if __name__ == '__main__':
    main()