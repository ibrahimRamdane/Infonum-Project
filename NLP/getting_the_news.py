import pickle
import pandas as pd
from newsapi import *
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class NewsFetcher:
    '''
    A class for fetching news articles using the News API.

    Attributes:
        sources (str): Comma-separated list of news sources.
        sortBy (str): Sorting option for the articles.
        api_key (str): API key for authentication.

    Methods:
        fetch_articles(from_time, to_time, query): Fetch articles within a time range for a specific query.
        fetch_news(start_date, end_date, query): Fetch news articles for a range of dates.
        articles_distribution_over_one_day(): Plot the distribution of articles by time of day.
        articles_distribution_over_days(start_date, end_date): Plot the distribution of articles by date.
        clean_data(): Clean the fetched news data.
        save_data(name_of_file): Save the cleaned news data to a file.
    '''
    def __init__(self, api_key, window_size=6):
        valid_window_sizes = [1, 2, 3, 4, 6, 8, 12, 24]
        self.window_size = window_size

        self.sources = 'ars-technica,associated-press,axios,breitbart-news,business-insider,buzzfeed,engadget,fortune,hacker-news,next-big-future,recode,time,wired,bbc-news,the-verge,abc-news,bloomberg,cbc-news,google-news,cnn,reuters,msnbc,the-wall-street-journal,the-washington-times'
        self.sortBy = 'popularity'
        self.api_key = api_key  # API key for authentication

    def fetch_articles(self, from_time, to_time, query)->list:
        """Fetch articles within a time range for a specific query."""
        articles = []
        try:
            response = requests.get(
                'https://newsapi.org/v2/everything',
                params={
                    'sources': self.sources,
                    'from': from_time,
                    'to': to_time,
                    'language': 'en',
                    'sortBy': self.sortBy,
                    'apiKey': self.api_key
                }
            )
            response.raise_for_status()  # Raise an error for bad responses
            articles = response.json().get('articles', [])
        except requests.RequestException as e:
            print(f"Failed to fetch articles: {e}")
        return articles

    def fetch_news(self, start_date, end_date, query)->dict:
        """Fetch news articles for a range of dates."""
        from datetime import datetime, timedelta

        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        current = start
        articles_total = {}

        while current <= end:
            # ISO 8601 date strings
            from_time = current.isoformat()
            to_time = (current + timedelta(hours=self.window_size)).isoformat()

            articles = self.fetch_articles(from_time, to_time, query)
            articles_total[from_time] = articles

            # Increment current time by self.window_size hours
            current += timedelta(hours=self.window_size)

            # Handle next day transition
            if current.day != (current - timedelta(hours=self.window_size)).day:
                # Adjust to start at the beginning of the next day
                current = datetime(current.year, current.month, current.day)
        
        self.articles_total = articles_total
        return articles_total
    
    def plot_articles_distribution_over_one_day(self):
        segments_per_day = 24 // self.window_size
        article_counts = [0] * segments_per_day  # Adjusted based on window size

    def plot_articles_distribution_over_one_day(self):
        '''Plot the distribution of articles by time of day.'''
        segments_per_day = 24 // self.window_size
        article_counts = [0] * segments_per_day  # Adjusted based on window size

        for timestamp in self.articles_total.keys():
            hour = int(timestamp[11:13])  # Extract hour from timestamp
            index = hour // self.window_size  # Determine which segment the hour belongs to
            article_counts[index] += len(self.articles_total[timestamp])

        # Set up the hour labels for the x-axis
        hours = ['00:00-05:59', '06:00-11:59', '12:00-17:59', '18:00-23:59']
        hours = [f"{i*self.window_size:02d}:00-{(i+1)*self.window_size-1:02d}:59" for i in range(segments_per_day)]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(hours, article_counts, color='skyblue')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Articles')
        plt.title('Distribution of Articles by Time of Day')
        plt.xticks(hours)
        plt.grid(axis='y')

        plt.show()

    def plot_articles_distribution_over_days(self, start_date, end_date):
        '''Plot the distribution of articles by date.'''
        # Initialize a dictionary to count articles per day
        article_counts_per_day = {}

        # Assuming articles is the result from your fetch_news method
        # articles = news_fetcher.fetch_news(start_date, end_date, query)

        start_date = datetime.fromisoformat(start_date)
        end_date = datetime.fromisoformat(end_date)
        current_date = start_date

        # Initialize article counts for each day in the range to ensure continuity in the plot
        while current_date <= end_date:
            formatted_date = current_date.date().isoformat()  # Just the date in YYYY-MM-DD format
            article_counts_per_day[formatted_date] = 0
            current_date += timedelta(days=1)

        # Count the number of articles for each day
        for timestamp, articles_list in self.articles_total.items():
            date = timestamp[:10]  # Extract just the date part from the timestamp
            if date in article_counts_per_day:
                article_counts_per_day[date] += len(articles_list)

        # Prepare data for plotting
        dates = list(article_counts_per_day.keys())
        counts = list(article_counts_per_day.values())

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(dates, counts, marker='o', linestyle='-', color='skyblue')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.title('Distribution of Articles by Date')
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

        plt.show()
    
    def dict_to_df(self):
        '''Create df of the articles, and drop duplicates.'''
        df = pd.DataFrame()
        for date, news in self.articles_total.items():
            df_date = pd.DataFrame([{'Date': date, 'Name': new['source']['name'], 'Title': new['title'], 'Abstract':new['description']} for new in news])
            df = pd.concat([df, df_date], ignore_index=True)
        
        df.drop_duplicates(inplace=True)
        self.df = df
        return df
    
    def save_df(self, name_of_file = "name_file"):
        '''Save the cleaned news data to a file.'''
        self.df.to_pickle(name_of_file)


# if __name__ == "__main__":
#     #api_key='0bd6a2cb38be48d69a501953ef7f8270'
#     # fgara : beea3e5555144725be547cee1ea3fa03
#     # fcs : 0bd6a2cb38be48d69a501953ef7f8270
#     api_key='0bd6a2cb38be48d69a501953ef7f8270'
#     window_size = 12
#     news_fetcher = NewsFetcher(api_key, window_size)
#     start_date = "2024-02-27"
#     end_date = "2024-02-28"
#     query = "your query here"

#     articles = news_fetcher.fetch_news(start_date, end_date, query)

#     news_fetcher.plot_articles_distribution_over_one_day()

#     news_fetcher.plot_articles_distribution_over_days(start_date, end_date)

#     df = news_fetcher.dict_to_df()
#     print(df.head())

#     news_fetcher.save_df('test Getting the news.pkl')
