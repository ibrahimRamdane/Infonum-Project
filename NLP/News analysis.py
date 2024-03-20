import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class NewsAnalyzer:

    def __init__(self, df):
        self.df = df
        self.data_frames = dict()
        self.data_frames['df'] = df
        self.data_frames['df_similar'] = dict()
        self.data_frames['df_entity'] = dict()
        self.summizer_loaded = False

    def plot_of_clusters(self):
        plt.figure(figsize=(16, 5))
        
        plt.subplot(1, 1, 1)
        sns.histplot(data=self.df, x=variable, kde=True, color='#006837', stat="density", alpha=0.5, bins=25) # Adjust bins
        plt.title('Histogram')
        plt.xlabel('Cluster')
        plt.ylabel('Density')
        plt.xticks(rotation=35)  # Rotate x-axis labels by 35 degrees
        plt.show()

    def add_df(self, type, name, data_frame):
        self.data_frames[type][str(name)] = data_frame

    def add_df_similar_embedding(self, similarity=0.5):
        df_similar= self.df[self.df['Similarity'] > similarity]
        self.add_df('df_similar', similarity, df_similar)

    def add_df_similar_entity(self, similarity=0.5):
        df_similar= self.df[self.df['Entity'] > similarity]
        self.add_df('df_entity', similarity, df_similar)

    def get_i_most_similar_embedding_index(self, name, i=4):
        index_to_consider_embedding = list()
        index_to_consider_embedding.append(self.data_frames['df_similar'][name]['Cluster'].value_counts()[0:i].index)
        self.index_to_consider_embedding = index_to_consider_embedding
        return index_to_consider_embedding
    
    def get_i_most_similar_entity_index(self, name, i=4):
        index_to_consider_entity = list()
        index_to_consider_entity.append(self.data_frames['df_entity'][name]['Cluster'].value_counts()[0:i].index)
        self.index_to_consider_entity = index_to_consider_entity
        return index_to_consider_entity

    def plot_cluster_evolution(self, cluster_numbers, start_date, end_date):
        plt.figure(figsize=(14, 7))
        
        for cluster_number in cluster_numbers:
            # Filter the dataframe for the given cluster number and date range
            filtered_df = self.df[(self.df['Cluster'] == cluster_number) & (self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)]
            
            # Group by 'Date' and count the occurrences
            count_series = filtered_df.groupby('Date').size()
            
            # Reindex the series to include all dates in the range, filling missing values with 0
            count_series = count_series.reindex(pd.date_range(start_date, end_date), fill_value=0)
            
            # Plotting
            plt.plot(count_series.index, count_series.values, marker='o', linestyle='-', label=f'Cluster {cluster_number}')
        
        plt.title('Evolution of the count of clusters over time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping
        plt.legend()
        plt.show()

    def print_most_important_news_embedding(self, i=0, start_date='2024-02-27', end_date='2024-02-28'):
        cluster = self.index_to_consider_embedding[0][i]
        df_identificado = self.df[(self.df['Cluster'] == cluster) & (self.df["Date"] >= start_date) & (self.df["Date"] <= end_date)]
        news = str()
        for i in range(len(df_identificado)):
            print(df_identificado.iloc[i]['Date'])
            print(df_identificado.iloc[i]['Title'])
            news += df_identificado.iloc[i]['Title']
        return news

    def print_most_important_news_entity(self, i=0, start_date='2024-02-27', end_date='2024-02-28'):
        cluster = self.index_to_consider_entity[0][i]
        df_identificado = self.df[(self.df['Cluster'] == cluster) & (self.df["Date"] >= start_date) & (self.df["Date"] <= end_date)]
        news = str()
        for i in range(len(df_identificado)):
            print(df_identificado.iloc[i]['Date'])
            print(df_identificado.iloc[i]['Title'])
            news += df_identificado.iloc[i]['Title']
        return news

    def load_summarizer(self):
        from transformers import BartForConditionalGeneration, BartTokenizer

        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    def summarize(self, news):
        if not summizer_loaded:
            load_summarizer()
            summizer_loaded = True
        
        inputs_ids = self.tokenizer.encode("summarize: " + news, return_tensors="pt", max_length=512, truncation=True)

        summary_ids = self.model.generate(inputs_ids, max_length=150, num_beams=2, length_penalty=2.0, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("Summary :", summary)


if __name__=='__main__':
    df = pd.read_pickle('df embedding and entity and cluster and similarity- 1 janvier to 7 mars')
    news_analyzer = NewsAnalyzer(df)
    news_analyzer.plot_of_clusters()
    news_analyzer.add_df_similar_embedding(0.6)
    news_analyzer.add_df_similar_entity(0.5)

    index_1 = news_analyzer.get_i_most_similar_embedding_index(0.6, 4)
    index_2 = news_analyzer.get_i_most_similar_entity_index(0.5, 4)
    index_1_2 = set(index_1 + index_2)
    start_date = datetime(2024, 2, 1)
    end_date = datetime(2024, 3, 1)
    news_analyzer.plot_cluster_evolution(index_1_2, start_date, end_date)
