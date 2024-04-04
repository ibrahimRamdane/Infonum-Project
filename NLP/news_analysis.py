import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


class NewsAnalyzer:
    '''
    A class to analyze news articles and provide insights based on clustering 
    and entity recognition.
    
    Parameters:
        df (DataFrame): A DataFrame containing news articles.  
    '''

    def __init__(self, df, df_topic_of_cluster):
        df['Date'] = pd.to_datetime(df['Date'])
        self.df = df
        self.df_topic_of_cluster = df_topic_of_cluster
        self.data_frames = dict()
        self.data_frames['df'] = df
        self.data_frames['df_similar_title'] = dict()
        self.data_frames['df_similar_abstract'] = dict()
        self.data_frames['df_entity_title'] = dict()
        self.data_frames['df_entity_abstract'] = dict()
        self.summizer_loaded = False

    def plot_of_clusters(self):
        '''
        Plot a histogram of the clusters
        '''
        # TODO Bins must be adjusted
        plt.figure(figsize=(16, 5))
        
        plt.subplot(1, 1, 1)
        sns.histplot(
            data=self.df, x='Cluster', 
            kde=True, color='#006837', stat="density", alpha=0.5, bins=25
        )
        plt.title('Histogram')
        plt.xlabel('Cluster')
        plt.ylabel('Density')
        plt.xticks(rotation=35)  # Rotate x-axis labels by 35 degrees
        plt.show()

    def add_df(self, type_, name, data_frame):
        '''
        Add a dataframe to the data_frames dictionary
        of dictionaries. The first key is df_similar or df_entity
        and the second key is the score of similarity or of entity
        '''
        self.data_frames[type_][str(name)] = data_frame

    def add_df_similar_embedding(self, similarity=0.6, column="Title"):
        '''
        Add a dataframe to the data_frames dictionary for 
        similar embeddings with a certain score = similarity
        '''
        df_similar= self.df[self.df[f'Similarity {column}'] > similarity]
        self.add_df(f'df_similar_{column.lower()}', similarity, df_similar)

    def add_df_similar_entity(self, similarity=0.5, column="Title"):
        '''
        Add a dataframe to the data_frames dictionary for 
        similar entity with a certain score = similarity
        '''
        df_similar= self.df[self.df[f'Entity {column}'] > similarity]
        self.add_df(f'df_entity_{column.lower()}', similarity, df_similar)

    def get_i_most_similar_embedding_index(self, name, i=4):
        '''
        Get the indices of the i most similar embeddings.

        Parameters:
            name (str): The name or score of similarity for which to find 
            similar embeddings.
            i (int): The number of similar embeddings to retrieve.

        Returns:
            list: The indices of the i most similar embeddings.
        '''
        # index_to_consider_embedding = list()
        # print("self.data_frames['df_similar']", self.data_frames['df_similar'])
        # index_to_consider_embedding.append(
        #     self.data_frames['df_similar'][str(name)]['Cluster'].value_counts()[0:i].index
        # )
        # index_to_consider_embedding = list([i for i in index_to_consider_embedding][0])
        # self.index_to_consider_embedding = index_to_consider_embedding
        # return index_to_consider_embedding
        #print(self.data_frames['df_similar_title'][str(name)])
        counts1 = self.data_frames['df_similar_title'][str(name)]['Cluster Title'].value_counts()[0:6]  # Selecting the top 2 entries
        counts2 = self.data_frames['df_similar_abstract'][str(name)]['Cluster Abstract'].value_counts()[0:6] # Selecting the top 2 entries

        # Creating a dictionary with cluster titles as keys and their counts as values
        top_clusters_dict1 = counts1.to_dict()
        top_clusters_dict2 = counts2.to_dict()

        for key, value in top_clusters_dict2.items():
            if key in top_clusters_dict1:
                top_clusters_dict1[key] += value
            else:
                top_clusters_dict1[key] = value

        self.top_clusters_dict_similar = top_clusters_dict1
        top_i_similarity = list()
        for key, value in dict(sorted(top_clusters_dict1.items(), key=lambda x: x[1], reverse=True)[:i]).items():
            top_i_similarity.append(key)
        self.index_to_consider_embedding = top_i_similarity
        print('top_i_similarity', top_i_similarity)
        return top_i_similarity
    
    def get_i_most_similar_entity_index(self, name, i=4):
        '''
        Get the indices of the i most similar to the entity.

        Parameters:
            name (str): The name or score of similarity for which to find similar entity.
            i (int): The number of similar embeddings to retrieve. Default is 4.

        Returns:
            list: The indices of the i most similar entity.
        '''
        # index_to_consider_entity = list()
        # index_to_consider_entity.append(
        #     self.data_frames['df_entity'][str(name)]['Cluster'].value_counts()[0:i].index
        # )
        # index_to_consider_entity = list([i for i in index_to_consider_entity][0])
        # self.index_to_consider_entity = index_to_consider_entity
        # return index_to_consider_entity
    
        counts1 = self.data_frames['df_entity_title'][str(name)]['Cluster Title'].value_counts()[0:6]  # Selecting the top 2 entries
        counts2 = self.data_frames['df_entity_abstract'][str(name)]['Cluster Abstract'].value_counts()[0:6] # Selecting the top 2 entries

        # Creating a dictionary with cluster titles as keys and their counts as values
        top_clusters_dict1 = counts1.to_dict()
        top_clusters_dict2 = counts2.to_dict()

        for key, value in top_clusters_dict2.items():
            if key in top_clusters_dict1:
                top_clusters_dict1[key] += value
            else:
                top_clusters_dict1[key] = value

        self.top_clusters_dict_entity = top_clusters_dict1

        top_i_entity = list()
        for key, value in dict(sorted(top_clusters_dict1.items(), key=lambda x: x[1], reverse=True)[:i]).items():
            top_i_entity.append(key)
        print('top_i_entity' , top_i_entity)
        self.index_to_consider_entity = top_i_entity
        
        return top_i_entity

    def plot_cluster_evolution(self, cluster_numbers, start_date, end_date):
        '''
        Plot the evolution of the count of clusters over time.
        
        Parameters:
            cluster_numbers (list): The list of cluster numbers to consider.
            start_date (str): The start date in the format 'YYYY-MM-DD'.
            end_date (str): The end date in the format 'YYYY-MM-DD'.
        '''
        plt.figure(figsize=(14, 7))
        print('cluster_numbers',cluster_numbers)
        for cluster_number in cluster_numbers:
            # Filter the dataframe for the given cluster number and date range
            filtered_df = self.df[
                (self.df['Cluster'] == cluster_number) & 
                (self.df['Date'] >= start_date) & 
                (self.df['Date'] <= end_date)
            ]
            
            # Group by 'Date' and count the occurrences
            count_series = filtered_df.groupby('Date').size()
            
            # Reindex the series to include all dates in the range, filling missing values with 0
            count_series = count_series.reindex(pd.date_range(start_date, end_date), fill_value=0)
            
            # Plotting
            plt.plot(
                count_series.index, count_series.values, 
                marker='o', linestyle='-', label=f'Cluster {cluster_number}'
            )
        
        plt.title('Evolution of the count of clusters over time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping
        plt.legend()
        plt.show()

    def most_important_news_embedding(
            self, i=2, start_date='2024-02-27', end_date='2024-02-28', print_=True, column="Title"
        ):
        '''
        Get the most important news (embedding criteria) for a given cluster number 
        and date range.
        '''
        # TODO: it should be a range : cluster = self.index_to_consider_embedding[:i]
        clusters = self.index_to_consider_embedding[:i]
        # print(clusters)
        # df_identificado = self.df[
        #     (self.df[f'Cluster {column}'] == cluster) &
        #     (self.df["Date"] >= start_date) &
        #     (self.df["Date"] <= end_date)
        # ]
        # Initialize an empty list to store the filtered data frames
        #dfs_identificados = []
        news_entity, news_no_entity = [], []
        
        # Iterate over the clusters in self.index_to_consider_embedding[:i]
        for cluster in self.index_to_consider_embedding[:i]:
            # Filter the DataFrame for each cluster and time range
            df_identificado = self.df[
                (self.df[f'Cluster {column}'] == cluster) &
                (self.df["Date"] >= start_date) &
                (self.df["Date"] <= end_date)
            ]
            # Append the filtered DataFrame to the list
            #dfs_identificados.append(df_identificado_temp)
        
            # Concatenate all filtered DataFrames into a single DataFrame
            #df_identificado = pd.concat(dfs_identificados, ignore_index=True)

            df_identificado_entity = df_identificado[df_identificado[f'Entity {column}'] > 0.1]
            df_identificado_no_entity = df_identificado[df_identificado[f'Entity {column}'] <= 0.1]
            news_entity_ = str()
            for i in range(len(df_identificado_entity)):
                if print_: 
                    print(df_identificado_entity.iloc[i]['Date'], df_identificado_entity.iloc[i][column])
                news_entity_ += df_identificado_entity.iloc[i][column] +". "
            news_no_entity_ = str()
            for i in range(len(df_identificado_no_entity)):
                if print_: 
                    print(df_identificado_no_entity.iloc[i]['Date'], df_identificado_no_entity.iloc[i][column])
                news_no_entity_ += df_identificado_no_entity.iloc[i][column] +". "

            news_entity.append((news_entity_, self.df_topic_of_cluster[self.df_topic_of_cluster['Cluster'] == cluster]['Topics']))
            news_no_entity.append((news_no_entity_, self.df_topic_of_cluster[self.df_topic_of_cluster['Cluster'] == cluster]['Topics']))
        
        return news_entity, news_no_entity

    def most_important_news_entity(
            self, i=2, start_date='2024-02-27', end_date='2024-02-28', print_=True, column="Title"
        ):
        '''
        Get the most important news (entity criteria) for a given cluster number and date range.
        '''
        news = []
        #cluster = self.index_to_consider_entity[:i]
        for cluster in self.index_to_consider_embedding[:i]:
            df_identificado = self.df[
                (self.df[f'Cluster {column}'] == cluster) & 
                (self.df["Date"] >= start_date) & 
                (self.df["Date"] <= end_date)
            ]
            news_ = str()
            for i in range(len(df_identificado)):
                if print_: 
                    print(df_identificado.iloc[i]['Date'], df_identificado.iloc[i][column])
                news_ += df_identificado.iloc[i][column] +". "
            news.append((news_, self.df_topic_of_cluster[self.df_topic_of_cluster['Cluster'] == cluster]['Topics']))
        return news

    def load_summarizer(self):
        '''
        Load the summarizer model from the transformers library.
        '''
        from transformers import BartForConditionalGeneration, BartTokenizer

        the_model = "facebook/bart-large-cnn"
        self.model = BartForConditionalGeneration.from_pretrained(the_model)
        self.tokenizer = BartTokenizer.from_pretrained(the_model)

    def summarize(self, news, max_length=100):
        '''
        Summarize a given news article.
        '''
        if not self.summizer_loaded:
            self.load_summarizer()
            self.summizer_loaded = True
        summary = []
        for new, topic in news:
            
            inputs_ids = self.tokenizer.encode(
                #"summarize: " + news, return_tensors="pt", 
                new, return_tensors="pt",
                max_length=512, truncation=True
            )
            summary_ids = self.model.generate(
                inputs_ids, max_length=max_length, num_beams=2, 
                length_penalty=2.0, early_stopping=True
            )
            summary.append((self.tokenizer.decode(summary_ids[0], skip_special_tokens=True), topic))
        return summary

def get_topic_of_cluster(file):
    df_topics = pd.read_pickle(file)

    cluster_topics = {}
    for index, row in df_topics.iterrows():
        # For each cluster number in the row, add the topic to the cluster's list in the dictionary
        for cluster in row['Clusters']:
            if cluster in cluster_topics:
                # If the cluster already has a list, append the topic if it's not already in the list
                if row['Topic'] not in cluster_topics[cluster]:
                    cluster_topics[cluster].append(row['Topic'])
            else:
                # If the cluster doesn't have a list, create one with the current topic
                cluster_topics[cluster] = [row['Topic']]
    
    # Convert the cluster_topics dictionary to a dataframe
    df_topic_of_cluster = pd.DataFrame(list(cluster_topics.items()), columns=['Cluster', 'Topics'])

    return df_topic_of_cluster

if __name__=='__main__':
    df_topic_of_cluster = get_topic_of_cluster('topics_embeddings.pkl')

    file = 'df embedding and entity and cluster and similarity- 1 janvier to 7 mars'
    file = 'df all - 1 janvier to 26 mars.pkl'
    df = pd.read_pickle(file)
    news_analyzer = NewsAnalyzer(df, df_topic_of_cluster)
    #news_analyzer.plot_of_clusters()
    news_analyzer.add_df_similar_embedding(0.6, "Title")
    news_analyzer.add_df_similar_embedding(0.6, "Abstract")
    news_analyzer.add_df_similar_entity(0.5, "Title")
    news_analyzer.add_df_similar_entity(0.5, "Abstract")
    
    index_1 = news_analyzer.get_i_most_similar_embedding_index(0.6, 4)
    index_2 = news_analyzer.get_i_most_similar_entity_index(0.5, 4)
    index_1_2 = set(index_1).union(set(index_2))
    
    start_date = datetime(2024, 2, 1)
    end_date = datetime(2024, 3, 1)
    #news_analyzer.plot_cluster_evolution(index_1_2, start_date, end_date)
    
    news_apple, news_no_apple = news_analyzer.most_important_news_embedding(
        1, '2024-02-03', '2024-02-03', False, "Title"
    )
    news_apple = news_analyzer.summarize(news_apple)
    print('news_apple ')
    for new, topic in news_apple:
        print(topic, end=" : ")
        print(new)
    print("---------------","\n", "-----------------")
    news_no_apple = news_analyzer.summarize(news_no_apple)
    print('news_no_apple ')
    for new, topic in news_no_apple:
        print(topic, end=" : ")
        print(new)
    print("---------------","\n", "-----------------")
    news = news_analyzer.most_important_news_entity(
        1, '2024-02-03', '2024-02-03', False, "Title"
    )
    news_entity = news_analyzer.summarize(news)
    print('news_entity ', news_entity)

    news_apple, news_no_apple = news_analyzer.most_important_news_embedding(
        2, '2024-02-03', '2024-02-03', False, "Abstract"
    )
    news_apple = news_analyzer.summarize(news_apple)
    print('news_apple ')
    for new, topic in news_apple:
        print(topic, end=" : ")
        print(new)
    print("---------------","\n", "-----------------")
    news_no_apple = news_analyzer.summarize(news_no_apple)
    print('news_no_apple ')
    for new, topic in news_no_apple:
        print(topic, end=" : ")
        print(new)
    print("---------------","\n", "-----------------")
    
    news = news_analyzer.most_important_news_entity(
        2, '2024-02-03', '2024-02-03', False, "Abstract"
    )
    news_entity = news_analyzer.summarize(news)
    print('news_entity ', news_entity)
