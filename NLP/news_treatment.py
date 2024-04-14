import pickle 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import torch
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import math

global contador_
contador_ = 0
global related_entities
related_entities = dict()


class TextProcessor:
    '''
    A class for processing text data.

    Methods:
    - get_embedding(text): Tokenizes the text and returns the embeddings.
    - add_embedding_column(): Adds an 'Embedding' column to the dataframe.
    - add_ner_column(word): Adds an 'Entity' column to the dataframe based on named entity recognition.
    - add_cluster_column(k): Adds a 'Cluster' column to the dataframe using K-means clustering.
    - add_similar_column(reference): Adds a 'Similarity' column to the dataframe based on cosine similarity.
    - save_df(file_name): Saves the dataframe to a pickle file.

    Parameters:
    - df (DataFrame): The dataframe to process.
    '''

    def __init__(self, df):
        self.df = df
        self.ner_model_loaded = False
        self.embedding_model_loaded = False
        self.topics = {
            "Politics": "Government policies, elections, diplomatic relations, and political events.",
            "Economics": "Global and local economic policies, trends, inflation rates, and economic growth.",
            "Technology": "Innovations, tech company news, emerging technologies like AI and blockchain, and impacts on society.",
            "Healthcare": "Healthcare policies, breakthroughs in medicine, pandemics, and health advice.",
            "Environment": "Climate change, sustainability efforts, natural disasters, and environmental policies.",
            "Education": "Trends in education, policy changes, educational technology, and issues facing schools and universities.",
            "Crime": "High-profile criminal cases, trends in crime rates, law enforcement tactics, and justice stories.",
            "Sports": "Major sports events, athlete profiles, sports politics, and updates on various sports leagues.",
            "Entertainment": "Movie releases, celebrity news, music industry updates, and cultural events.",
            "Science": "Discoveries, research updates, space exploration, and discussions on scientific phenomena.",
            "Business": "Corporate news, mergers and acquisitions, startups, and profiles on significant business personalities.",
            "International Affairs": "News on international relations, conflicts, treaties, and global events affecting multiple countries.",
            "Human Interest": "Individuals or events that appeal to emotions, including achievements, challenges, and inspirational tales.",
            "Technology and Cybersecurity": "Cyber threats, data breaches, and developments in cybersecurity technologies.",
            "Stock Market and Finance": "Updates on the stock market, financial analysis, investment strategies, and economic forecasts."
        }
        self.related_entities = dict()

    def get_embedding(self, text):
        # Tokenize and convert to input IDs
        inputs = self.tokenizer_bert(text, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            outputs = self.model_bert(**inputs)

        # Mean pooling to get one vector per sequence
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze()

    def load_embedding_model_from_preexisting(self, tokenizer_bert, model_bert):
        self.tokenizer_bert = tokenizer_bert
        self.model_bert = model_bert

        self.embedding_model_loaded = True
        print("embedding loaded")

    def load_embedding_model(self):
        from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
        self.tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model_bert = AutoModel.from_pretrained("bert-base-uncased")

        self.embedding_model_loaded = True
        print("embedding loaded")

    def add_embedding_column(self):
        '''
        Adds an 'Embedding' column to the dataframe.
        '''
        global contador_
        if not self.embedding_model_loaded:
            self.load_embedding_model()

        def get_embedding_i(row, column):
            try:
                embedding = self.get_embedding(row[column])
                return embedding
            except Exception as e:
                print("Error :(", e)
                return None
        
        def get_embedding_of_title(row):
            global contador_
            contador_ += 1
            if contador_ % 1000 == 0:
                print(".",end="")
            return get_embedding_i(row, 'Title')

        def get_embedding_of_abstract(row):
            global contador_
            contador_ += 1
            if contador_ % 1000 == 0:
                print(".",end="")
            return get_embedding_i(row, 'Abstract')

        def tensor_to_np(tensor):
            return tensor.numpy()
            
        contador_ = 0
        print("Embedding to title...")
        self.df['Embedding Title'] = self.df.apply(get_embedding_of_title, axis=1)
        contador_ = 0
        print("Embedding to abstract...")
        self.df['Embedding Abstract'] = self.df.apply(get_embedding_of_abstract, axis=1)
        self.df = self.df.dropna(subset=['Embedding Title'])
        self.df = self.df.dropna(subset=['Embedding Abstract'])
        self.df['Embedding Title'] = self.df['Embedding Title'].apply(tensor_to_np)
        self.df['Embedding Abstract'] = self.df['Embedding Abstract'].apply(tensor_to_np)
        print("Embedding finish")

        return self.df
    
    def load_ner_model_from_preexisting(self, tokenizer_ner, model_ner):
        self.tokenizer_ner = tokenizer_ner
        self.model_ner = model_ner
        self.ner = pipeline("ner", model=self.model_ner, tokenizer=self.tokenizer_ner)
        self.ner_model_loaded = True
        print('loaded ner...')

    def load_ner_model(self, tokenizer_ner, model_ner):
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        self.tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        self.model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER", force_download=True, resume_download=False)
        self.ner = pipeline("ner", model=self.model_ner, tokenizer=self.tokenizer_ner)
        print('loaded ner...')

    def add_ner_column(self, word='Apple'):
        '''
        Adds an 'Entity' column to the dataframe based on named entity recognition.

        Parameters:
        - word (str): The word to search for in the named entities. Default is 'Apple'.
        '''
        global contador_
        if not self.ner_model_loaded:
            self.load_ner_model()

        global related_entities
            
        def get_entity(row, column, word):
            global related_entities
            try:
                ner_results = self.ner(row[column])
                has_apple_entity = any(entity['word'] == word and entity['entity'] == 'B-ORG' for entity in ner_results)
                score = 0
                if has_apple_entity:
                    for entity in ner_results:
                        if entity['word'] == word:
                            score = entity['score']
                        else:
                            if not entity['word'] in self.related_entities:
                                print(entity['word'], end=" - ")
                                related_entities[entity['word']] = 0
                            else:
                                related_entities[entity['word']] += 1
                return score
            except ValueError as e:
                return 0
        def get_entity_title(row, word=word):
            global contador_
            contador_ += 1
            if contador_ % 1000 == 0:
                print(".",end="")
            return get_entity(row, column="Title", word=word)
        def get_entity_abstract(row, word=word):
            global contador_
            contador_ += 1
            if contador_ % 1000 == 0:
                print(".",end="")
            return get_entity(row, column="Abstract", word=word)
        contador_ = 0
        print("Embedding to title...")
        self.df['Entity Title'] = self.df.apply(get_entity_title, axis=1)
        contador_ = 0
        print("Embedding to title...")
        self.df['Entity Abstract'] = self.df.apply(get_entity_abstract, axis=1)
        print("Entity finish")

        print('self.related_entities', related_entities)
        self.related_entities = related_entities
        df_related_entities = pd.DataFrame([self.related_entities])
        df_related_entities.to_pickle("df_related_entities.pkl")

        return self.df

    def add_cluster_column(self, k=25):
        '''
        Adds a 'Cluster' column to the dataframe using K-means clustering.

        Parameters:
        - k (int): The number of clusters. Default is 25.
        '''
        # # Concatenate the arrays from the 'embedding' column
        # data = np.concatenate(self.df['Embedding'].values)

        # # Reshape the data to have shape (n_samples, n_features)
        # # Since each embedding has 768 features, and there are multiple embeddings,
        # # we reshape it to (-1, 768)
        # data = data.reshape(-1, 768)

        data_titles = np.concatenate(self.df['Embedding Title'].values).reshape(-1, 768)
        print('len(data_titles)',len(data_titles))
        data_abstracts = np.concatenate(self.df['Embedding Abstract'].values).reshape(-1, 768)
        print('len(data_abstracts)',len(data_abstracts))
        #data_combined = np.concatenate((data_titles, data_abstracts), axis=1)
        data_combined = np.vstack((data_titles, data_abstracts))
        print('len(data_combined)',len(data_combined))

        infinito = math.inf
        smallest_inertia = infinito
        km_final = None
        for _ in range(10):
            km = KMeans(n_clusters=k)
            km = km.fit(data_combined)
            if km.inertia_ <= smallest_inertia:
                smallest_inertia = km.inertia_
                km_final = km

        best_model = km_final
        
        labels_list = best_model.labels_.tolist()
        print('len(labels_list)',len(labels_list))
        midpoint = len(labels_list) // 2  # Use floor division to get an integer
        self.df['Cluster Title'] = labels_list[:midpoint]
        self.df['Cluster Abstract'] = labels_list[midpoint:]

        return self.df
    
    
    def assign_topics_to_clusters(self, name_file="df_topics.pkl"):
        topic_embeddings = [self.get_embedding(self.topics[topic]) for topic in self.topics]
        
        # Get the centroids of the clusters
        centroids = self.best_model.cluster_centers_
        
        # Calculate distances between each cluster centroid and each topic embedding
        distances = euclidean_distances(centroids, topic_embeddings)
        
        # For each cluster, find the closest topic
        closest_topics_indices = distances.argmin(axis=1)
        topics_list = list(self.topics.keys())
        closest_topics = [topics_list[index] for index in closest_topics_indices]
        
        # Map each cluster to its closest topic
        self.dict_topic_of_cluster = {cluster: topic for cluster, topic in enumerate(closest_topics)}
        df_topics = pd.DataFrame(data)
        df_topics.to_pickle(name_file)
        
        return self.dict_topic_of_cluster


    def add_similar_column(self, reference):
        '''
        Adds a 'Similarity' column to the dataframe based on cosine similarity.

        Parameters:
        - reference (str): The reference text for calculating similarity.
        '''
        # Here use the functions of the class from apple
        embedding_ref = self.get_embedding(reference)

        def similarity_to_0_title(row):
            return 1 - cosine(row['Embedding Title'], embedding_ref)
        def similarity_to_0_abstract(row):
            return 1 - cosine(row['Embedding Abstract'], embedding_ref)

        self.df['Similarity Title'] = self.df.apply(similarity_to_0_title, axis=1)
        self.df['Similarity Abstract'] = self.df.apply(similarity_to_0_abstract, axis=1)

        return self.df

    def save_df(self, file_name='test_News_treatment.pkl'):
        '''Saves the dataframe to a pickle file.'''
        self.df.to_pickle(file_name)


if __name__ == '__main__':


    from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_bert = AutoModel.from_pretrained("bert-base-uncased")

    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER", force_download=True, resume_download=False)

    # Define your file and load DataFrame outside the try-except blocks
    file = 'df simple - 1 janvier to 26 mars.pkl'
    file = "test_getting_the_news.pkl"
    #file = 'test Getting the news.pkl'
    df = pd.read_pickle(file)

    text_processor = TextProcessor(df)
    text_processor.load_embedding_model_from_preexisting(tokenizer_bert, model_bert)
    #text_processor.load_embedding_model()
    text_processor.load_ner_model_from_preexisting(tokenizer_ner, model_ner)
    
    steps = [
        'load_dataframe', 
        'add_embedding_column',
        'add_ner_column',
        'add_cluster_column',
        'add_similar_column',
        'save_df'
    ]

    while steps:
        current_step = steps[0]
        
        if current_step == 'load_dataframe':
            # Assuming loading is done above, just a placeholder here
            print("DataFrame loaded.")
        
        elif current_step == 'add_embedding_column':
            text_processor.add_embedding_column()
            print("Embedding column added.")
            
        elif current_step == 'add_ner_column':
            text_processor.add_ner_column('Apple')
            print("NER column added.")
            
        elif current_step == 'add_cluster_column':
            text_processor.add_cluster_column(10)
            print("Cluster column added.")
            
        elif current_step == 'add_similar_column':
            ref_apple = """Apple Inc. (formerly Apple Computer, Inc.) is an American multinational technology company headquartered in Cupertino, California, in Silicon Valley. It designs, develops, and sells consumer electronics, computer software, and online services."""
            text_processor.add_similar_column(ref_apple)
            print("Similar column added.")
            
        elif current_step == 'save_df':
            text_processor.save_df('test_News_treatment.pkl')
            print("DataFrame saved.")
            
        # Remove the current step from the list if it completes successfully
        steps.pop(0)
