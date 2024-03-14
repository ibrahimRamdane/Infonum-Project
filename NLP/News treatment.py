import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import pandas as pd  # Assuming you're using pandas DataFrame for 'df'

from sklearn.cluster import KMeans
import math

from scipy.spatial.distance import cosine


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

    def get_embedding(self, text):
        # Tokenize and convert to input IDs
        inputs = self.tokenizer_bert(text, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            outputs = self.model_bert(**inputs)

        # Mean pooling to get one vector per sequence
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze()

    def add_embedding_column(self):
        '''
        Adds an 'Embedding' column to the dataframe.
        '''
        self.tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model_bert = AutoModel.from_pretrained("bert-base-uncased")

        def get_embedding_of_title(row):
            try:
                embedding = self.get_embedding(row['Title'])
                return embedding
            except Exception as e:
                return None

        def tensor_to_np(tensor):
            return tensor.numpy()

        self.df['Embedding'] = self.df.apply(get_embedding_of_title, axis=1)
        self.df = self.df.dropna(subset=['Embedding'])
        self.df['Embedding'] = self.df['Embedding'].apply(tensor_to_np)

        return self.df

    def add_ner_column(self, word='Apple'):
        '''
        Adds an 'Entity' column to the dataframe based on named entity recognition.

        Parameters:
        - word (str): The word to search for in the named entities. Default is 'Apple'.
        '''
        self.tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        self.model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER", force_download=True, resume_download=False)
        self.ner = pipeline("ner", model=self.model_ner, tokenizer=self.tokenizer_ner)

        def get_entity(row, word=word):
            try:
                ner_results = self.ner(row["Title"])
                has_apple_entity = any(entity['word'] == word and entity['entity'] == 'B-ORG' for entity in ner_results)
                if has_apple_entity:
                    for entity in ner_results:
                        if entity['word'] == word:
                            return entity['score']
                return 0
            except ValueError as e:
                return 0

        self.df['Entity'] = self.df.apply(get_entity, axis=1)

        return self.df

    def add_cluster_column(self, k=30):
        '''
        Adds a 'Cluster' column to the dataframe using K-means clustering.

        Parameters:
        - k (int): The number of clusters. Default is 30.
        '''
        # Concatenate the arrays from the 'embedding' column
        data = np.concatenate(self.df['Embedding'].values)

        # Reshape the data to have shape (n_samples, n_features)
        # Since each embedding has 768 features, and there are multiple embeddings,
        # we reshape it to (-1, 768)
        data = data.reshape(-1, 768)

        infinito = math.inf
        smallest_inertia = infinito
        km_final = None
        for _ in range(10):
            km = KMeans(n_clusters=k)
            km = km.fit(data)
            if km.inertia_ <= smallest_inertia:
                smallest_inertia = km.inertia_
                km_final = km

        best_model = km_final
        self.df['Cluster'] = best_model.labels_.tolist()

        return self.df

    def add_similar_column(self, reference):
        '''
        Adds a 'Similarity' column to the dataframe based on cosine similarity.

        Parameters:
        - reference (str): The reference text for calculating similarity.
        '''
        # Here use the functions of the class from apple
        embedding_ref = self.get_embedding(reference)

        def similarity_to_0(row):
            global embedding_ref
            return 1 - cosine(row['Embedding'], embedding_ref)

        self.df['Similarity'] = self.df.apply(similarity_to_0, axis=1)

        return self.df

    def save_df(self, file_name='file_name'):
        '''
        Saves the dataframe to a pickle file.

        Parameters:
        - file_name (str): The name of the pickle file. Default is 'file_name'.
        '''
        self.df.to_pickle(file_name)

        
if __name__=='__main__':
    df = pd.read_pickle('df simple - 1 janvier to 7 mars.pkl')
    text_processor = TextProcessor(df)
    text_processor.add_embedding_column()
    text_processor.add_ner_column('Apple')
    text_processor.add_cluster_column(30)

    ref_apple = """Apple Inc. (formerly Apple Computer, Inc.) is an American multinational technology company headquartered\
    in Cupertino, California, in Silicon Valley. It designs, develops, and sells consumer electronics, computer\
    software, and online services. Devices include the iPhone, iPad, Mac, Apple Watch, and Apple TV; operating\
    systems include iOS and macOS; and software applications and services include iTunes, iCloud, and Apple Music\
    """

    text_processor.add_similar_column(ref_apple)

    text_processor.save_df()

    
if __name__=='__main__':
    df = pd.read_pickle('df simple - 1 janvier to 7 mars.pkl')
    text_processor = TextProcessor(df)
    text_processor.add_embedding_column()
    text_processor.add_ner_column('Apple')
    text_processor.add_cluster_column(30)

    ref_apple = """Apple Inc. (formerly Apple Computer, Inc.) is an American multinational technology company headquartered\
    in Cupertino, California, in Silicon Valley. It designs, develops, and sells consumer electronics, computer\
    software, and online services. Devices include the iPhone, iPad, Mac, Apple Watch, and Apple TV; operating\
    systems include iOS and macOS; and software applications and services include iTunes, iCloud, and Apple Music\
    """

    text_processor.add_similar_column(ref_apple)

    text_processor.save_df()




    

        


        




        
            
        
            
        
        