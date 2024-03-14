import requests
from transformers import AutoTokenizer, AutoModel
import torch


class WikipediaAPI:
    '''
    A class for fetching Wikipedia page content and generating embeddings.

    Attributes:
        page_title (str): The title of the Wikipedia page.
        max_chars (int, optional): The maximum number of characters to include in the reference. Defaults to None.
        reference (str): The extracted text from the Wikipedia page.

    Methods:
        __init__(self, page_title: str, max_chars: int = None): Initializes a new instance of the WikipediaAPI class.
        get_embedding(self) -> torch.Tensor: Retrieves the embedding for the reference text.
        embedding_of_text(self) -> torch.Tensor: Retrieves the embedding for the reference text using BERT model.
    '''

    def __init__(self, page_title: str, max_chars: int = None):
        self.page_title = page_title
        self.max_chars = max_chars
        self.reference = None

        endpoint = "https://en.wikipedia.org/w/api.php"
        params = {
            "format": "json",
            "action": "query",
            "prop": "extracts",
            "explaintext": "",
            "titles": page_title
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()

            if "query" in data and "pages" in data["query"]:
                pages = data["query"]["pages"]
                page_id = next(iter(pages))
                if "extract" in pages[page_id]:
                    full_text = pages[page_id]["extract"]

                    if max_chars is not None and len(full_text) > max_chars:
                        self.reference = full_text[:max_chars]
                    else:
                        self.reference = full_text

        except requests.RequestException as e:
            print(f"Error making API request: {e}")

    def get_embedding(self)-> torch.Tensor:
        '''Retrieves the embedding for the reference text.'''
        # Tokenize and convert to input IDs
        inputs = self.tokenizer_bert(self.reference, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            outputs = self.model_bert(**inputs)

        # Mean pooling to get one vector per sequence
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze()
    
    def embedding_of_text(self)-> torch.Tensor:
        '''Retrieves the embedding for the reference text using BERT model.'''       
        model = "bert-base-uncased"
        self.tokenizer_bert = AutoTokenizer.from_pretrained(model)
        self.model_bert = AutoModel.from_pretrained(model)

        self.embedding_ref = self.get_embedding()

        return self.embedding_ref.numpy()


if __name__ == "__main__":
    enterprise = "Apple Inc."
    max_chars = 2400

    apple_reference = WikipediaAPI(enterprise, max_chars)

    print(apple_reference.reference)

    embedding = apple_reference.embedding_of_text()
    
    print(embedding)
    