from openai.embeddings_utils import cosine_similarity
import numpy as np

def analyze(documents_dataframe, query_embeddings, k):
    ''' Take a list of document embeddings and compute most similar
    document to the query. Returns top k similar documents.
    '''
    documents_dataframe["similarity"] = documents_dataframe.embedding.apply(lambda x: cosine_similarity(x, query_embeddings))
    results = (
        documents_dataframe.sort_values("similarity", ascending=False)
        .head(k)
    )
    return list(results["documentID"])
