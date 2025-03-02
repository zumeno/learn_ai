import chromadb
import os

DIRECTORY_CONTAINING_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DB_STORE_PATH = os.path.join(DIRECTORY_CONTAINING_SCRIPT, 'db')

chroma_client = chromadb.PersistentClient(path=DB_STORE_PATH)

def get_or_create_collection(name):
    collection = chroma_client.get_or_create_collection(name=name)
    return collection

def get_context(collection, input_text):
    query_results = collection.query(
        query_texts=[input_text],
        n_results=2
    )
    return query_results['text']
