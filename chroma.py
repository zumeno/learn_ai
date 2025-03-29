import chromadb
import os

def get_db_store_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.basename(current_dir) == 'learn_ai':
        return os.path.join(current_dir, 'db')

    learn_ai_path = os.path.join(current_dir, 'learn_ai')
    if os.path.isdir(learn_ai_path):
        return os.path.join(learn_ai_path, 'db')
    
    return os.path.join(current_dir, 'db')

def get_or_create_collection(name):
    collection = chroma_client.get_or_create_collection(name=name)
    return collection

def get_context(collection, input_text):
    query_results = collection.query(
        query_texts=[input_text],
        n_results=2
    )
    return query_results['documents']

DB_STORE_PATH = get_db_store_path()
print(f"DB_STORE_PATH: {DB_STORE_PATH}")

chroma_client = chromadb.PersistentClient(path=DB_STORE_PATH)
