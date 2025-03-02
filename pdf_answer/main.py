from chroma import *
from huggingface import *
from utils import *

def store_document(collection_name, file_path):
    collection = get_or_create_collection(collection_name)
    text = ""
    if file_path.lower().endswith(".docx"):
        text = read_docx(file_path)
    elif file_path.lower().endswith(".pdf"):
        text = read_pdf(file_path)

    summarized_text = text_summarize(text)
    texts = text_splitter.split_text(text)
            
    for text in texts:
        collection.add(
            documents=text,
            metadatas={"Summary" : summarized_text},
            ids=str(collection.count()+1000),
        )
