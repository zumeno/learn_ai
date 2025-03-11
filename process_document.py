from chroma import *
from free_ai import *
from premium_ai import *
from utils import *
import time, random
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=0) 

def store_document(username, collection_name, file_path):
    collection = get_or_create_collection(f"{username}.collection_name.text")
    text = ""
    if file_path.lower().endswith(".docx"):
        text = read_docx(file_path)
    elif file_path.lower().endswith(".pdf"):
        text = read_pdf(file_path)

    texts = text_splitter.split_text(text)
            
    for text in texts:
        collection.add(
            documents=text,
            ids=str(collection.count()+1000),
        )

def store_qa_pairs(tier, username, collection_name, file_path):
    collection = get_or_create_collection(f"{username}.{collection_name}.qa")
    text = ""

    if file_path.lower().endswith(".docx"):
        text = read_docx(file_path)
    elif file_path.lower().endswith(".pdf"):
        text = read_pdf(file_path)
   
    qa_pairs = {}
    if tier == "free":
        qa_pairs = free_generate_questions_and_answers(text)
    elif tier == "premium":
        qa_pairs = premium_generate_questions_and_answers(text)

    for question, answer in qa_pairs.items():
        collection.add(
            documents=question,  
            metadatas={"answer": answer},  
            ids=str(collection.count() + 1000),  
        )
