from learn_ai.chroma import *
from learn_ai.ai import *
from learn_ai.utils import *
from learn_ai.py_fsrs import *
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
   
    qa_pairs = generate_questions_and_answers(text)

    for question, answer in qa_pairs.items():
        collection.add(
            documents=question,  
            metadatas={"answer": answer},  
            ids=str(collection.count() + 1000),  
        )

def store_qa_pairs(tier, username, collection_name, file_path):
    collection = get_or_create_collection(f"{username}.{collection_name}.qa")
    text = ""

    if file_path.lower().endswith(".docx"):
        text = read_docx(file_path)
    elif file_path.lower().endswith(".pdf"):
        text = read_pdf(file_path)
   
    qa_pairs = generate_questions_and_answers(text)

    for question, answer in qa_pairs.items():
        card_dict = Card().to_dict() 

        filtered_card_dict = {k: v for k, v in card_dict.items() if v is not None}
        metadata = {"answer": answer, **filtered_card_dict}

        collection.add(
            documents=[question],
            metadatas=[metadata],
            ids=[str(collection.count() + 1000)]
        )

def get_next_card(username, collection_name):
    collection = get_or_create_collection(f"{username}.{collection_name}.qa")
    items = collection.get()
    cards = []
    
    for id, doc, metadata in zip(items['ids'], items['documents'], items['metadatas']):
        card_dict = {k: metadata.get(k, None) for k in Card().to_dict().keys()}
        card = Card.from_dict(card_dict)
        cards.append({
            'id': id,
            'question': doc,
            'answer': metadata['answer'],
            'card': card
        })
    
    sorted_cards = sorted(cards, key=lambda x: x['card'].due)
    return sorted_cards[0] if sorted_cards else None

def update_card(username, collection_name, card_id, rating):
    collection = get_or_create_collection(f"{username}.{collection_name}.qa")
    item = collection.get(ids=[card_id])
    metadata = item['metadatas'][0]
    
    card_dict = {k: metadata[k] for k in metadata.keys() if k in Card().to_dict().keys()}
    card_dict = {k: metadata.get(k, None) for k in Card().to_dict().keys()}
    card = Card.from_dict(card_dict)
    
    new_card, _ = reviewCard(scheduler, card, rating) 
    new_card_dict = new_card.to_dict()
   
    filtered_card_dict = {k: v for k, v in new_card_dict.items() if v is not None}

    new_metadata = {"answer": metadata['answer'], **filtered_card_dict}
    collection.update(ids=[card_id], metadatas=[new_metadata])
