from chroma import *
from huggingchat import *
from utils import *
import time, random

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

def store_qa_pairs(username, collection_name, file_path):
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

store_qa_pairs("username", "collection_name", "temp/A_Brief_Introduction_To_AI.pdf");
collection = get_or_create_collection("username.collection_name.qa")

question_index = 0

while True:
    results = collection.get()  
    if not results or "documents" not in results or not results["documents"]:
        print("No questions found in the collection.")
        break
    
    questions = results["documents"]
    metadata = results["metadatas"]
    
    if question_index >= len(questions):
        break;
    
    question = questions[question_index]
    answer = metadata[question_index]["answer"]

    print("Question: ", question)
    print("Answer: ", answer)
    
    question_index += 1
