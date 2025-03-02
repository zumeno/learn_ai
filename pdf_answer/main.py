from chroma import *
from huggingface import *
from utils import *

file_path = ""
question = ""

def store_document(collection, file_path):
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

def answer_question(collection, question):
    context = get_context(collection, question)
    answer = gemma7b_response(context, question)
    return answer

collection = create_collection("")
store_document(collection, file_path)
answer = answer_question(collection, question)
