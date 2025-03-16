from process_document import *

store_qa_pairs("free", "username", "collection_name", "temp/CODE OF BUSINESS ETHICS.docx")
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
