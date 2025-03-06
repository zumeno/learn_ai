from huggingface_hub import login
from langchain_community.llms.huggingface_hub import HuggingFaceHub
import os
from langchain_text_splitters import TokenTextSplitter
from transformers import pipeline 
import re

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
login("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")

gemma7b = HuggingFaceHub(repo_id='google/gemma-2-27b-it')
text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=0) 

def generate_gemma7b_response(context, instruction, question, response_key):
    template = f"""
    ###guideline: Never mention that you were given a context or instructions. Respond naturally as if you are directly addressing the user.Also remember that you are not responding to anyone except the user.
    ###context:{context}
    ###instruction:{instruction}
    ###length: short
    ###question:{question}
    {response_key}:
    """
    response = gemma7b(template, temperature=0.3, max_new_token=1000)
    return response.rsplit(response_key + ":", 1)[-1].strip()

def gemma7b_answer(context, question):
    instruction = """
    Answer the question using only the given information.
    - If the correct answer is present in the context, provide it concisely.
    - If the correct answer is NOT in the context, respond with exactly: 'I am not aware about it.'
    - Do NOT mention the context or refer to external sources.
    """
    return generate_gemma7b_response(context, instruction, question, "###answer")

def gemma7b_hint(context, question):
    instruction = """
    Provide a hint to help answer the question without giving away the full answer.
    - The hint should be useful but should not explicitly state the answer.
    - Do NOT mention that you are providing a hint.
    - Do NOT refer to any context or external sources.
    """
    return generate_gemma7b_response(context, instruction, question, "###hint")

def gemma7b_provide_feedback(context, question, user_answer):
    instruction = """
    Evaluate the user's answer based on the correct answer found in the context.
    - Identify any missing or incorrect points in the user's answer.
    - Provide a clear and constructive explanation of these points under the section '###feedback'.
    - Do NOT mention that you are referring to a provided context or external text.
    - Respond naturally as if you are directly addressing the user.
    """
    return generate_gemma7b_response(context, instruction, f"{question}\n###user_answer:{user_answer}", "###feedback")

def gemma7b_check_verdict(context, question, user_answer, feedback):
    instruction = """
    Based on the correct answer found in the context and the provided feedback, determine if the user's answer conveys the same meaning.
    - If the user's answer is correct, respond with 'Correct'.
    - If the user's answer is incorrect, respond with 'Incorrect'.
    - Do NOT provide additional explanations.
    """
    return generate_gemma7b_response(context, f"{question}\n###user_answer:{user_answer}\n###feedback{feedback}", instruction, "###verdict")

def create_questions(context):
    template = f"""
    ###context:{context}
    ###instruction:
    Generate a set of distinct questions that comprehensively cover every concept in the context.
    - Do NOT number the questions.
    - Do NOT include numbering formats like 1., 2., 3. at the start of any question.
    - Ensure each question is unique and does not repeat concepts.
    - Separate each question with a question mark (?), ensuring proper readability.

    ###length: Generate as many questions as required to fully understand the content.
    ###question:
    """
    response = gemma7b(template, temperature=0.1, max_new_token=1000)
    
    return response.rsplit("###question:", 1)[-1].strip().split("?")

def generate_questions_and_answers(context):
    chunk_size = 8192 
    chunk_overlap = 1024
    text_chunks = text_splitter.split_text(context)

    qa_pairs = {}
    
    for chunk in text_chunks:
        questions = create_questions(chunk)

        for question in questions:
            answer = gemma7b_answer(chunk, question)  
            qa_pairs[question] = answer  

    return qa_pairs
