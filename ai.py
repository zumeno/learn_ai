from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Tokenizer 
import torch
import os
import nltk
from nltk.tokenize import sent_tokenize

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
login(HUGGINGFACEHUB_API_TOKEN)

nltk.download("punkt")
nltk.download('punkt_tab')

def initialize_gemma():
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    return model, tokenizer

def initialize_t5_small_squad2():
    model_name = "allenai/t5-small-squad2-question-generation"  

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    return model, tokenizer

def initialize_t5_base():
    model_name = "google-t5/t5-base"  

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    return model, tokenizer

def ai_generate(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt")

    output = model.generate(
        **inputs,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=8192,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def ai_response(model, tokenizer, context, instruction, question, response_key):
    template = f"""
    ###guideline: Never mention that you were given a context or instructions. Respond naturally as if you are directly addressing the user.Also remember that you are not responding to anyone except the user.
    ###context:{context}
    ###instruction:{instruction}
    ###length: short
    ###question:{question}
    {response_key}:
    """
    response = ai_generate(model, tokenizer, template) 

    return response.rsplit(response_key + ":", 1)[-1].strip()

def ai_answer(model, tokenizer, context, question):
    instruction = """
    Answer the question using only the given information.
    - If the correct answer is present in the context, provide it concisely.
    - If the correct answer is NOT in the context, respond with exactly: 'I am not aware about it.'
    - Do NOT mention the context or refer to external sources.
    """
    return ai_response(model, tokenizer, context, instruction, question, "###answer")

def ai_hint(model, tokenizer, context, question):
    instruction = """
    Provide a hint to help answer the question without giving away the full answer.
    - The hint should be useful but should not explicitly state the answer.
    - Do NOT mention that you are providing a hint.
    - Do NOT refer to any context or external sources.
    """
    return ai_response(model, tokenizer, context, instruction, question, "###hint")

def ai_feedback(model, tokenizer, context, question, user_answer):
    instruction = """
    Evaluate the user's answer based on the correct answer found in the context.
    - Identify any missing or incorrect points in the user's answer.
    - Provide a clear and constructive explanation of these points under the section '###feedback'.
    - Do NOT mention that you are referring to a provided context or external text.
    - Respond naturally as if you are directly addressing the user.
    """
    return ai_response(model, tokenizer, context, instruction, f"{question}\n###user_answer:{user_answer}", "###feedback")

def ai_verdict(model, tokenizer, context, question, user_answer, feedback):
    instruction = """
    Based on the correct answer found in the context and the provided feedback, determine if the user's answer conveys the same meaning.
    - If the user's answer is correct, respond with 'Correct'.
    - If the user's answer is incorrect, respond with 'Incorrect'.
    - Do NOT provide additional explanations.
    """
    return ai_response(model, tokenizer, context, f"{question}\n###user_answer:{user_answer}\n###feedback{feedback}", instruction, "###verdict")

def create_question(model, tokenizer, sentence):
    template = f"""
    ###sentence:{sentence}
    ###instruction:
    Generate a question to cover the topic in the sentence given.
    - Do NOT number the question.
    - Do NOT include numbering formats like 1., 2., 3. at the start of any question.
    ###question:
    """
    response = ai_generate(model, tokenizer, template)
    
    return response.rsplit("###question:", 1)[-1].strip()

def generate_questions_and_answers(question_model, question_tokenizer, answer_model, answer_tokenizer, context):
    qa_pairs = {}
    sentences = sent_tokenize(context) 

    for sentence in sentences:
        question = create_question(question_model, question_tokenizer, sentence)
        answer = ai_answer(answer_model, answer_tokenizer, context, question)  

        print(question)
        print(answer)
        
        qa_pairs[question] = answer  

    print("Finished generating questions and answers.")

    return qa_pairs

gemma_model, gemma_tokenizer = initialize_gemma() 
t5_small_squad2_model, t5_small_squad2_tokenizer = initialize_t5_small_squad2()
t5_base_model, t5_base_tokenizer = initialize_t5_base()
