from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, Gemma3ForCausalLM  
import torch
from torch.cuda.amp import autocast
import torch._dynamo
from nltk.tokenize import sent_tokenize
import os
import time
import re

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')
torch._dynamo.config.suppress_errors = True

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
login(HUGGINGFACEHUB_API_TOKEN)

def initialize_model():
    model_name = "arunchess/gemma3-1b-it-4bit"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Gemma3ForCausalLM.from_pretrained(model_name, device_map="balanced")

    return model, tokenizer, device

def ai_generate(input_text, max_new_tokens):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.amp.autocast('cuda'):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clean_response(response, response_key):
    output = response.split(f"{response_key}:", 1)[-1].strip()

    output = re.sub(r'###.*', '', output)

    lines = output.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]

    return '\n'.join(non_empty_lines)

def ai_response(context, instruction, question, response_key, max_new_tokens):
    template = f"""
    ###guideline: Never mention that you were given a context or instructions. Respond naturally as if you are directly addressing the user as in you are talking to him/her.Also remember that you are not responding to anyone except the user.Also please avoid the usage of unnecessary symbols like # at the end.
    ###context:{context}
    ###instruction:{instruction}
    ###length: short
    ###question:{question}
    {response_key}:
    """
    return clean_response(ai_generate(template, max_new_tokens), response_key)

def ai_answer(context, question):
    instruction = """
    Answer the question using only the given information.
    - If the correct answer is present in the context, provide it concisely.
    - If the correct answer is NOT in the context, respond with exactly: 'I am not aware about it.'
    - Do NOT mention the context or refer to external sources.
    - Make sure to always tell the answer completely
    """
    return ai_response(context, instruction, question, "###answer", 1024)

def ai_hint(context, question, correct_answer):
    instruction = """
    Provide a hint to help answer the question without giving away the correct/full answer.
    - The hint should be useful but should not explicitly state the answer.
    - Do NOT mention that you are providing a hint.
    - Do NOT refer to any context or external sources.
    - Keep your response concise and focused on one key insight.
    - Do not create any new unnecessary sections we just need the hint(that means strictly no answer,questions or anything else which are not part of the hint)
    """
    return ai_response(context, instruction, f"{question}\n###correct_answer:{correct_answer}", "###hint", 512)

def ai_feedback(context, question, user_answer, correct_answer):
    instruction = """
    Evaluate the user's answer with complete objectivity and strict factual accuracy.
    - Detect all factual errors, missing details, and inaccuracies without compromise.
    - Directly reference the context for validation and provide detailed feedback on incorrect or incomplete points.
    - Do NOT validate incorrect answers or offer unnecessary praise.
    - Provide feedback in a clear, straightforward, and constructive manner under the section '###feedback'.
    - If the answer is completely wrong, explicitly state so and explain the correct answer.
    """
    return ai_response(context, instruction, f"{question}\n###user_answer:{user_answer}\n###correct_answer:{correct_answer}", "###feedback", 1024)

def ai_verdict(context, question, user_answer, correct_answer, feedback):
    instruction = """
    Based on the correct answer found in the context and the provided feedback, determine if the user's answer conveys the same meaning.
    - If the user's answer is correct, respond with 'Correct'.
    - If the user's answer is incorrect, respond with 'Incorrect'.
    - Do NOT provide additional explanations.
    """
    return ai_response(context, instruction, f"{question}\n###user_answer:{user_answer}\n###correct_answer{correct_answer}\n###feedback{feedback}", "###verdict", 32)

def ai_predict_rating(context, question, user_answer, correct_answer, feedback, verdict):
    instruction = """
    Predict exactly one rating based on the user's performance. Follow these strict rules:
    1. 'again' - Totally incorrect answer or no answer
    2. 'hard' - Partially correct with significant errors/hesitation
    3. 'good' - Mostly correct with minor mistakes
    4. 'easy' - Perfect answer with confidence
    
    Consider these factors:
    - Verdict: {verdict}
    - Feedback analysis: {feedback}
    - Comparison between user's answer and correct answer
    
    Respond ONLY with the exact rating word in lowercase: 'again', 'hard', 'good', or 'easy'.
    No explanations. No multiple answers.
    """
    return ai_response(context, instruction, f"{question}\n###user_answer:{user_answer}\n###correct_answer{correct_answer}\n###feedback{feedback}\n###verdict:{verdict}", "###rating", 32)

def split_into_chunks(text, chunk_size):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def generate_questions_and_answers(context, chunk_size=8192, batch_size=4):
    chunks = split_into_chunks(context, chunk_size)
    qa_pairs = {}

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]  

        batch_responses = []
        for chunk in batch_chunks:
            template = f"""
            ###context:{chunk}
            ###instruction:
            Generate a set of distinct questions that comprehensively cover every concept in the context while still reducing the number of questions generated.
            - Do NOT number the questions.
            - Do NOT include numbering formats like 1., 2., 3. at the start of any question.
            - Ensure each question is unique and does not repeat concepts.
            - Separate each question with a question mark (?), ensuring proper readability.

            After generating the questions, provide detailed and accurate answers for each of them.
            - Ensure the answers are well-structured and informative.
            - Maintain clarity and completeness in the responses.

            ###output_format:
            Question: <Generated Question>
            Answer: <Generated Answer>

            ###length: Generate as many questions as required to fully understand the content.
            ###qa_pairs:
            """
            response = ai_generate(template, 2048)  
            batch_responses.append(response)

        for response in batch_responses:
            qa_list = response.rsplit("###qa_pairs:", 1)[-1].strip().split("Question:")
            for index, qa in enumerate(qa_list):
                if index == 0:  
                    continue
                if "Answer:" in qa:
                    question, answer = qa.split("Answer:", 1)
                    qa_pairs[question.strip()] = re.sub(r'###.*', '', answer.strip())
                    

        torch.cuda.empty_cache()

    return qa_pairs

model, tokenizer, device = initialize_model()
