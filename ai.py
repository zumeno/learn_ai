from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
import textwrap
import torch
from torch.cuda.amp import autocast
import torch._dynamo
from nltk.tokenize import sent_tokenize
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._dynamo.config.suppress_errors = True

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
login(HUGGINGFACEHUB_API_TOKEN)

def initialize_model():
    model_name = "google/gemma-2-2b-it"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True,  
        bnb_4bit_quant_type="nf4"  
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return model, tokenizer, device

def ai_generate(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with autocast():  
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def ai_response(context, instruction, question, response_key):
    template = f"""
    ###guideline: Never mention that you were given a context or instructions. Respond naturally as if you are directly addressing the user.Also remember that you are not responding to anyone except the user.
    ###context:{context}
    ###instruction:{instruction}
    ###length: short
    ###question:{question}
    {response_key}:
    """
    response = ai_generate(template) 

    return response.rsplit(response_key + ":", 1)[-1].strip()

def ai_answer(context, question):
    instruction = """
    Answer the question using only the given information.
    - If the correct answer is present in the context, provide it concisely.
    - If the correct answer is NOT in the context, respond with exactly: 'I am not aware about it.'
    - Do NOT mention the context or refer to external sources.
    """
    return ai_response(context, instruction, question, "###answer")

def ai_hint(context, question):
    instruction = """
    Provide a hint to help answer the question without giving away the full answer.
    - The hint should be useful but should not explicitly state the answer.
    - Do NOT mention that you are providing a hint.
    - Do NOT refer to any context or external sources.
    """
    return ai_response(context, instruction, question, "###hint")

def ai_feedback(context, question, user_answer):
    instruction = """
    Evaluate the user's answer based on the correct answer found in the context.
    - Identify any missing or incorrect points in the user's answer.
    - Provide a clear and constructive explanation of these points under the section '###feedback'.
    - Do NOT mention that you are referring to a provided context or external text.
    - Respond naturally as if you are directly addressing the user.
    """
    return ai_response(context, instruction, f"{question}\n###user_answer:{user_answer}", "###feedback")

def ai_verdict(context, question, user_answer, feedback):
    instruction = """
    Based on the correct answer found in the context and the provided feedback, determine if the user's answer conveys the same meaning.
    - If the user's answer is correct, respond with 'Correct'.
    - If the user's answer is incorrect, respond with 'Incorrect'.
    - Do NOT provide additional explanations.
    """
    return ai_response(context, f"{question}\n###user_answer:{user_answer}\n###feedback{feedback}", instruction, "###verdict")

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

def generate_questions_and_answers(context, chunk_size=1024, batch_size=4):
    chunks = split_into_chunks(context, chunk_size)
    qa_pairs = {}

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]  
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)//batch_size) + 1}...")

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
            response = ai_generate(template)  
            batch_responses.append(response)

        for response in batch_responses:
            qa_list = response.rsplit("###qa_pairs:", 1)[-1].strip().split("Question:")
            for qa in qa_list:
                if "Answer:" in qa:
                    question, answer = qa.split("Answer:", 1)
                    qa_pairs[question.strip()] = answer.strip()

    print(f"Total questions generated: {len(qa_pairs)}")
    return qa_pairs

model, tokenizer, device = initialize_model() 
