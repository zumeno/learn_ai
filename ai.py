from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import torch
import os
import nltk
from nltk.tokenize import sent_tokenize

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(HUGGINGFACEHUB_API_TOKEN)

nltk.download("punkt")
nltk.download('punkt_tab')

def initialize_gemma():
    model_name = "google/gemma-2-2b-it"
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model

def initialize_t5_small_squad2():
    model_name = "allenai/t5-small-squad2-question-generation"  
    quant_config = BitsAndBytesConfig(load_in_4bit=True)  

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")

    return model

def initialize_t5_base():
    model_name = "t5-base"  
    quant_config = BitsAndBytesConfig(load_in_4bit=True)  

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")

    return model
