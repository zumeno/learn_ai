from huggingface_hub import login
from langchain_community.llms.huggingface_hub import HuggingFaceHub
import os
from langchain_text_splitters import TokenTextSplitter
from transformers import pipeline 

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
login("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")

gemma7b = HuggingFaceHub(repo_id='google/gemma-1.1-7b-it')
text_summarizer = HuggingFaceHub(repo_id='Falconsai/text_summarization')

text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=0) 

def gemma7b_response(context, question):
    template = f"""
    ###context:{context},
    ###instruction:Please provide your response based solely on the information provided in the context and provide the complete answer, If the answer is not in the context please respond with "I am not aware about it" and avoid responding with anything else if its not there,
    ###length: short
    ###question:{question},
    ###answer:
"""
    response = gemma7b(template,temperature=0.3,max_new_token=1000)
    answer = response.rsplit("###answer:", 1)[-1].strip()

    return answer

def text_summarize(input_text):
    response = text_summarizer(str(input_text))
    return response
