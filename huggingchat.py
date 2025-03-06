from langchain_text_splitters import TokenTextSplitter
from hugchat import hugchat
from hugchat.login import Login
import time

sign = Login("arungeorgesaji@gmail.com", "ZCAG&sJ@AxgqnDSzg2hM")
cookies = sign.login()
cookie_path_dir = "./cookies_snapshot"
sign.saveCookiesToDir(cookie_path_dir)
chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

def generate_hugchat_response(context, instruction, question, response_key):
    template = f"""
    ###guideline: Never mention that you were given a context or instructions. Respond naturally as if you are directly addressing the user.Also remember that you are not responding to anyone except the user.
    ###context:{context}
    ###instruction:{instruction}
    ###length: short
    ###question:{question}
    {response_key}:
    """

    id = chatbot.new_conversation()
    chatbot.change_conversation(id)
    response = str(chatbot.chat(template))
    return response

def hugchat_answer(context, question):
    instruction = """
    Answer the question using only the given information.
    - If the correct answer is present in the context, provide it concisely.
    - If the correct answer is NOT in the context, respond with exactly: 'I am not aware about it.'
    - Do NOT mention the context or refer to external sources.
    """
    return generate_hugchat_response(context, instruction, question, "###answer")

def hugchat_hint(context, question):
    instruction = """
    Provide a hint to help answer the question without giving away the full answer.
    - The hint should be useful but should not explicitly state the answer.
    - Do NOT mention that you are providing a hint.
    - Do NOT refer to any context or external sources.
    """
    return generate_hugchat_response(context, instruction, question, "###hint")

def hugchat_provide_feedback(context, question, user_answer):
    instruction = """
    Evaluate the user's answer based on the correct answer found in the context.
    - Identify any missing or incorrect points in the user's answer.
    - Provide a clear and constructive explanation of these points under the section '###feedback'.
    - Do NOT mention that you are referring to a provided context or external text.
    - Respond naturally as if you are directly addressing the user.
    """
    return generate_hugchat_response(context, instruction, f"{question}\n###user_answer:{user_answer}", "###feedback")

def hugchat_check_verdict(context, question, user_answer, feedback):
    instruction = """
    Based on the correct answer found in the context and the provided feedback, determine if the user's answer conveys the same meaning.
    - If the user's answer is correct, respond with 'Correct'.
    - If the user's answer is incorrect, respond with 'Incorrect'.
    - Do NOT provide additional explanations.
    """
    return generate_hugchat_response(context, f"{question}\n###user_answer:{user_answer}\n###feedback{feedback}", instruction, "###verdict")

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
    id = chatbot.new_conversation()
    chatbot.change_conversation(id)
    response = str(chatbot.chat(template))

    return response.rsplit("###question:", 1)[-1].strip().split("?")

def generate_questions_and_answers(context):
    qa_pairs = {}
    
    questions = create_questions(context)

    for question in questions:
        answer = hugchat_answer(context, question)  
        qa_pairs[question] = answer  

    return qa_pairs
