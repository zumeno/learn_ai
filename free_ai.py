from ai import *

def free_ai_response(context, instruction, question, response_key):
    return ai_generate(t5_base_model, t5_base_tokenizer, context, instruction, question, response_key) 

def free_ai_hint(context, question):
    return ai_hint(t5_base_model, t5_base_tokenizer, context, question)

def free_ai_feedback(context, question, user_answer):
    return ai_feedback(t5_base_model, t5_base_tokenizer, context, question)

def free_ai_verdict(context, question, user_answer, feedback):
    return ai_verdict(t5_base_model, t5_base_tokenizer, context, question, user_answer, feedback)

def free_generate_questions_and_answers(context):
    return generate_questions_and_answers(t5_small_squad2_model, t5_small_squad2_tokenizer, t5_base_model, t5_base_tokenizer, context)

