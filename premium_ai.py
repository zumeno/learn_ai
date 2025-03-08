from ai import *

def premium_ai_response(context, instruction, question, response_key):
    return ai_generate(gemma_model, gemma_tokenizer, context, instruction, question, response_key) 

def premium_ai_hint(context, question):
    return ai_hint(gemma_model, gemma_tokenizer, context, question)

def premium_ai_feedback(context, question, user_answer):
    return ai_feedback(gemma_model, gemma_tokenizer, context, question)

def premium_ai_verdict(context, question, user_answer, feedback):
    return ai_verdict(gemma_model, gemma_tokenizer, context, question, user_answer, feedback)

def premium_generate_questions_and_answers(context):
    return generate_questions_and_answers(t5_small_squad2_model, t5_small_squad2_tokenizer, gemma_model, gemma_tokenizer, context)
