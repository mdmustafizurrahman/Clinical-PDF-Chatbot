from transformers import pipeline

# Load Flan-T5-Base for instruction-tuned QA
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(question, context):
    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}"
    result = qa_pipeline(prompt, max_new_tokens=128)[0]["generated_text"]
    return result.strip()