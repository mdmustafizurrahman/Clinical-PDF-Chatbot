import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

nli_model_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_classifier = pipeline("text-classification", model=nli_model, tokenizer=nli_tokenizer)

similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def evaluate_faithfulness(context, answer):
    # Check if the answer is entailed by the context
    input_text = f"{context} </s> {answer}"
    max_length = 512
    tokens = nli_tokenizer.tokenize(input_text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        input_text = nli_tokenizer.convert_tokens_to_string(tokens)

    result = nli_classifier(input_text)[0]

    return {
        "label": result["label"],
        "score": round(result["score"], 3)
    }

def evaluate_relevance(question, answer):
    q_embed = similarity_model.encode(question, convert_to_tensor=True)
    a_embed = similarity_model.encode(answer, convert_to_tensor=True)
    score = util.pytorch_cos_sim(q_embed, a_embed).item()
    return round(score, 3)

def run_evaluation(context, question, answer):
    faithfulness = evaluate_faithfulness(context, answer)
    relevance = evaluate_relevance(question, answer)
    return {
        "faithfulness_label": faithfulness["label"],
        "faithfulness_score": faithfulness["score"],
        "relevance_score": relevance
    }
