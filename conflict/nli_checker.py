from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


MODEL_NAME = "facebook/bart-large-mnli"

_tokenizer = None
_model = None


def load_nli_model():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print("Loading NLI model for the first time...")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        _model.eval()

    return _tokenizer, _model


def compare_statements(old_text, new_text):
    tokenizer, model = load_nli_model()

    inputs = tokenizer(
        old_text,
        new_text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    labels = {
        0: "contradiction",
        1: "neutral",
        2: "entailment"
    }

    predicted_index = torch.argmax(probabilities).item()

    return {
        "relationship": labels[predicted_index],
        "scores": {
            labels[index]: round(
                probabilities[index].item(),
                4
            )
            for index in range(len(labels))
        }
    }