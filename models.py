# Hugging Face tools for model loading and inference
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# For loading sentence embedding models
from sentence_transformers import (
    SentenceTransformer,
)


# Load a fine-tuned model pipeline for detecting whether text is a question
def load_question_classifier():

    # Load tokenizer specific to the question detection model
    tokenizer = AutoTokenizer.from_pretrained("mrsinghania/asr-question-detection")

    # Load the fine-tuned transformer model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "mrsinghania/asr-question-detection"
    )

    # Create a text classification pipeline with the model and tokenizer
    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, top_k=None
    )
    return classifier  # Return the pipeline for inference use


# Load a lightweight sentence embedding model for semantic similarity and vector-based tasks
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
