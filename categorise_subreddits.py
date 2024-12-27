from transformers import pipeline
from typing import List, Dict, Tuple

# Load the zero-shot classification pipeline
# This may download a ~1.3GB model the first time you run it.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def preprocess_queries(queries: List[str]) -> str:
    """
    Combines and lowercases all queries into a single string.
    """
    queries_text = " ".join(q.lower() for q in queries)
    return queries_text

def categorize_subreddit(description: str, queries: List[str]) -> Tuple[str, Dict[str, float]]:
    """
    Uses a zero-shot classification model to assign one of four 
    political compass categories to the combined text of the 
    description and queries.
    
    Returns:
        category (str): The predicted political quadrant.
        scores (Dict[str, float]): Confidence scores for each quadrant.
    """
    # Define the four political compass quadrants as labels
    candidate_labels = [
        "Left-Libertarian", 
        "Right-Libertarian", 
        "Left-Authoritarian", 
        "Right-Authoritarian"
    ]
    
    # Combine description and queries into a single text block
    combined_text = (description or "").lower() + " " + preprocess_queries(queries or [])
    
    # Run zero-shot classification
    prediction = classifier(combined_text, candidate_labels=candidate_labels)
    
    # prediction['labels'] is a list of labels sorted by descending score
    # prediction['scores'] is a list of confidence scores corresponding to those labels
    # e.g. prediction['labels'][0] is the label with the highest confidence
    
    # Create a dictionary of label -> score
    scores = {label: 0.0 for label in candidate_labels}
    for label, score in zip(prediction["labels"], prediction["scores"]):
        scores[label] = score
    
    # The highest-confidence label is our predicted category
    category = prediction['labels'][0]
    return category, scores


# -------------- DEMO --------------
if __name__ == "__main__":
    # Example subreddit descriptions and queries
    example_description = "We discuss economic freedom, free markets, and individual rights here."
    example_queries = ["economic freedom", "libertarian discourse"]
    
    predicted_category, category_scores = categorize_subreddit(example_description, example_queries)
    
    print("Predicted Category:", predicted_category)
    print("Scores:", category_scores)