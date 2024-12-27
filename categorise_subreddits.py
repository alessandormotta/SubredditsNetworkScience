from transformers import pipeline
from typing import List, Dict, Tuple


class ZeroShotPoliticalClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize the zero-shot classification pipeline once.
        
        Parameters:
            model_name (str): The name of the model to load. 
                              Defaults to "facebook/bart-large-mnli".
        """
        print("Loading zero-shot classification model. This may take a while the first time...")
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        
        # Define the political compass quadrants
        self.candidate_labels = [
            "Left-Libertarian",
            "Right-Libertarian",
            "Left-Authoritarian",
            "Right-Authoritarian"
        ]
        print("Model loaded successfully!\n")
    
    def _preprocess_text(self, description: str, queries: List[str]) -> str:
        """
        Combine subreddit description and queries into a single text block.
        Convert to lowercase to normalize.
        """
        description = (description or "").lower()
        query_text = " ".join((q or "").lower() for q in queries)
        return f"{description} {query_text}".strip()
    
    def categorize_subreddit(self, description: str, queries: List[str]) -> Tuple[str, Dict[str, float]]:
        """
        Uses a zero-shot classification model to assign one of four 
        political compass categories to the combined text.
        
        Returns:
            category (str): The predicted political quadrant.
            scores (Dict[str, float]): Confidence scores for each quadrant.
        """
        # Preprocess the text
        combined_text = self._preprocess_text(description, queries)
        
        # Perform zero-shot classification
        prediction = self.classifier(
            combined_text, 
            candidate_labels=self.candidate_labels
        )
        
        # Build a dict of {label: score}
        scores = {
            label: score for label, score in zip(prediction["labels"], prediction["scores"])
        }
        
        # The top label is our predicted category
        category = prediction["labels"][0]
        return category, scores


# ---------------- USAGE EXAMPLE ----------------
if __name__ == "__main__":
    # Initialize the classifier once
    classifier = ZeroShotPoliticalClassifier()

    # Example usage
    subreddit_description = "We discuss economic freedom, free markets, and individual rights here."
    queries = ["economic freedom", "libertarian discourse"]
    
    predicted_category, category_scores = classifier.categorize_subreddit(
        subreddit_description, queries
    )
    
    print("Predicted Category:", predicted_category)
    print("Scores:", category_scores)