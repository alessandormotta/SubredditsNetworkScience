import os
from huggingface_hub import snapshot_download
from transformers import pipeline
from typing import List, Dict, Tuple

class ZeroShotPoliticalClassifier:
    def __init__(
        self, 
        model_id: str = "facebook/bart-large-mnli", 
        local_dir: str = "./local_model",
        force_download: bool = False
    ):
        """
        Initialize the zero-shot classification pipeline once, showing a progress bar
        during the model download if it's not already cached locally.
        
        Args:
            model_id (str): The Hugging Face model ID (e.g. "facebook/bart-large-mnli").
            local_dir (str): The local directory where the model will be cached.
            force_download (bool): If True, forces re-download of the model even if it exists.
        """
        
        # Check if the local folder already has the model
        # If not, or if force_download = True, download with a progress bar
        if force_download or not os.path.isdir(local_dir):
            print(f"Downloading {model_id} model weights. This may take a while...")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                resume_download=True,     # If download is interrupted, it can resume
                force_download=force_download
                # progress=True is enabled by default so you get a nice progress bar
            )
        
        # Now load the pipeline from the local folder
        print("Loading the zero-shot classification pipeline...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=local_dir
        )
        print("Model loaded successfully!\n")
        
        # Define the political compass quadrants (candidate labels)
        self.candidate_labels = [
            "Left-Libertarian",
            "Right-Libertarian",
            "Left-Authoritarian",
            "Right-Authoritarian"
        ]
    
    def categorize_subreddit(self, description: str, queries: List[str]) -> Tuple[str, Dict[str, float]]:
        """
        Uses a zero-shot classification model to assign one of the four
        political compass categories to the combined text of 'description' and 'queries'.
        
        Returns:
            category (str): The predicted political quadrant.
            scores (Dict[str, float]): A dictionary of label -> confidence score.
        """
        # Combine description + queries into one text
        description = (description or "").lower()
        queries_text = " ".join((q or "").lower() for q in queries)
        combined_text = f"{description} {queries_text}".strip()
        
        # Perform zero-shot classification
        prediction = self.classifier(
            combined_text,
            candidate_labels=self.candidate_labels
        )
        
        # Build a dictionary of label -> score
        scores = dict(zip(prediction["labels"], prediction["scores"]))
        
        # The highest score label is our predicted category
        category = prediction["labels"][0]
        return category, scores


# ------------------- DEMO USAGE -------------------
if __name__ == "__main__":
    # Instantiating the classifier once
    classifier = ZeroShotPoliticalClassifier(
        model_id="facebook/bart-large-mnli", 
        local_dir="./bart_mnli_model",
        force_download=False  # Set to True if you want to force a re-download
    )

    # Example subreddit description and queries
    example_description = "We discuss economic freedom, free markets, and individual rights here."
    example_queries = ["economic freedom", "libertarian discourse"]

    # Perform classification
    predicted_category, category_scores = classifier.categorize_subreddit(example_description, example_queries)
    print("Predicted Category:", predicted_category)
    print("Scores:", category_scores)