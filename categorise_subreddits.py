import os
from typing import List, Dict, Tuple
from huggingface_hub import snapshot_download
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ZeroShotPoliticalClassifier:
    def __init__(
        self,
        model_id: str = "facebook/bart-large-mnli",
        local_dir: str = "./local_bart_mnli",
        force_download: bool = False
    ):
        """
        Initialize a zero-shot classifier without using the transformers pipeline.

        Args:
            model_id (str): The Hugging Face model ID (default: "facebook/bart-large-mnli").
            local_dir (str): Directory to store or load the downloaded model files.
            force_download (bool): If True, force re-download even if 'local_dir' already exists.
        """
        # Check if the local folder already has the model files
        if force_download or not os.path.isdir(local_dir) or not os.listdir(local_dir):
            print(f"Downloading model '{model_id}' to '{local_dir}'... (progress bar below)")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                resume_download=True,
                force_download=force_download
            )
            print("Model download complete.\n")

        # Load tokenizer and model from local directory
        print("Loading tokenizer and model from local directory...")
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(local_dir)
        print("Model loaded successfully!\n")

        # Political categories to classify among
        self.candidate_labels = [
            "Left-Libertarian",
            "Right-Libertarian",
            "Left-Authoritarian",
            "Right-Authoritarian"
        ]

    def _build_hypothesis(self, label: str) -> str:
        """
        Build a hypothesis statement for a given label.
        E.g., label="Left-Libertarian" => "This text is about Left-Libertarian."
        """
        return f"This text is about {label}."

    def _get_label_entailment_score(
        self, premise: str, hypothesis: str
    ) -> float:
        """
        Given a premise (the user text) and a hypothesis (label text), return
        the entailment probability from the MNLI model output.
        """
        # Tokenize the premise-hypothesis pair
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            # logits shape: [batch_size, 3] for MNLI => (contradiction, neutral, entailment)

        # We want the probability of "entailment" (index 2)
        probs = F.softmax(logits, dim=-1)
        entailment_prob = probs[0, 2].item()
        return entailment_prob

    def categorize_subreddit(self, description: str, queries: List[str]) -> Tuple[str, Dict[str, float]]:
        """
        Uses the MNLI-based zero-shot approach to assign one of the four 
        political compass categories to the combined text of 'description' and 'queries'.

        Returns:
            category (str): The predicted political quadrant.
            scores (Dict[str, float]): Dictionary {label: entailment_score}.
        """
        # Combine description + queries into one text block
        description = (description or "").strip().lower()
        queries_joined = " ".join((q or "").strip().lower() for q in queries)
        premise = f"{description} {queries_joined}".strip()

        # Calculate entailment probability for each candidate label
        scores = {}
        for label in self.candidate_labels:
            hypothesis = self._build_hypothesis(label)
            entailment_score = self._get_label_entailment_score(premise, hypothesis)
            scores[label] = entailment_score

        # Pick the label with highest entailment score
        category = max(scores, key=scores.get)
        return category, scores


# ------------------- DEMO USAGE -------------------
if __name__ == "__main__":
    # Initialize the classifier
    classifier = ZeroShotPoliticalClassifier(
        model_id="facebook/bart-large-mnli",
        local_dir="./local_bart_mnli",
        force_download=False  # Change to True to re-download
    )

    # Example subreddit info
    description = "We discuss economic freedom, free markets, and individual rights."
    queries = ["economic freedom", "libertarian discourse"]

    # Classify into one of the four quadrants
    predicted_category, category_scores = classifier.categorize_subreddit(description, queries)
    print("Predicted Category:", predicted_category)
    print("Scores by label:", category_scores)