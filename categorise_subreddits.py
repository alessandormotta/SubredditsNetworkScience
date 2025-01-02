import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple


class SubredditZeroShotClassifier:
    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"  # Change to "cuda" for GPU support
    ):
        """
        Initialize a zero-shot classifier using pre-trained sentence-transformer embeddings.
        """
        self.device = torch.device(device)

        # Load model and tokenizer
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        print("Model loaded successfully!")

        # Define political categories and their descriptions
        self.candidate_labels = {
            "Left-Libertarian": "Personal freedom, social equality, progressive ideas",
            "Right-Libertarian": "Free markets, economic freedom, minimal government",
            "Left-Authoritarian": "Social equality with strong state control",
            "Right-Authoritarian": "Conservative views, national traditions, law and order"
        }

    def _embed_text(self, text: str) -> torch.Tensor:
        """
        Generate embeddings for input text using mean pooling.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
        return hidden_states.mean(dim=1)

    def categorize(self, description: str, queries: List[str]) -> Tuple[str, Dict[str, float]]:
        """
        Classify subreddit description and queries into political categories.
        """
        # Combine description and queries
        combined_text = f"{description.strip()} {' '.join(q.strip() for q in queries)}".lower()

        # Generate embeddings for the combined text
        subreddit_emb = self._embed_text(combined_text)

        # Compute similarity scores for each candidate label
        scores = {}
        for label, label_desc in self.candidate_labels.items():
            label_emb = self._embed_text(label_desc)
            similarity = F.cosine_similarity(subreddit_emb, label_emb, dim=-1).item()
            scores[label] = similarity

        # Find the label with the highest similarity score
        predicted_label = max(scores, key=scores.get)
        return predicted_label, scores


# Demo
if __name__ == "__main__":
    classifier = SubredditZeroShotClassifier(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"  # Use "cuda" for GPU
    )

    # Example subreddit description and saved queries
    description = "This subreddit discusses economic freedom, free markets, and minimal government intervention."
    queries = ["libertarian ideas", "economic freedom"]

    category, scores = classifier.categorize(description, queries)
    print("Predicted Category:", category)
    print("Similarity Scores:", scores)