import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import pandas as pd


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

        # Define new political categories and their descriptions
        self.candidate_labels = {
            "Left": "Supports social equality, progressive policies, and public welfare.",
            "Right": "Advocates for free markets, individual responsibility, and traditional values.",
            "Extreme Left": "Radical focus on overthrowing systemic inequality, collectivism, and revolution.",
            "Extreme Right": "Advocates for authoritarian nationalism, ultraconservatism, and exclusionary policies.",
            "Center": "Balances between progressive and conservative values, focusing on pragmatic solutions.",
            "Neutral": "Avoids political ideology, focuses on general or non-political content."
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

    # Load subreddit data
    subreddit_data = pd.read_csv("subreddit_with_query_types.csv")

    # Extract the required columns
    subreddits = subreddit_data[['id', 'name', 'description']]
    classified_results = []

    # Classify each subreddit
    for _, row in subreddits.iterrows():
        subreddit_id = row['id']
        description = row['description'] if isinstance(row['description'], str) else ""
        # Use the classifier to determine the category
        category, _ = classifier.categorize(description, [])
        classified_results.append({'id': subreddit_id, 'category': category})

    # Create a DataFrame from the results
    classified_df = pd.DataFrame(classified_results)

    # Save to a new CSV file
    output_file_path = 'subreddit_categories_ext.csv'
    classified_df.to_csv(output_file_path, index=False)

    print(f"Results saved to: {output_file_path}")