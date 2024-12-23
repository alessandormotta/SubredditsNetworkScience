import re
from typing import List, Dict, Tuple

# A basic English stopwords list. You can expand this as needed.
CUSTOM_STOPWORDS = {
    'the', 'and', 'a', 'to', 'in', 'of', 'for', 'on', 'at', 'from', 
    'by', 'with', 'about', 'as', 'into', 'like', 'through', 'after',
    'over', 'between', 'out', 'up', 'down', 'off', 'above', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
    'so', 'than', 'too', 'very', 'can', 'will', 'just'
}

def preprocess_text(text: str) -> List[str]:
    """
    Preprocesses text by tokenizing, removing stopwords, and converting to lowercase.
    """
    text = text.lower()
    # Use regex to extract words (alphanumeric sequences)
    words = re.findall(r'\w+', text)
    # Filter out stopwords
    filtered_words = [word for word in words if word not in CUSTOM_STOPWORDS]
    return filtered_words

def categorize_subreddit(description: str, queries: List[str]) -> Tuple[str, Dict[str, int]]:
    """
    Categorizes a subreddit into one of the four political compass quadrants 
    based on its description and associated queries.
    """
    keywords = {
        'Left-Libertarian': ["anarchy", "progressive", "socialism", "justice", "activism", "equality"],
        'Right-Libertarian': ["libertarian", "freedom", "capitalism", "markets", "individual"],
        'Left-Authoritarian': ["state", "socialism", "authoritarian", "collectivism", "regulation"],
        'Right-Authoritarian': ["conservative", "nationalist", "patriot", "tradition", "law", "order"]
    }
    
    scores = {key: 0 for key in keywords.keys()}
    
    processed_description = preprocess_text(description or "")
    processed_queries = [q.lower() for q in (queries or [])]
    
    for quadrant, quadrant_keywords in keywords.items():
        scores[quadrant] += sum(1 for word in processed_description if word in quadrant_keywords)
        scores[quadrant] += sum(1 for word in processed_queries if word in quadrant_keywords)
    
    category = max(scores, key=scores.get)
    return category, scores