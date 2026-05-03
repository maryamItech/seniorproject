"""
BM25 keyword search for exact legal term matching.
"""

from rank_bm25 import BM25Okapi
import re


# def tokenize_arabic(text: str) -> list[str]:
#     """Simple tokenizer for Arabic: split on whitespace and punctuation."""
#     # Remove punctuation, split on whitespace
#     text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
#     return [t for t in text.split() if t]
def tokenize_arabic(text: str) -> list[str]:
    """Simple tokenizer for Arabic: split on whitespace and punctuation."""
    # Convert text to lowercase (for foreign words if present) and strip whitespaces
    text = text.lower().strip()

    # 1. Replace special characters with spaces (while preserving Arabic letters and numbers)
    # Added Arabic unicode range \u0600-\u06FF and both Arabic and Hindi-Indic digits
    text = re.sub(r"[^\u0600-\u06FF0-9٠-٩]", " ", text)

    # 2. Tokenize based on whitespace
    tokens = [t for t in text.split() if len(t) > 1] # Ignore single characters to increase search precision
    return tokens

class BM25Index:
    """BM25 index for Arabic legal chunks."""

    def __init__(self):
        self.index: BM25Okapi | None = None
        self.chunks: list[dict] = []
        self.tokenized: list[list[str]] = []

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        paired = []
        for c in chunks:
            tokens = tokenize_arabic(c.get("text", ""))
            if tokens:
                paired.append((c, tokens))
        self.chunks = [item[0] for item in paired]
        self.tokenized = [item[1] for item in paired]
        corpus = self.tokenized
        if not corpus:
            self.index = None
            return
        self.index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        """
        Search and return top_k chunks with scores.

        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None:
            return []
        if not self.chunks:
            return []
        tokens = tokenize_arabic(query)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        indices = scores.argsort()[::-1][:top_k]
        results = []
        for i in indices:
            if i < len(self.chunks) and scores[i] > 0:
                results.append((self.chunks[i], float(scores[i])))
        return results
