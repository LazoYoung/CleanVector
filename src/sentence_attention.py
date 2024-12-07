from typing import List, Dict, Tuple

import nltk

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


class AttentionScoreComputer:
    def __init__(self):
        """
        Initialize the Attention Score Computer with various scoring criteria.
        """
        # Predefined weights for different scoring criteria
        self.pos_weights = {
            'NN': 1.0,  # Noun
            'NNS': 0.9,  # Plural Noun
            'NNP': 1.1,  # Proper Noun
            'NNPS': 1.0,  # Plural Proper Noun
            'VB': 0.7,  # Verb, base form
            'VBD': 0.6,  # Verb, past tense
            'VBG': 0.5,  # Verb, gerund/present participle
            'VBN': 0.5,  # Verb, past participle
            'VBP': 0.6,  # Verb, non-3rd person singular present
            'VBZ': 0.6,  # Verb, 3rd person singular present
            'JJ': 0.4,  # Adjective
            'RB': 0.3  # Adverb
        }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize the input text into words.

        Args:
            text (str): Input text to tokenize

        Returns:
            List[str]: List of tokens
        """
        return nltk.word_tokenize(text)

    @staticmethod
    def pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Perform Part-of-Speech tagging on tokens.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            List[Tuple[str, str]]: List of (token, POS tag) tuples
        """
        return nltk.pos_tag(tokens)

    @staticmethod
    def compute_position_score(tokens: List[str], token_index: int) -> float:
        """
        Compute positional importance score for a token.
        Gives higher scores to tokens in important positions.

        Args:
            tokens (List[str]): List of tokens
            token_index (int): Index of the current token

        Returns:
            float: Positional score
        """
        total_tokens = len(tokens)

        # Stronger weight for tokens near the beginning or end of the sentence
        if token_index < total_tokens // 4:
            return 1.0
        elif token_index >= total_tokens * 3 // 4:
            return 0.8
        else:
            return 0.5

    def compute_token_attention_score(self, text: str) -> Dict[str, float]:
        """
        Compute attention scores for each token in the text.

        Args:
            text (str): Input text to analyze

        Returns:
            Dict[str, float]: Dictionary of tokens and their attention scores
        """
        # Tokenize the text
        tokens = self.tokenize(text)

        # Get POS tags
        pos_tags = self.pos_tag(tokens)

        # Compute attention scores
        attention_scores = {}
        for i, (token, pos) in enumerate(pos_tags):
            # Base score from POS tag
            pos_score = self.pos_weights.get(pos, 0.1)

            # Position score
            position_score = self.compute_position_score(tokens, i)

            # Combine scores (can be adjusted)
            attention_score = pos_score * position_score

            attention_scores[token] = attention_score

        return attention_scores

    def get_most_relevant_tokens(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get the most relevant tokens based on attention scores.

        Args:
            text (str): Input text to analyze
            top_k (int, optional): Number of top tokens to return. Defaults to 3.

        Returns:
            List[Tuple[str, float]]: List of (token, score) tuples sorted by score
        """
        attention_scores = self.compute_token_attention_score(text)

        # Sort tokens by attention score in descending order
        sorted_tokens = sorted(
            attention_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_tokens[:top_k]


# Example usage
def main():
    # Create an instance of the attention score computer
    attention_computer = AttentionScoreComputer()

    # Example sentences
    sentences = [
        # "The quick brown fox jumps over the lazy dog.",
        # "Machine learning is transforming the field of artificial intelligence.",
        # "Python programming provides powerful tools for data analysis and visualization.",
        "a brown cat face",
        "stop sign",
        "dog"
    ]

    # Analyze each sentence
    for sentence in sentences:
        print(f"\nSentence: {sentence}")

        # Compute full attention scores
        full_scores = attention_computer.compute_token_attention_score(sentence)
        print("\nFull Token Attention Scores:")
        for token, score in sorted(full_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{token}: {score:.4f}")

        # Get most relevant tokens
        print("\nMost Relevant Tokens:")
        top_tokens = attention_computer.get_most_relevant_tokens(sentence)
        for token, score in top_tokens:
            print(f"{token}: {score:.4f}")


if __name__ == "__main__":
    main()
