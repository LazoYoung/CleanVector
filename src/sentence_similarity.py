from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F


class SentenceSimilarity:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeds = model_output[0]
        input_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        return torch.sum(token_embeds * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

    def embedding(self, sentences):
        encoded_input = self.tokenizer(text=sentences, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def compute_score(self, source, target):
        """
        Compute similarity scores for semantic search

        @param source: The source sentence (embedding or string)
        @param target: Sentences to compare to (embeddings or strings)
        @return: Similarity scores of each target sentence
        """
        if isinstance(source, str):
            source = self.embedding(source)[0]

        if isinstance(target, str) or isinstance(target, list):
            target_embeds = self.embedding(target)
        else:
            target_embeds = target

        source = source.squeeze(dim=0)
        scores = []

        for target in target_embeds:
            score = torch.dot(source, target)
            scores.append(score.item())

        return scores if len(scores) > 1 else scores[0]


def test():
    sen_sim = SentenceSimilarity()
    source_sentence = "That is a happy person"
    target_sentences = [
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day",
    ]
    embeddings = sen_sim.embedding(target_sentences)
    scores = sen_sim.compute_score(source_sentence, target_sentences)
    print(embeddings)
    print(scores)


if __name__ == "__main__":
    test()
