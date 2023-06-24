import json
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F
import faiss
import streamlit as st


class Retriever:
    def __init__(self, faiss_path, idx_to_metadata_path):
        self.faiss_index = faiss.read_index(faiss_path)

        with open(idx_to_metadata_path) as f:
            self.idx_to_metadata = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/all-MiniLM-L12-v2')
        self.model = AutoModel.from_pretrained(
            'sentence-transformers/all-MiniLM-L12-v2')

    def get_final_answer(self, sentence):
        distances, indices = self.query_database(sentence)

        response = []
        if len(distances) == 0:
            return None
        else:
            for score, index in zip(distances, indices):
                response.append(
                    {**self.idx_to_metadata[index], 'score': score})

        set_answer, answer1, answer2, similar_items = self.get_answer_candidates(
            response)
        matching_episodes = answer1.drop_duplicates(['season', 'episode'])
        matching_episodes = matching_episodes.iloc[:6]

        return {'correct': matching_episodes, 'fallback': set_answer, 'similar': similar_items}

    def get_answer_candidates(self, response):
        response = pd.DataFrame(response)
        response.drop_duplicates(['scene summary'], keep='first', inplace=True)

        set_answer = response.iloc[:1]
        set_answer = set_answer[set_answer['score'] > 0.50]

        answer1 = response[response['score'] >
                           0.67].drop_duplicates(['season', 'episode'])

        answer2 = get_similar_multiple_matches(response, thresh=0.55)

        similar_items = response[response['score'] > 0.48].drop_duplicates(
            ['season', 'episode'], keep='first').iloc[:8]
        similar_items = pd.concat(
            [answer2, similar_items], axis=0).drop_duplicates(['season', 'episode'])
        similar_items = similar_items[~similar_items.index.isin(answer1.index)]

        return [set_answer, answer1, similar_items, similar_items]

    def query_database(self, sentence):
        query_vector = self.encode([sentence])
        distances, indices = self.faiss_index.search(query_vector, k=20)
        distances, indices = distances[0, :], indices[0, :]

        indices = indices[distances > 0.2]
        distances = distances[distances > 0.2]

        return distances, indices

    def encode(self, sentences):
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(
                -1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        assert isinstance(sentences, list)

        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = mean_pooling(
            model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.detach().cpu().numpy()


def get_similar_multiple_matches(response, thresh=0.55):
    temp = response[response['score'] > thresh]
    temp2 = temp.groupby(['season', 'episode'])['score'].apply(len)

    temp2 = temp2[temp2 > 1]

    q = np.array([False] * len(temp))
    for pair in temp2.to_frame().index:
        q = q | ((temp.season == pair[0]) & (temp.episode == pair[1]))

    answer2 = temp[q].drop_duplicates(['season', 'episode'])
    return answer2
