import torch
from torch import nn
import scipy.sparse as sps
import numpy as np
import similaripy
import sys
from tqdm import tqdm


class UPCF(nn.Module):
    def __init__(
        self,
        user_num,
        item_num,
        recency,
        q,
        alpha,
        nearest_neighbors_num,
        corpus
    ):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.recency = recency
        self.q = q
        self.alpha = alpha
        self.nearest_neighbors_num = nearest_neighbors_num if nearest_neighbors_num is not None else self.user_num
        self.corpus = corpus
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        
        self.calculate_embeddings(corpus)
        
        self._similarity_matrix = similaripy.asymmetric_cosine(
            self.user_emb, alpha=self.alpha, k=self.nearest_neighbors_num, verbose=False
        ).power(self.q)
    
    def calculate_embeddings(self, corpus):
        self.user_emb = []
        for _ in range(self.user_num):
            self.user_emb.append({})
        
        for i in range(len(corpus.data["train"])):
            user_id = corpus.data["train"][i]["user_id"]
            click = corpus.book[user_id][corpus.data["train"][i]["click_order"]]
            if click.time not in self.user_emb[user_id]:
                self.user_emb[user_id][click.time] = set()
            self.user_emb[user_id][click.time].add(click.item_id)
               
        print("UPCF fitting...")
        sys.stdout.flush()
        self.user_preferences_emb = np.zeros((self.user_num, self.item_num))
        for i in tqdm(range(self.user_num)):
            user_baskets = self.user_emb[i]
            timestamps = list(user_baskets.keys())
            timestamps.sort()
            timestamps = timestamps[-self.recency:]
            for j in range(len(timestamps)):
                self.user_preferences_emb[i, list(user_baskets[timestamps[j]])] += 1 / len(timestamps)
        self.user_emb = sps.csr_matrix(1 * (self.user_preferences_emb > 0))
        self.user_preferences_emb = sps.csr_matrix(self.user_preferences_emb)
    
    def predict_for_user(
            self,
            user_id,
            t=None,
            length=None,
            history_time=None
    ):
        scores = similaripy.dot_product(
            self._similarity_matrix,
            self.user_preferences_emb,
            target_rows=[user_id.cpu().item()],
            k=self.item_num,
            verbose=False
        ).getrow(user_id.cpu().item()).toarray()[0]
        scores = torch.tensor(scores).to(self.device)
        return scores
