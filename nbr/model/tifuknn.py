from .nbrknn import NBRKNN
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import sys


class TIFUKNN(nn.Module):
    def __init__(
        self,
        item_num,
        user_num,
        group_num,
        within_decay_rate,
        group_decay_rate,
        nearest_neighbors_num,
        alpha,
        corpus
    ):
        super().__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.group_num = group_num
        self.within_decay_rate = within_decay_rate
        self.group_decay_rate = group_decay_rate
        self.nearest_neighbors_num = nearest_neighbors_num
        self.alpha = alpha
        
        self.user_emb = []
        for _ in range(self.user_num):
            self.user_emb.append({})
        self.calculate_embeddings(corpus)
        
        self.predicter = NBRKNN(
            self.item_num,
            self.user_num,
            self.nearest_neighbors_num,
            self.alpha,
            self.user_emb
        )

    def reset(self):
        self.user_emb = []
        for _ in range(self.user_num):
            self.user_emb.append({})
        self.predicter.reset()
    
    def calculate_embeddings(self, corpus):
        for i in range(len(corpus.data["train"])):
            user_id = corpus.data["train"][i]["user_id"]
            click = corpus.book[user_id][corpus.data["train"][i]["click_order"]]
            if click.time not in self.user_emb[user_id]:
                self.user_emb[user_id][click.time] = set()
            self.user_emb[user_id][click.time].add(click.item_id)
        
        print("TIFUKNN fitting...")
        sys.stdout.flush()
        for i in tqdm(range(self.user_num)):
            user_baskets = self.user_emb[i]
            timestamps = list(user_baskets.keys())
            timestamps.sort()
            res = np.zeros((self.item_num, len(timestamps)))
            for j in range(len(timestamps)):
                processed_data = np.zeros(self.item_num)
                processed_data[list(user_baskets[timestamps[j]])] = 1
                res[:, j] = processed_data * np.power(self.within_decay_rate, len(timestamps) - j - 1)
            if res.shape[1] >= self.group_num:
                group_size = int(res.shape[1] / self.group_num)
                extra_baskets_num = res.shape[1] % group_size
                group_num = self.group_num
            else:
                group_size = 1
                extra_baskets_num = 0
                group_num = res.shape[1]
            grouped_res = np.zeros((self.item_num, group_num))
            tmp = 0
            for j in range(group_num):
                if j < group_num - extra_baskets_num:
                    grouped_res[:, j] = res[:, group_size * j: group_size * (j + 1)].mean(axis=1) * np.power(self.group_decay_rate, group_num - 1 - j)
                else:
                    grouped_res[:, j] = res[:, group_size * j + tmp: group_size * (j + 1) + tmp + 1].mean(axis=1) * np.power(self.group_decay_rate, group_num - 1 - j)
                    tmp += 1
            self.user_emb[i] = grouped_res.mean(axis=1)
        self.user_emb = np.array(self.user_emb)
    
    def precalculate(self):
        self.predicter.precalculate()
    
    def forward(
            self,
            user_ids,
            item_ids,
            t=None,
            length=None,
            history_time=None,
            get_l2_reg=False
    ):
        return self.predicter(
            user_ids,
            item_ids,
            t,
            length,
            history_time,
            get_l2_reg
        )
    
    def predict_for_user(
            self,
            user_id,
            t=None,
            length=None,
            history_time=None
    ):
        return self.predicter.predict_for_user(
            user_id,
            t,
            length,
            history_time
        )
