from .nbrknn import NBRKNN
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import sys


class TIFUKNNTimeDays(nn.Module):
    def __init__(
        self,
        item_num,
        user_num,
        group_size_days,
        within_decay_rate,
        group_decay_rate,
        nearest_neighbors_num,
        alpha,
        use_log,
        corpus
    ):
        super().__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.group_size_days = group_size_days
        self.within_decay_rate = within_decay_rate
        self.group_decay_rate = group_decay_rate
        self.nearest_neighbors_num = nearest_neighbors_num
        self.alpha = alpha
        self.use_log = use_log
        
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
        
        print("TIFUKNNTimeDays fitting...")
        sys.stdout.flush()
        for i in tqdm(range(self.user_num)):
            user_baskets = self.user_emb[i]
            timestamps = list(user_baskets.keys())
            timestamps.sort()
            res = np.zeros((self.item_num, len(timestamps)))
            groups = np.zeros(len(timestamps))
            max_timestamp = pd.to_datetime(timestamps[-1], unit='s')
            for j in range(len(timestamps)):
                global_days_diff = (max_timestamp - pd.to_datetime(timestamps[j], unit='s')) / pd.Timedelta(days=1)
                group = int(global_days_diff // self.group_size_days)
                group_max_timestamp = max_timestamp - (group * pd.Timedelta(days=self.group_size_days))
                local_days_diff = (group_max_timestamp - pd.to_datetime(timestamps[j], unit='s')) / pd.Timedelta(days=1)
                
                processed_data = np.zeros(self.item_num)
                processed_data[list(user_baskets[timestamps[j]])] = 1
                if self.use_log:
                    res[:, j] = processed_data * np.power(self.group_decay_rate, group) * np.power(self.within_decay_rate, np.log(1 + local_days_diff))
                else:
                    res[:, j] = processed_data * np.power(self.group_decay_rate, group) * np.power(self.within_decay_rate, local_days_diff)
                groups[j] = group
            for j in range(len(timestamps)):
                global_days_diff = (max_timestamp - pd.to_datetime(timestamps[j], unit='s')) / pd.Timedelta(days=1)
                group = int(global_days_diff // self.group_size_days)
                group_size = (groups == group).sum()
                res[:, j] = res[:, j] / (group_size * (np.unique(groups).shape[0]))
            self.user_emb[i] = res.sum(axis=1)
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
            history_time=None,
    ):
        return self.predicter.predict_for_user(
            user_id,
            t,
            length,
            history_time
        )
