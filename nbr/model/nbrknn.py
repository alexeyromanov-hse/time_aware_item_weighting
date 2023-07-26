from .annoy import Annoy
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import sys


class NBRKNN(nn.Module):
    def __init__(
        self,
        item_num,
        user_num,
        nearest_neighbors_num,
        alpha,
        user_emb
    ):
        super().__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.nearest_neighbors_num = nearest_neighbors_num
        self.alpha = alpha
        self.user_emb = user_emb
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        
        self.model = Annoy()
        self.model.set_query_arguments(self.model._n_trees * self.nearest_neighbors_num // 2)
        self.model.fit(self.user_emb)
        self.neighbors_emb = torch.tensor(self.user_emb).to(self.device)
        self.user_emb = torch.tensor(self.user_emb).to(self.device)

        self.precalc_indices = None
    
    def set_emb(self, user_emb):
        self.user_emb = torch.tensor(user_emb).to(self.device)
    
    def reset(self):
        self.user_emb = None
        self.neighbors_emb = None
        self.model.reset()
        self.precalc_indices = None
        self.model = None
    
    def precalculate(self):
        print("KNN precalculating...")
        sys.stdout.flush()
        self.precalc_indices = np.zeros((self.user_num, self.nearest_neighbors_num), dtype=np.int64)
        for user_id in tqdm(range(self.user_num)):
            user_emb = self.user_emb[user_id].cpu().detach().numpy()
            user_indices = np.array(self.model.query(v=user_emb, n=self.nearest_neighbors_num), dtype=np.int64)
            self.precalc_indices[user_id, :] = user_indices
        self.precalc_indices = torch.tensor(self.precalc_indices).to(self.device)
    
    def forward(
            self,
            user_ids,
            item_ids,
            t=None,
            length=None,
            history_time=None,
            get_l2_reg=False
    ):
        user_emb = self.user_emb[user_ids].reshape((-1, self.item_num))
        if self.precalc_indices is None:
            indices = np.zeros((user_emb.shape[0], self.nearest_neighbors_num), dtype=np.int64)
            for i, emb in enumerate(user_emb.cpu().detach().numpy()):
                user_indices = np.array(self.model.query(v=emb, n=self.nearest_neighbors_num), dtype=np.int64)
                indices[i, :] = user_indices
            indices = torch.tensor(indices).to(self.device)
        else:
            indices = self.precalc_indices[user_ids].reshape((-1, self.nearest_neighbors_num))
        predictions = self.alpha * user_emb + (1 - self.alpha) * self.neighbors_emb[indices].mean(axis=1)
        predictions = predictions[torch.arange(predictions.shape[0]).to(self.device), item_ids]

        if get_l2_reg:
            return predictions, torch.tensor(0).to(self.device)
        else:
            return predictions
    
    def predict_for_user(
            self,
            user_id,
            t=None,
            length=None,
            history_time=None,
    ):
        user_emb = self.user_emb[user_id].reshape((-1, self.item_num))
        if self.precalc_indices is None:
            indices = np.zeros((user_emb.shape[0], self.nearest_neighbors_num), dtype=np.int64)
            for i, emb in enumerate(user_emb.cpu().detach().numpy()):
                user_indices = np.array(self.model.query(v=emb, n=self.nearest_neighbors_num), dtype=np.int64)
                indices[i, :] = user_indices
            indices = torch.tensor(indices).to(self.device)
        else:
            indices = self.precalc_indices[user_id].reshape((-1, self.nearest_neighbors_num))
        predictions = (self.alpha * user_emb + (1 - self.alpha) * self.neighbors_emb[indices].mean(axis=1)).reshape(-1)
        return predictions
