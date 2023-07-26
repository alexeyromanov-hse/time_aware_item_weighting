import numpy as np
import torch
from torch import nn


class TopPopular(nn.Module):
    def __init__(
        self,
        item_num,
        user_num,
        corpus
    ):
        super().__init__()
        self.item_num = item_num
        self.user_num = user_num
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        
        self.items_cnts = np.zeros(self.item_num)
        self.pretrain(corpus)
        self.items_cnts = torch.tensor(self.items_cnts).to(self.device)
    
    def pretrain(self, corpus):
        for i in range(len(corpus.data["train"])):
            user_id = corpus.data["train"][i]["user_id"]
            click = corpus.book[user_id][corpus.data["train"][i]["click_order"]]
            self.items_cnts[click.item_id] += 1
        for i in range(len(corpus.data["dev"])):
            user_id = corpus.data["dev"][i]["user_id"]
            for order in corpus.data["dev"][i]["gt_click_order"]:
                click = corpus.book[user_id][order]
                self.items_cnts[click.item_id] += 1
    
    def forward(
            self,
            user_ids,
            item_ids,
            t=None,
            length=None,
            history_time=None,
            get_l2_reg=False
    ):
        if get_l2_reg:
            return self.items_cnts[item_ids], torch.tensor(0).to(self.device)
        else:
            return self.items_cnts[item_ids]
    
    def predict_for_user(
            self,
            user_id,
            t=None,
            length=None,
            history_time=None,
    ):
        return self.items_cnts
