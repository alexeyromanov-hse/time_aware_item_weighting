import torch
from torch import nn


class BPR(nn.Module):
    def __init__(self, emb_size, user_num, item_num, click_num):
        super().__init__()
        self.emb_size = emb_size
        self.user_num = user_num
        self.item_num = item_num
        self.click_num = click_num
        self.user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.item_emb = nn.Embedding(self.item_num, self.emb_size)
        self.user_emb.weight.data.normal_(mean=0.0, std=0.01)
        self.item_emb.weight.data.normal_(mean=0.0, std=0.01)
        self.global_bias = nn.Parameter(torch.tensor(self.click_num / self.user_num / self.item_num, dtype=torch.float32, requires_grad=True))
        self.register_parameter(name='global_bias', param=self.global_bias)
    
    def forward(
            self,
            user_ids,
            item_ids,
            t=None,
            length=None,
            history_time=None,
            get_l2_reg=False
    ):
        user_embs = self.user_emb(user_ids)
        item_embs = self.item_emb(item_ids)
        scores = (user_embs * item_embs).sum(dim=1) + self.global_bias
        if get_l2_reg:
            l2_reg = (torch.sum(user_embs ** 2) / 2) + (torch.sum(item_embs ** 2) / 2)
            return scores, l2_reg
        else:
            return scores
    
    def predict_for_user(
            self,
            user_id,
            t=None,
            length=None,
            history_time=None,
    ):
        user_emb = self.user_emb(user_id)
        scores = (self.item_emb.weight * user_emb).sum(dim=1) + self.global_bias
        return scores
