import torch
from torch import nn
from nbr.common import *


class RepurchaseModule(nn.Module):
    def __init__(
            self,
            item_num,
            avg_repeat_interval
    ):
        super().__init__()
        self.item_num = item_num
        self.avg_interval = avg_repeat_interval / TIME_SCALAR
        self.eps, self.inf = EPS, INF

        self.global_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32, requires_grad=True))
        self.register_parameter(name='global_alpha', param=self.global_alpha)
        self.item_alpha = nn.Parameter(torch.zeros(self.item_num, dtype=torch.float32, requires_grad=True))
        self.register_parameter(name='item_alpha', param=self.item_alpha)
        self.item_pi = nn.Parameter(torch.normal(0.5, 0.01, size=(self.item_num,), dtype=torch.float32, requires_grad=True))
        self.register_parameter(name='item_pi', param=self.item_pi)
        self.item_mu = nn.Parameter(torch.normal(self.avg_interval, 0.01, size=(self.item_num,), dtype=torch.float32, requires_grad=True))
        self.register_parameter(name='item_mu', param=self.item_mu)
        self.item_beta = nn.Parameter(torch.ones(self.item_num, dtype=torch.float32, requires_grad=True))
        self.register_parameter(name='item_beta', param=self.item_beta)
        self.item_sigma = nn.Parameter(torch.ones(self.item_num, dtype=torch.float32, requires_grad=True))
        self.register_parameter(name='item_sigma', param=self.item_sigma)
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
    
    def forward(
            self,
            user_ids,
            item_ids,
            t=None,
            length=None,
            history_time=None,
            get_l2_reg=False
    ):
        ranking_score = torch.tensor([0] * user_ids.shape[0], dtype=torch.float64).to(self.device)
        if history_time is not None:
            alpha = torch.clamp(self.item_alpha[item_ids] + self.global_alpha, min=0., max=self.inf)
            beta = torch.clamp(self.item_beta[item_ids], min=self.eps, max=self.inf)
            pi = torch.clamp(self.item_pi[item_ids], min=0., max=1.)
            mu = self.item_mu[item_ids]
            sigma = torch.clamp(self.item_sigma[item_ids], min=self.eps, max=self.inf)

            delta_t = torch.unsqueeze(t, -1) - history_time
            delta_t = torch.clamp(delta_t, min=self.eps, max=self.inf)
            mask = torch.arange(0, length.max(), 1).to(self.device)
            mask = mask < torch.unsqueeze(length, dim=-1)
            exp_dist = torch.distributions.exponential.Exponential(rate=1 / torch.unsqueeze(beta, dim=-1))
            norm_dist = torch.distributions.normal.Normal(loc=torch.unsqueeze(mu, dim=-1), scale=torch.unsqueeze(sigma, dim=-1))
            sum_k_t = ((1. - torch.unsqueeze(pi, dim=-1)) * (torch.e ** exp_dist.log_prob(delta_t)) + torch.unsqueeze(pi, dim=-1) * (torch.e ** norm_dist.log_prob(delta_t))).sum(dim=1)
            
            ranking_score = ranking_score + sum_k_t

        if get_l2_reg:
            return ranking_score, torch.tensor(0).to(self.device)
        else:
            return ranking_score
    
    def predict_for_user(
            self,
            user_id,
            t=None,
            length=None,
            history_time=None,
    ):
        ranking_score = torch.tensor([0] * self.item_num, dtype=torch.float64).to(self.device)
        if history_time is not None:
            item_ids = torch.arange(self.item_num).to(self.device)
            
            alpha = torch.clamp(self.item_alpha[item_ids] + self.global_alpha, min=0., max=self.inf)
            beta = torch.clamp(self.item_beta[item_ids], min=self.eps, max=self.inf)
            pi = torch.clamp(self.item_pi[item_ids], min=0., max=1.)
            mu = self.item_mu[item_ids]
            sigma = torch.clamp(self.item_sigma[item_ids], min=self.eps, max=self.inf)

            delta_t = torch.unsqueeze(t, -1) - history_time
            delta_t = torch.clamp(delta_t, min=self.eps, max=self.inf)
            mask = torch.arange(0, length.max(), 1).to(self.device)
            mask = mask < torch.unsqueeze(length, dim=-1)
            exp_dist = torch.distributions.exponential.Exponential(rate=1 / torch.unsqueeze(beta, dim=-1))
            norm_dist = torch.distributions.normal.Normal(loc=torch.unsqueeze(mu, dim=-1), scale=torch.unsqueeze(sigma, dim=-1))
            sum_k_t = ((1. - torch.unsqueeze(pi, dim=-1)) * (torch.e ** exp_dist.log_prob(delta_t)) + torch.unsqueeze(pi, dim=-1) * (torch.e ** norm_dist.log_prob(delta_t))).sum(dim=1)
            
            ranking_score = ranking_score + sum_k_t

        return ranking_score
