from nbr.dataset import NBRDataset, train_collate_fn, eval_collate_fn
from nbr.common import get_precision, get_recall, get_ndcg
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys


class NBRTrainer:
    def __init__(self, corpus, max_epochs, topk, early_stop_num):
        self.corpus = corpus
        self.train_dataset = NBRDataset(corpus=self.corpus, mode="train")
        self.dev_dataset = NBRDataset(corpus=self.corpus, mode="dev")
        self.test_dataset = NBRDataset(corpus=self.corpus, mode="test")
        self.max_epochs = max_epochs
        self.topk = topk
        self.early_stop_num = early_stop_num
        self.device = 'cpu'
        self.dev_dataloader = DataLoader(self.dev_dataset, batch_size=1, collate_fn=eval_collate_fn)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, collate_fn=eval_collate_fn)
    
    def init_hyperparams(self, **params):
        for name, param in params.items():
            setattr(self, name, param)
        if hasattr(self, 'batch_size'):
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=train_collate_fn)
        if hasattr(self, 'lr'):
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.criterion = nn.LogSigmoid()
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        if hasattr(self, 'model'):
            self.model = self.model.to(self.device)
        self.best_score = None
    
    def train(self, evaluation_flg=True):
        best_metric = 0.
        bad_epoch_num = 0

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch + 1}:")
            sys.stdout.flush()
            self.run_epoch()
            print()
            
            if evaluation_flg:
                print("Evaluation (dev):")
                sys.stdout.flush()
                metrics = self.evaluate(mode="dev")
                print("\n", metrics)
                sys.stdout.flush()

                if metrics["ndcg"] >= best_metric:
                    torch.save(self.model.state_dict(), "best_checkpoint.pth")
                    best_metric = metrics["ndcg"]
                    bad_epoch_num = 0
                else:
                    bad_epoch_num += 1
                
                if bad_epoch_num >= self.early_stop_num:
                    break
            else:
                torch.save(self.model.state_dict(), "best_checkpoint.pth")
        
        with open("best_checkpoint.pth", "rb") as f:
            checkpoint = torch.load(f)
        self.model.load_state_dict(checkpoint)
        self.best_score = best_metric
        return self.model

    def run_epoch(self):
        self.model.train()

        loss = None
        progress_bar = tqdm(self.train_dataloader)
        for batch in progress_bar:
            if loss is not None:
                progress_bar.set_description(f"Batch loss = {np.round(loss.item(), 6)}")

            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.model.zero_grad()

            positive_items_scores, reg_loss = self.model(
                user_ids=batch["user_id"],
                item_ids=batch["item_id"],
                t=batch["t"],
                length=batch["length"],
                history_time=batch["history_time"],
                get_l2_reg=True
            )
            negative_items_scores = self.model(
                user_ids=batch["user_id"],
                item_ids=batch["neg_sample"],
                get_l2_reg=False
            )

            loss = -torch.mean(self.criterion(positive_items_scores - negative_items_scores)) + self.l2_reg_coef * reg_loss
            loss.backward()
            self.opt.step()
    
    def evaluate(self, mode="dev"):
        if mode == "dev":
            test_dataloader = self.dev_dataloader
        else:
            test_dataloader = self.test_dataloader

        precisions, recalls, ndcgs = [], [], []
        self.model.eval()

        progress_bar = tqdm(test_dataloader)
        for batch in progress_bar:
            proper_items = batch["proper_items"]
            batch = {k: v.to(self.device) for k, v in batch.items() if k != "proper_items"}

            items_scores = self.model.predict_for_user(
                user_id=batch["user_id"][0],
                t=batch["t"],
                length=batch["length"],
                history_time=batch["history_time"],
            )
            items_scores = items_scores.view(-1, self.corpus.n_items)
            top_items = torch.topk(items_scores, k=self.topk, dim=1).indices

            top_items = top_items.cpu().detach().numpy()

            for i in range(top_items.shape[0]):
                precision = get_precision(top_items[i], proper_items[i])
                recall = get_recall(top_items[i], proper_items[i])
                ndcg = get_ndcg(top_items[i], proper_items[i])

                precisions.append(precision)
                recalls.append(recall)
                ndcgs.append(ndcg)
        
        precision = np.array(precisions).mean()
        recall = np.array(recalls).mean()
        ndcg = np.array(ndcgs).mean()

        return {
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg
        }
    
    def evaluate_fully(self, mode="dev", topk=[10,]):
        if mode == "dev":
            test_dataloader = self.dev_dataloader
        else:
            test_dataloader = self.test_dataloader

        precisions, recalls, ndcgs = {f"@{k}": [] for k in topk}, {f"@{k}": [] for k in topk}, {f"@{k}": [] for k in topk}
        self.model.eval()

        progress_bar = tqdm(test_dataloader)
        for batch in progress_bar:
            proper_items = batch["proper_items"]
            batch = {k: v.to(self.device) for k, v in batch.items() if k != "proper_items"}

            items_scores = self.model.predict_for_user(
                user_id=batch["user_id"][0],
                t=batch["t"],
                length=batch["length"],
                history_time=batch["history_time"],
            )
            items_scores = items_scores.view(-1, self.corpus.n_items)
            
            for k in topk:
                top_items = torch.topk(items_scores, k=k, dim=1).indices
                top_items = top_items.cpu().detach().numpy()
                for i in range(top_items.shape[0]):
                    precision = get_precision(top_items[i], proper_items[i])
                    recall = get_recall(top_items[i], proper_items[i])
                    ndcg = get_ndcg(top_items[i], proper_items[i])

                    precisions[f"@{k}"].append(precision)
                    recalls[f"@{k}"].append(recall)
                    ndcgs[f"@{k}"].append(ndcg)
        
        for k in topk:
            precisions[f"@{k}"] = np.array(precisions[f"@{k}"]).mean()
            recalls[f"@{k}"] = np.array(recalls[f"@{k}"]).mean()
            ndcgs[f"@{k}"] = np.array(ndcgs[f"@{k}"]).mean()

        return {
            "precision": precisions,
            "recall": recalls,
            "ndcg": ndcgs
        }
    
    def get_predictions(self, mode="dev"):
        if mode == "dev":
            test_dataloader = self.dev_dataloader
        else:
            test_dataloader = self.test_dataloader

        self.model.eval()
        
        predictions = np.zeros((self.corpus.n_users, self.corpus.n_items))

        progress_bar = tqdm(test_dataloader)
        for batch in progress_bar:
            proper_items = batch["proper_items"]
            batch = {k: v.to(self.device) for k, v in batch.items() if k != "proper_items"}

            items_scores = self.model.predict_for_user(
                user_id=batch["user_id"][0],
                t=batch["t"],
                length=batch["length"],
                history_time=batch["history_time"],
            )
            items_scores = items_scores.cpu().detach().numpy()
            predictions[batch["user_id"][0].item()] = items_scores
        
        return predictions
