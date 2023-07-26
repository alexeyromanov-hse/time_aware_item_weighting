from torch.utils.data import Dataset
from nbr.common import *
import numpy as np
import torch
from tqdm import tqdm


class NBRDataset(Dataset):
    def __init__(self, corpus, mode):
        if mode not in ["train", "dev", "test"]:
            raise Exception("Possible values of 'mode' are 'train', 'dev', 'test'.")
        self.corpus = corpus
        self.mode = mode
        print(f"{self.mode} dataset preparing...")
        if self.mode == "train":
            consumed_items = [[click.item_id for click in corpus.book[user_id]] for user_id in range(corpus.n_users)]
            self.neg_samples = [np.array(list(set(np.arange(self.corpus.n_items)).difference(set(consumed_items[user_id])))) for user_id in tqdm(range(corpus.n_users))]
        if self.mode in ["dev", "test"]:
            self.user_proper_items = []
            self.ts = []
            self.history_times = []
            for index in tqdm(range(len(self.corpus.data[self.mode]))):
                user_id = self.corpus.data[self.mode][index]["user_id"]
                orders = self.corpus.data[self.mode][index]["gt_click_order"]
                proper_items = np.array([self.corpus.book[user_id][order].item_id for order in orders])
                self.user_proper_items.append(proper_items)
                t = self.corpus.book[user_id][orders[0]].time / TIME_SCALAR
                self.ts.append(t)
                self.history_times.append({})
                for click in self.corpus.book[user_id]:
                    if click.time >= self.corpus.book[user_id][orders[0]].time:
                        break
                    if click.item_id not in self.history_times[index]:
                        self.history_times[index][click.item_id] = np.array([])
                    self.history_times[index][click.item_id] = np.append(self.history_times[index][click.item_id], click.time / TIME_SCALAR)

    def __len__(self):
        return len(self.corpus.data[self.mode])
    
    def __getitem__(self, index):
        if self.mode == "train":
            data = self.corpus.data[self.mode][index]
            user_id = data["user_id"]
            click = self.corpus.book[user_id][data["click_order"]]
            item_id = click.item_id
            t = click.time / TIME_SCALAR
            history_time = np.array(click.repeat_info) / TIME_SCALAR
            length = len(history_time)
            neg_sample = np.random.choice(self.neg_samples[user_id], size=1)[0]
            return user_id, item_id, t, length, history_time, neg_sample
        if self.mode in ["dev", "test"]:
            data = self.corpus.data[self.mode][index]
            user_id = data["user_id"]
            user_ids = np.array([user_id] * self.corpus.n_items)
            proper_items = self.user_proper_items[index]
            t = np.array([self.ts[index]] * self.corpus.n_items)
            history_time = [np.array([])] * self.corpus.n_items
            length = np.zeros(self.corpus.n_items)
            for item_id in self.history_times[index]:
                history_time[item_id] = self.history_times[index][item_id]
                length[item_id] = history_time[item_id].shape[0]
            return user_ids, t, length, history_time, proper_items


def train_collate_fn(batch):
    user_ids, item_ids, ts, lengths, history_times, neg_samples = zip(*batch)

    max_len = max(map(len, history_times))
    padded_history_times = np.zeros([len(history_times), max_len], dtype=np.float64)
    for i in range(len(history_times)):
        padded_history_times[i, :len(history_times[i])] = history_times[i]

    return {
        "user_id": torch.tensor(user_ids),
        "item_id": torch.tensor(item_ids),
        "t": torch.tensor(ts),
        "length": torch.tensor(lengths),
        "history_time": torch.tensor(np.array(padded_history_times)),
        "neg_sample": torch.tensor(neg_samples)
    }


def eval_collate_fn(batch):
    user_ids, ts, lengths, history_times, proper_items = zip(*batch)

    user_ids = np.concatenate(user_ids, axis=0)
    ts = np.concatenate(ts, axis=0)
    lengths = np.concatenate(lengths, axis=0)
    concated_history_times = []
    for history_time in history_times:
        concated_history_times.extend(history_time)
    history_times = concated_history_times
    max_len = max(map(len, history_times))
    padded_history_times = np.zeros([len(history_times), max_len], dtype=np.float64)
    for i in range(len(history_times)):
        padded_history_times[i, :len(history_times[i])] = history_times[i]

    return {
        "user_id": torch.tensor(user_ids),
        "t": torch.tensor(ts),
        "length": torch.tensor(lengths),
        "history_time": torch.tensor(np.array(padded_history_times)),
        "proper_items": list(proper_items)
    }
