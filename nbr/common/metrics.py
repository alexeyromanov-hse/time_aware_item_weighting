import math


def get_precision(rank_lst, purchase_lst):
    intersection = set(rank_lst).intersection(set(purchase_lst))
    precision = len(intersection) / len(rank_lst)
    return precision

def get_recall(rank_lst, purchase_lst):
    intersection = set(rank_lst).intersection(set(purchase_lst))
    recall = len(intersection) / len(purchase_lst)
    return recall

def get_ndcg(rank_lst, purchase_lst):
    dcg = 0
    for i in range(len(rank_lst)):
        item = rank_lst[i]
        if item in purchase_lst:
            dcg += math.log(2) / math.log(i + 2)
    idcg = 0
    for i in range(len(purchase_lst)):
        idcg += math.log(2) / math.log(i + 2)
    return float(dcg / idcg)
