import math
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from config import *

def get_total_num(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        seq = f.readlines()[0].split()
        entityNum = seq[0]
        relationNum = seq[1]
        return int(entityNum), int(relationNum)

def load_quadruples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        quadList = []
        times = set()
        for line in f:
            if args.dataset == 'WIKI':
                line = line.replace(' ', '\t')
            seq = line.split('\t')
            head = int(seq[0])
            rel = int(seq[1])
            tail = int(seq[2])
            T = int(seq[3])
            quadList.append([head, rel, tail, T])
            times.add(T)
        times = list(times)
        times.sort()
        return np.asarray(quadList), np.asarray(times)

def get_quad_with_time(data, T):
    triples = data[np.where(data[:, 3] == T)][:, :3]
    return triples

def get_id_dict(file_path):
    id_name = {}
    name_id = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().replace('\n', '')
            seq = line.split('\t')
            if len(seq) > 1:
                name = seq[0]
                idx = int(seq[1])
            else:
                raw = seq[0]
                # 取出raw中的数字
                idx = int(''.join(filter(str.isdigit, raw)))
                # 替换raw中的数字为空
                name = raw.replace(str(idx), '')
                name = name.strip()
            id_name[idx] = name
            name_id[name] = idx
    return id_name, name_id

def get_id_dict_rel(file_path, file_path_v, num_rel):
    id_name = {}
    name_id = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().replace('\n', '')
            seq = line.split('\t')
            if len(seq) > 1:
                name = seq[0]
                idx = int(seq[1])
            else:
                raw = seq[0]
                # 取出raw中的数字
                idx = int(''.join(filter(str.isdigit, raw)))
                # 替换raw中的数字为空
                name = raw.replace(str(idx), '')
                name = name.strip()
            id_name[idx] = name
            name_id[name] = idx
    with open(file_path_v, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().replace('\n', '')
            seq = line.split('\t')
            if len(seq) > 1:
                name = seq[0]
                idx = int(seq[1])
            else:
                raw = seq[0]
                # 取出raw中的数字
                idx = int(''.join(filter(str.isdigit, raw)))
                # 替换raw中的数字为空
                name = raw.replace(str(idx), '')
                name = name.strip()
            id_name[idx+num_rel] = name
            name_id[name] = idx+num_rel
    return id_name, name_id

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def calc_mrr(score, labels, hits=[]):
    with torch.no_grad():
        ranks = sort_and_rank(score, labels)
        ranks += 1
        mrr = torch.mean(1.0 / ranks.float())
        hitsRet = [torch.mean((ranks <= i).float()).item() for i in hits]
        return mrr.item(), hitsRet

def calc_mrr_count(score, labels, hits=[]):
    with torch.no_grad():
        ranks = sort_and_rank(score, labels)
        ranks += 1
        mrrWoMean = 1.0 / ranks.float()
        mrrCount = torch.sum(mrrWoMean).item()
        hitsCount = [torch.sum((ranks <= i).float()).item() for i in hits]
        return mrrCount, hitsCount

def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t, no_use = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score
def calc_mrr_count_filter(score, labels, quads, hits=[], all_ans={}):
    with torch.no_grad():
        score = filter_score(quads, score, all_ans)
        ranks = sort_and_rank(score, labels)
        ranks += 1
        mrrWoMean = 1.0 / ranks.float()
        mrrCount = torch.sum(mrrWoMean).item()
        hitsCount = [torch.sum((ranks <= i).float()).item() for i in hits]
        return mrrCount, hitsCount