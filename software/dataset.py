# Dataset
import numpy as np

from config import args
from utils import *


class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset
        statTxt = './data/{}/stat.txt'.format(dataset)
        self.numEnt, self.numRel = get_total_num(statTxt)

    def get_num(self):
        return self.numEnt, self.numRel

    def get_stamp(self):
        trainQuadList, trainTimes = load_quadruples('./data/{}/train.txt'.format(self.dataset))
        return trainTimes[1] - trainTimes[0]

    def get_stamp_and_max_time(self):
        trainQuadList, trainTimes = load_quadruples('./data/{}/train.txt'.format(self.dataset))
        testQuadList, testTimes = load_quadruples('./data/{}/test.txt'.format(self.dataset))
        if self.dataset == 'ICEWS05':
            print('special timestamp for ICEWS05')
            return 1, -1
        return trainTimes[1] - trainTimes[0], testTimes[-1]

    def get_entity_by_id(self):
        return get_id_dict(self.datadir + '/{}/entity2id.txt'.format(self.dataset))

    def get_rel_by_id(self):
        return get_id_dict_rel(self.datadir + '/{}/rel2idno.txt'.format(self.dataset),
                               self.datadir + '/{}/rel2idv.txt'.format(self.dataset), self.numRel)

    def data_for_dynamic_train(self):
        trainQuadList, trainTimes = load_quadruples('./data/{}/train.txt'.format(self.dataset))
        return trainQuadList

    def data_for_dynamic_valid(self):
        validQuadList, validTimes = load_quadruples('./data/{}/valid.txt'.format(self.dataset))
        return validQuadList

    def data_for_dynamic_test(self):
        testQuadList, testTimes = load_quadruples('./data/{}/test.txt'.format(self.dataset))
        return testQuadList

    # inspired by TiRGN
    def split_by_time(self, data):
        snapshot_list = []
        snapshot = []
        snapshots_num = 0
        latest_t = 0
        for i in range(len(data)):
            t = data[i][3]
            train = data[i]
            if latest_t != t:
                # show snapshot
                latest_t = t
                if len(snapshot):
                    # snapshot_list.append(np.array(snapshot).copy())
                    snapshot_list.append(np.array([item.cpu().detach().numpy() for item in snapshot]))
                    snapshots_num += 1
                snapshot = []
            snapshot.append(train[:])
        if len(snapshot) > 0:
            # snapshot_list.append(np.array(snapshot).copy())
            snapshot_list.append(np.array([item.cpu().detach().numpy() for item in snapshot]))
            snapshots_num += 1

        union_num = [1]
        nodes = []
        rels = []
        for snapshot in snapshot_list:
            uniq_v, edges = np.unique((snapshot[:, 0], snapshot[:, 2]), return_inverse=True)  # relabel
            uniq_r = np.unique(snapshot[:, 1])
            edges = np.reshape(edges, (2, -1))
            nodes.append(len(uniq_v))
            rels.append(len(uniq_r) * 2)
        times = set()
        for triple in data:
            times.add(triple[3])
        times = list(times)
        times.sort()
        return snapshot_list, np.asarray(times)

    def load_all_answers_for_time_filter(self, total_data, num_rels, num_nodes, rel_p=False):
        all_ans_list = []
        all_snap, nouse = self.split_by_time(total_data)
        for snap in all_snap:
            all_ans_t = self.load_all_answers_for_filter(snap, num_rels, rel_p)
            all_ans_list.append(all_ans_t)
        return all_ans_list

    def load_all_answers_for_filter(self, total_data, num_rel, rel_p=False):
        # store subjects for all (rel, object) queries and
        # objects for all (subject, rel) queries
        def add_relation(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not e2 in d[e1]:
                d[e1][e2] = set()
            d[e1][e2].add(r)

        def add_subject(e1, e2, r, d, num_rel):
            if not e2 in d:
                d[e2] = {}
            if not r + num_rel in d[e2]:
                d[e2][r + num_rel] = set()
            d[e2][r + num_rel].add(e1)

        def add_object(e1, e2, r, d, num_rel):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        all_ans = {}
        for line in total_data:
            s, r, o = line[: 3]
            if rel_p:
                add_relation(s, o, r, all_ans)
                add_relation(o, s, r + num_rel, all_ans)
            else:
                add_subject(s, o, r, all_ans, num_rel=num_rel)
                add_object(s, o, r, all_ans, num_rel=0)
        return all_ans