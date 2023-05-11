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
        return get_id_dict('./data/{}/entity2id.txt'.format(self.dataset))

    def get_rel_by_id(self):
        return get_id_dict('./data/{}/rel2id.txt'.format(self.dataset))

    def data_for_dynamic_train(self):
        trainQuadList, trainTimes = load_quadruples('./data/{}/train.txt'.format(self.dataset))
        return trainQuadList

    def data_for_dynamic_valid(self):
        validQuadList, validTimes = load_quadruples('./data/{}/valid.txt'.format(self.dataset))
        return validQuadList

    def data_for_dynamic_test(self):
        testQuadList, testTimes = load_quadruples('./data/{}/test.txt'.format(self.dataset))
        return testQuadList
