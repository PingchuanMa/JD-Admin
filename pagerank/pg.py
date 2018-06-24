# -*- coding: UTF-8 -*-

import os
from urllib.parse import urljoin
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import json
import math


class LinkAnalyzer(object):

    def __init__(self):
        self.row_cnt = 0
        self.col_cnt = 0
        self.row = []
        self.col = []
        self.data = []
        self.href_cnt = 0
        self.href = dict()
        self.pr_matrix = None
        self.pr_vector = None
        self.time = 0

    def save_state(self):
        root_dir = "./json/"
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        with open(os.path.join(root_dir, "href.json"), 'w') as f:
            json.dump(self.href, f)
        with open(os.path.join(root_dir, "pr_vector.json"), 'w') as f:
            json.dump(self.pr_vector.squeeze().tolist(), f)

    def clear_temp(self):
        self.col = None
        self.row = None
        self.data = None

    def load_state(self):
        root_dir = "./json/"
        assert os.path.exists(root_dir)
        with open(os.path.join(root_dir, "href.json"), 'r') as f:
            self.href = dict(json.load(f))
        with open(os.path.join(root_dir, "pr_vector.json"), 'r') as f:
            self.pr_vector = np.array(json.load(f))

    def add_href(self, full_href):
        if full_href not in self.href:
            self.href[full_href] = self.href_cnt
            self.href_cnt += 1

    def set_row(self, full_href):
        self.row_cnt = self.href[full_href]

    def add_col(self, full_href):
        if self.row_cnt != self.href[full_href]:
            self.row.append(self.row_cnt)
            self.col.append(self.href[full_href])
            self.col_cnt += 1

    def add_row(self):
        if self.col_cnt > 0:
            self.data.extend([1 / self.col_cnt] * self.col_cnt)
            self.col_cnt = 0

    def make_pg_matrix(self):
        self.pr_matrix = csr_matrix((self.data, (self.row, self.col)), dtype=np.float32, shape=(self.href_cnt, self.href_cnt))
        # added by Yujun
        #------------------------
        deadend_lines = []
        for node in self.deadend_nodes:
            deadend_lines.append(self.href[node])
        self.pr_matrix[deadend_lines, :] = 1/self.href_cnt;
        #------------------------
        self.pr_matrix = self.pr_matrix.transpose()

    def compute_pr(self, p=0.85):
        process = 0
        pr = np.ones((self.href_cnt, 1), dtype=np.float32) * (1 / self.href_cnt)
        pr_ = p * self.pr_matrix.dot(pr) + (1 - p) * (1 / self.href_cnt)
        while not np.isclose(pr, pr_).all():
            pr = pr_
            pr_ = p * self.pr_matrix.dot(pr) + (1 - p) * (1 / self.href_cnt)
            process += 1
        self.pr_vector = pr_
        print("item num: %d" % len(self.data))
        print("dimension: %d" % self.href_cnt)
        print("iteration num: %d" % process)

    # block stripe strategy.
    # this code can just simulate the algorithm.
    # since the data can be fit into the memory
    def compute_pr_block_stripe(self, p=0.85, stripe=1000):
        num_block = int(math.ceil(self.href_cnt/stripe))
        stripe_pr_matrix = [];
        for i in range(num_block):
            low_bound = i*stripe
            up_bound = min((i+1)*stripe, self.href_cnt)
            stripe_pr_matrix.append(self.pr_matrix[low_bound:up_bound,:])

        process = 0
        pr = np.ones((self.href_cnt, 1), dtype=np.float32) * (1 / self.href_cnt)
        pr_ = np.zeros(pr.shape, dtype=np.float32)
        converge = False
        while not converge:
            tmp_converge = True
            for i in range(num_block):
                low_bound = i*stripe
                up_bound = min((i+1)*stripe, self.href_cnt)
                pr_[low_bound:up_bound] = p*stripe_pr_matrix[i].dot(pr)+(1-p)*(1/self.href_cnt)
                tmp_converge = tmp_converge and np.isclose(pr_[low_bound:up_bound], pr[low_bound:up_bound]).all()
            process += 1
            pr = np.copy(pr_)
            converge = tmp_converge
        self.pr_vector = pr
        print("item num: %d" % len(self.data))
        print("dimension: %d" % self.href_cnt)
        print("iteration num: %d" % process)


    def get_pr(self, full_href):
        return self.pr_vector[self.href[full_href]][0]

    def add_in(self, full_href):
        self.add_href(full_href)
        self.add_col(full_href)

    def add_out(self, full_href):
        self.add_href(full_href)
        self.set_row(full_href)

    def load_wiki(self, path):
        with open(path, 'r') as f:
            dat = pd.read_csv(f, sep='\t', header=None, names=['a', 'b']).sort_values(by='a')

        # added by Yujun
        #--------------------------
        all_nodes = np.unique(np.hstack([dat.a.values, dat.b.values]))
        self.deadend_nodes = set(all_nodes) - set(np.unique(dat.a.values))
        #--------------------------

        pre = None
        for _, line in dat.iterrows():
            # `a` is out-node, `b` is in-node
            if pre != line.a:
                self.add_row()
                # explicitly convert to `int` for serializing `np.int32`
                self.add_out(int(line.a))
                pre = line.a
            self.add_in(int(line.b))
        self.add_row()


    def get_delta_time(self):
        delta = time.time() - self.time
        self.time = time.time()
        return delta

    def output_result(self):
        inv_href = {v: k for k, v in self.href.items()}
        pr_vector_with_id = np.hstack([self.pr_vector,
            np.expand_dims(np.arange(self.href_cnt), axis=1)])
        sorted_idx = np.argsort(-pr_vector_with_id, axis=0)[:, 0]
        result_vector_with_id = pr_vector_with_id[sorted_idx]
        with open('result.txt', 'w') as f:
            for i in range(100):
                f.write(str(inv_href[int(result_vector_with_id[i, 1])]) + '\t' + str(result_vector_with_id[i, 0]) + '\n')


if __name__ == '__main__':
    la = LinkAnalyzer()
    la.load_wiki('./info/WikiData.txt')
    la.get_delta_time()
    la.make_pg_matrix()
    #la.compute_pr()
    la.compute_pr_block_stripe()
    print("Page Rank Done:", la.get_delta_time())
    la.save_state()
    la.output_result()
