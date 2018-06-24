# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import json
import math
import scipy.sparse
import pickle


class LinkAnalyzer(object):

    def __init__(self, rtol=1e-5, atol=1e-8):
        """
        Initialize all variables
        """
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
        self.deadend_nodes = None

        self.rtol = rtol
        self.atol = atol

    def save_state(self):
        """
        Save the final state for future use
        :return: None
        """
        root_dir = "./json/"
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        with open(os.path.join(root_dir, "href.json"), 'w') as f:
            json.dump(self.href, f)
        with open(os.path.join(root_dir, "pr_vector.json"), 'w') as f:
            json.dump(self.pr_vector.squeeze().tolist(), f)

    def clear_temp(self):
        """
        Clear large temp variable to save memory
        :return: None
        """
        self.col = None
        self.row = None
        self.data = None

    def load_state(self):
        """
        Load previous computed vector and hash
        :return: None
        """
        root_dir = "./json/"
        assert os.path.exists(root_dir)
        with open(os.path.join(root_dir, "href.json"), 'r') as f:
            self.href = dict(json.load(f))
        with open(os.path.join(root_dir, "pr_vector.json"), 'r') as f:
            self.pr_vector = np.array(json.load(f))

    def add_href(self, full_href):
        """
        Add a new href to hash if it doesn't exist
        :param full_href: the href to be added
        :return: None
        """
        if full_href not in self.href:
            self.href[full_href] = self.href_cnt
            # Increment `href_cnt` for future hash
            self.href_cnt += 1

    def set_row(self, full_href):
        """
        Seek to the exact row, then process all its out-degree nodes
        Somehow resemble to seek_frame in FFmpeg
        :param full_href: the href
        :return: None
        """
        self.row_cnt = self.href[full_href]

    def add_col(self, full_href):
        """
        Add a out-degree node for an exact row if this node is not itself
        :param full_href: the href
        :return: None
        """
        if self.row_cnt != self.href[full_href]:
            self.row.append(self.row_cnt)
            self.col.append(self.href[full_href])
            # increment the column counter by $1$
            self.col_cnt += 1

    def add_row(self):
        """
        Here we assume a row is finished and there will not be any out-degree nodes
        of this node anymore. So, we add a column-number-normalized row to the matrix
        which represents that the higher the out-degree is, the lower the importance is
        :return: None
        """
        if self.col_cnt > 0:
            self.data.extend([1 / self.col_cnt] * self.col_cnt)
            # move to the next row
            self.col_cnt = 0

    def make_pg_matrix(self):
        """
        Construct a sparse matrix according to the structure introduced in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
        :return: None
        """
        self.pr_matrix = csr_matrix((self.data, (self.row, self.col)),
                                    dtype=np.float32, shape=(self.href_cnt, self.href_cnt))
        # deal with dead end here. randomly walk to any node if it's a dead end node
        deadend_lines = []
        for node in self.deadend_nodes:
            deadend_lines.append(self.href[node])
        self.pr_matrix[deadend_lines, :] = 1 / self.href_cnt;

        # we have to transpose to get the actual page rank matrix
        self.pr_matrix = self.pr_matrix.transpose()

    def compute_pr(self, p=0.85):
        # statistical value
        process = 0

        # the current page rank vector
        pr = np.ones((self.href_cnt, 1), dtype=np.float32) * (1 / self.href_cnt)

        # the page rank vector computed after iteration
        pr_ = p * self.pr_matrix.dot(pr) + (1 - p) * (1 / self.href_cnt)

        # Let's iterate till the end of the world! (just kidding)
        # Here we use `np.isclose()` method to judge whether the
        # computed vector is enough close to the real value, using
        # relative tolerance `rtol` and absolute tolerance `atol`
        while not np.isclose(pr, pr_, rtol=self.rtol, atol=self.atol).all():
            pr = pr_
            # next iteration
            pr_ = p * self.pr_matrix.dot(pr) + (1 - p) * (1 / self.href_cnt)
            process += 1
        # collect the final result
        self.pr_vector = pr_
        # visualize the process
        self.print_info()
        print("iteration num: %d" % process)

    # block stripe strategy.
    # this code can just simulate the algorithm.
    # since the data can be fit into the memory
    def compute_pr_block_stripe(self, p=0.85, stripe=1000):
        num_block = int(math.ceil(self.href_cnt / stripe))
        # slice the original sparse matrix into `num_block` stripes
        stripe_pr_matrix = [];
        for i in range(num_block):
            # lower bound and upper bound index for the submatrix of the
            # original page rank matrix
            low_bound = i * stripe
            up_bound = min((i + 1) * stripe, self.href_cnt)
            # put the submatrix into the list
            stripe_pr_matrix.append(self.pr_matrix[low_bound:up_bound, :])

        process = 0
        pr = np.ones((self.href_cnt, 1), dtype=np.float32) * (1 / self.href_cnt)
        pr_ = np.zeros(pr.shape, dtype=np.float32)
        converge = False
        while not converge:
            tmp_converge = True
            for i in range(num_block):
                low_bound = i * stripe
                up_bound = min((i + 1) * stripe, self.href_cnt)
                # update a subset of the nodes by the submatrix
                pr_[low_bound:up_bound] = p * stripe_pr_matrix[i].dot(pr) + (1 - p) * (1 / self.href_cnt)
                # only when all subset of the nodes converge can we say that
                # the whole system converged.
                tmp_converge = tmp_converge and np.isclose(pr_[low_bound:up_bound], pr[low_bound:up_bound], rtol=self.rtol, atol=self.atol).all()
            process += 1
            pr = np.copy(pr_)
            converge = tmp_converge
        # store the result
        self.pr_vector = pr
        self.print_info()
        print("iteration num: %d" % process)

    # block stripe strategy.
    # this code can just simulate the algorithm.
    # since the data can be fit into the memory

    # scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    # sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

    def compute_pr_block_stripe_disk(self, p=0.85, stripe=1000):
        num_block = int(math.ceil(self.href_cnt / stripe))
        # slice the original sparse matrix into `num_block` stripes
        for i in range(num_block):
            # lower bound and upper bound index for the submatrix of the
            # original page rank matrix
            low_bound = i * stripe
            up_bound = min((i + 1) * stripe, self.href_cnt)
            # put the submatrix into the list
            scipy.sparse.save_npz('diskcache/' + str(i) + '.npz', self.pr_matrix[low_bound:up_bound, :])

        process = 0
        pr = np.ones((self.href_cnt, 1), dtype=np.float32) * (1 / self.href_cnt)
        converge = False
        while not converge:
            tmp_converge = True
            for i in range(num_block):
                low_bound = i * stripe
                up_bound = min((i + 1) * stripe, self.href_cnt)
                # update a subset of the nodes by the submatrix
                sub_pr_matrix = scipy.sparse.load_npz('diskcache/' + str(i) + '.npz')

                tmp_res = p * sub_pr_matrix.dot(pr) + (1 - p) * (1 / self.href_cnt)
                with open('diskcache/tmp_rank_' + str(i) + '.p', 'wb') as f:
                    pickle.dump(tmp_res, f)
                # only when all subset of the nodes converge can we say that
                # the whole system converged.
                tmp_converge = tmp_converge and np.isclose(tmp_res, pr[low_bound:up_bound], rtol=self.rtol, atol=self.atol).all()
            process += 1

            for i in range(num_block):
                low_bound = i * stripe
                up_bound = min((i + 1) * stripe, self.href_cnt)
                with open('diskcache/tmp_rank_' + str(i) + '.p', 'rb') as f:
                    pr[low_bound:up_bound] = pickle.load(f)
            converge = tmp_converge
        # store the result
        self.pr_vector = pr
        self.print_info()
        print("iteration num: %d" % process)

    def get_pr(self, full_href):
        """
        Get the page rank value of a href according to hash
        :param full_href: the href
        :return: the page rank value
        """
        return self.pr_vector[self.href[full_href]][0]

    def add_in(self, full_href):
        """
        A warped method for adding in-node
        :param full_href: the in-node
        :return: None
        """
        self.add_href(full_href)
        self.add_col(full_href)

    def add_out(self, full_href):
        """
        A warped method for adding out-node
        :param full_href: the out-node
        :return: None
        """
        self.add_href(full_href)
        self.set_row(full_href)

    def load_wiki(self, path):
        with open(path, 'r') as f:
            dat = pd.read_csv(f, sep='\t', header=None, names=['a', 'b']).sort_values(by='a')

        # obtain all dead end nodes
        all_nodes = np.unique(np.hstack([dat.a.values, dat.b.values]))
        self.deadend_nodes = set(all_nodes) - set(np.unique(dat.a.values))

        pre = None
        for _, line in dat.iterrows():
            # `a` is out-node, `b` is in-node, if this is a new row
            if pre != line.a:
                # add this new row
                self.add_row()
                # explicitly convert to `int` for serializing `np.int32`
                self.add_out(int(line.a))
                # to identify the new row
                pre = line.a
            # add a new in-node
            self.add_in(int(line.b))
        # finalize the result by normalize the last row
        self.add_row()

    def get_delta_time(self):
        """
        Get the statistical running time
        :return: the delta time
        """
        delta = time.time() - self.time
        self.time = time.time()
        return delta

    def print_info(self):
        """
        Print key informations
        """
        print("link num: %d" % len(self.data))
        print("dimension: %d" % self.href_cnt)
        print("tolerance: %.2e(r), %.2e(a)" % (self.rtol, self.atol))

    def output_result(self):
        """
        Simply output our page rank vector according to the given format
        :return: None
        """
        inv_href = {v: k for k, v in self.href.items()}
        pr_vector_with_id = np.hstack([self.pr_vector, np.expand_dims(np.arange(self.href_cnt), axis=1)])
        sorted_idx = np.argsort(-pr_vector_with_id, axis=0)[:, 0]
        result_vector_with_id = pr_vector_with_id[sorted_idx]
        with open('result.txt', 'w') as f:
            for i in range(100):
                f.write(str(inv_href[int(result_vector_with_id[i, 1])]) + '\t' +
                        str(result_vector_with_id[i, 0]) + '\n')


if __name__ == '__main__':
    la = LinkAnalyzer()
    la.load_wiki('./info/WikiData.txt')
    la.get_delta_time()
    la.make_pg_matrix()
    # la.compute_pr()
    # la.compute_pr_block_stripe()
    la.compute_pr_block_stripe_disk()
    print("page-rank done in %.2f seconds" % la.get_delta_time())
    la.save_state()
    la.output_result()
