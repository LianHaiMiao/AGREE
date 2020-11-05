'''
Created on Nov 10, 2017
Deal something

@author: Lianhai Miao
'''
import torch
from torch.autograd import Variable
import numpy as np
import math
import heapq

class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        """
        Initialize the object.

        Args:
            self: (todo): write your description
        """
        self.timber = True

    def gen_group_member_dict(self, path):
        """
        Parse a member_m_member

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        return g_m_d

    def evaluate_model(self, model, testRatings, testNegatives, K, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []

        for idx in range(len(testRatings)):
            (hr,ndcg) = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx)
            hits.append(hr)
            ndcgs.append(ndcg)
        return (hits, ndcgs)


    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx):
        """
        Evaluate a single item

        Args:
            self: (todo): write your description
            model: (todo): write your description
            testRatings: (str): write your description
            testNegatives: (todo): write your description
            K: (todo): write your description
            type_m: (str): write your description
            idx: (todo): write your description
        """
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users)
        users_var = users_var.long()
        items_var = torch.LongTensor(items)
        if type_m == 'group':
            predictions = model(users_var, None, items_var)
        elif type_m == 'user':
            predictions = model(None, users_var, items_var)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.numpy()[i]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        """
        Return the first item in a list.

        Args:
            self: (todo): write your description
            ranklist: (list): write your description
            gtItem: (str): write your description
        """
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        """
        Returns the first rank of a list of ranklist.

        Args:
            self: (todo): write your description
            ranklist: (list): write your description
            gtItem: (str): write your description
        """
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0