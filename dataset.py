'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
'''

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class GDataset(object):

    def __init__(self, user_path, group_path, num_negatives):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives
        # user data
        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")
        self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.txt")
        self.user_testNegatives = self.load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_trainMatrix.shape
        # group data
        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")
        self.group_testRatings = self.load_rating_file_as_list(group_path + "Test.txt")
        self.group_testNegatives = self.load_negative_file(group_path + "Negative.txt")


    def load_rating_file_as_list(self, filename):
        """
        Reads a list of rating files.

        Args:
            self: (todo): write your description
            filename: (str): write your description
        """
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        """
        Loads a negative negative negative negative negative numbers.

        Args:
            self: (todo): write your description
            filename: (str): write your description
        """
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        """
        Loads ascii as a matrix.

        Args:
            self: (todo): write your description
            filename: (str): write your description
        """
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_train_instances(self, train):
        """
        Get training instances of the dataset.

        Args:
            self: (todo): write your description
            train: (bool): write your description
        """
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_user_dataloader(self, batch_size):
        """
        Returns a tensor for a tensor.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
        """
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        """
        Gets the tensors.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
        """
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader






