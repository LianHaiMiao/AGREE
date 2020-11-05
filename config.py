'''
Created on Nov 10, 2017
Store parameters

@author: Lianhai Miao
'''

class Config(object):
    def __init__(self):
        """
        Initialize embedding and embeddings.

        Args:
            self: (todo): write your description
        """
        self.path = './data/CAMRa2011/'
        self.user_dataset = self.path + 'userRating'
        self.group_dataset = self.path + 'groupRating'
        self.user_in_group_path = "./data/CAMRa2011/groupMember.txt"
        self.embedding_size = 32
        self.epoch = 30
        self.num_negatives = 6
        self.batch_size = 256
        self.lr = [0.000005, 0.000001, 0.0000005]
        self.drop_ratio = 0.2
        self.topK = 5
