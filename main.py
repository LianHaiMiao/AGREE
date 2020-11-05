'''
Created on Nov 10, 2017
Main function

@author: Lianhai Miao
'''

from model.agree import AGREE
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from time import time
from config import Config
from utils.util import Helper
from dataset import GDataset

from tqdm import tqdm

# train the model
def training(model, train_loader, epoch_id, config, type_m):
    """
    Training function.

    Args:
        model: (todo): write your description
        train_loader: (todo): write your description
        epoch_id: (int): write your description
        config: (todo): write your description
        type_m: (todo): write your description
    """
    # user trainning
    learning_rates = config.lr
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 15 and epoch_id < 25:
        lr = learning_rates[1]
    elif epoch_id >=20:
        lr = learning_rates[2]
    # lr decay
    if epoch_id % 5 == 0:
        lr /= 2

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr)

    losses = []
    print('%s train_loader length: %d' % (type_m, len(train_loader)))
    for batch_id, (u, pi_ni) in tqdm(enumerate(train_loader)):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]
        # Forward
        if type_m == 'user':
            pos_prediction = model(None, user_input, pos_item_input)
            neg_prediction = model(None, user_input, neg_item_input)
        elif type_m == 'group':
            pos_prediction = model(user_input, None, pos_item_input)
            neg_prediction = model(user_input, None, neg_item_input)
        # Zero_grad
        model.zero_grad()
        # Loss
        loss = torch.mean((pos_prediction - neg_prediction -1) **2)
        # record loss history
        losses.append(loss)  
        # Backward
        loss.backward()
        optimizer.step()

    print('Iteration %d, loss is [%.4f ]' % (epoch_id, torch.mean(torch.stack(losses))))


def evaluation(model, helper, testRatings, testNegatives, K, type_m):
    """
    Evaluate the model.

    Args:
        model: (todo): write your description
        helper: (todo): write your description
        testRatings: (str): write your description
        testNegatives: (todo): write your description
        K: (todo): write your description
        type_m: (str): write your description
    """
    model.eval()
    (hits, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return hr, ndcg


if __name__ == '__main__':
    # initial parameter class
    config = Config()

    # initial helper
    helper = Helper()

    # get the dict of users in group
    g_m_d = helper.gen_group_member_dict(config.user_in_group_path)

    # initial dataSet class
    dataset = GDataset(config.user_dataset, config.group_dataset, config.num_negatives)

    # get group number
    num_group = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items

    # build AGREE model
    agree = AGREE(num_users, num_items, num_group, config.embedding_size, g_m_d, config.drop_ratio)

    # config information
    print("AGREE at embedding size %d, run Iteration:%d, NDCG and HR at %d" %(config.embedding_size, config.epoch, config.topK))
    # train the model
    for epoch in range(config.epoch):
        agree.train()
        # 开始训练时间
        t1 = time()
        training(agree, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')

        training(agree, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')
        print("user and group training time is: [%.1f s]" % (time()-t1))
        # evaluation
        t2 = time()
        u_hr, u_ndcg = evaluation(agree, helper, dataset.user_testRatings, dataset.user_testNegatives, config.topK, 'user')
        print('User Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' % (
            epoch, time() - t1, u_hr, u_ndcg, time() - t2))

        hr, ndcg = evaluation(agree, helper, dataset.group_testRatings, dataset.group_testNegatives, config.topK, 'group')
        print(
            'Group Iteration %d [%.1f s]: HR = %.4f, '
            'NDCG = %.4f, [%.1f s]' % (epoch, time() - t1, hr, ndcg, time() - t2))


    print("Done!")










