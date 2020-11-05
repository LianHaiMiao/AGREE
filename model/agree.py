'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''

import torch
import torch.nn as nn

class AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        """
        Initialize the network states.

        Args:
            self: (todo): write your description
            num_users: (int): write your description
            num_items: (int): write your description
            num_groups: (int): write your description
            embedding_dim: (int): write your description
            group_member_dict: (dict): write your description
            drop_ratio: (float): write your description
        """
        super(AGREE, self).__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, group_inputs, user_inputs, item_inputs):
        """
        Parameters ---------- group_inputs : tuple )

        Args:
            self: (todo): write your description
            group_inputs: (todo): write your description
            user_inputs: (todo): write your description
            item_inputs: (todo): write your description
        """
        # train group
        if (group_inputs is not None) and (user_inputs is None):
            out = self.grp_forward(group_inputs, item_inputs)
        # train user
        else:
            out = self.usr_forward(user_inputs, item_inputs)
        return out

    # group forward
    def grp_forward(self, group_inputs, item_inputs):
        """
        Perform a forward computation.

        Args:
            self: (todo): write your description
            group_inputs: (todo): write your description
            item_inputs: (todo): write your description
        """
        group_embeds = torch.Tensor()
        item_embeds_full = self.itemembeds(torch.LongTensor(item_inputs))
        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[i.item()]
            members_embeds = self.userembeds(torch.LongTensor(members))
            items_numb = []
            for _ in members:
                items_numb.append(j)
            item_embeds = self.itemembeds(torch.LongTensor(items_numb))
            group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
            at_wt = self.attention(group_item_embeds)
            g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
            group_embeds_pure = self.groupembeds(torch.LongTensor([i]))
            g_embeds = g_embeds_with_attention + group_embeds_pure
            group_embeds = torch.cat((group_embeds, g_embeds))

        element_embeds = torch.mul(group_embeds, item_embeds_full)  # Element-wise product
        new_embeds = torch.cat((element_embeds, group_embeds, item_embeds_full), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

    # user forward
    def usr_forward(self, user_inputs, item_inputs):
        """
        Forward computation on a single layer.

        Args:
            self: (todo): write your description
            user_inputs: (todo): write your description
            item_inputs: (todo): write your description
        """
        user_embeds = self.userembeds(user_inputs)
        item_embeds = self.itemembeds(item_inputs)
        element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
        new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        """
        Initialize the embedding.

        Args:
            self: (todo): write your description
            num_users: (int): write your description
            embedding_dim: (int): write your description
        """
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):
        """
        Parameters ---------- user_embeds

        Args:
            self: (todo): write your description
            user_inputs: (todo): write your description
        """
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        """
        Initialize the embedding.

        Args:
            self: (todo): write your description
            num_items: (int): write your description
            embedding_dim: (int): write your description
        """
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        """
        Forward the given item.

        Args:
            self: (todo): write your description
            item_inputs: (todo): write your description
        """
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        """
        Initialize the embedding.

        Args:
            self: (todo): write your description
            number_group: (int): write your description
            embedding_dim: (int): write your description
        """
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):
        """
        Groups num_groups of num_group

        Args:
            self: (todo): write your description
            num_group: (int): write your description
        """
        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        """
        Initialize the embedding.

        Args:
            self: (todo): write your description
            embedding_dim: (int): write your description
            drop_ratio: (float): write your description
        """
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        out = self.linear(x)
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            embedding_dim: (int): write your description
            drop_ratio: (float): write your description
        """
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        out = self.linear(x)
        return out

