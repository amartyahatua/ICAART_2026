import torch
import random
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ordered_fr_2 = None
        self.rank_fr_2 = None
        self.topK_fr_2 = None
        self.random_fr_2 = None
        self.random_fr_1 = None
        self.topK_fr_1 = None
        self.ordered_fr_1 = None
        self.rank_fr_1 = None
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.50)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def get_index(self, df_avg):
        """
        It sorts the value of the dataframe and returns the sorted index

        param: df_avg Dataframe having activation value of each neuron
        return: list of sorted index
        """
        L = []
        for val in df_avg.values.tolist():
            L.extend(val)
        x = tuple(k[1] for k in sorted((x[1], j) for j, x in enumerate(
            sorted((x, i) for i, x in enumerate(L)))))
        ord_index = [max(x) - i for i in list(x)]
        return ord_index

    def node_order(self, weights):
        """
        Get sorted index based on the values of the weights

        param: weights Dataframe having activation value of each neuron
        return: list of sorted index
        """
        average = torch.mean(weights, axis=0)
        new_average = pd.DataFrame(average.cpu().detach().numpy())
        ord_index = self.get_index(new_average)
        return ord_index

    def forward(self, x):
        if self.type == 'learning':
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)

            ########################################################################################
            ### Calculate the ranks of the nodes using 4 different techniques for first FC layer ###
            ########################################################################################

            # Rank forget rate
            self.rank_fr_1 = self.node_order(x)
            self.rank_fr_1 = torch.tensor(self.rank_fr_1)
            self.rank_fr_1 = self.rank_fr_1.to(self.device)

            # Ordered forget rate
            self.ordered_fr_1 = torch.tensor([i for i in range(x.shape[1])])
            self.ordered_fr_1 = self.ordered_fr_1.to(self.device)

            # Top K forget rate
            rank = self.node_order(x)
            self.topK_fr_1 = torch.tensor(rank)
            self.topK_fr_1 = self.topK_fr_1.to(self.device)

            # Random forget rate
            random_numbers = random.sample(range(0, x.shape[1] - 1), random.randint(0, x.shape[1] - 1))
            rank = self.node_order(x)
            for rn in random_numbers:
                try:
                    rank[rn] = 1
                except:
                    continue
            self.random_fr_1 = torch.tensor(rank)
            self.random_fr_1 = self.random_fr_1.to(self.device)

            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)

            ########################################################################################
            ### Calculate the ranks of the nodes using 4 different techniques for second FC layer ###
            ########################################################################################

            # Rank forget rate
            self.rank_fr_2 = self.node_order(x)
            self.rank_fr_2 = torch.tensor(self.rank_fr_2)
            self.rank_fr_2 = self.rank_fr_2.to(self.device)

            # Ordered forget rate
            self.ordered_fr_2 = torch.tensor([i for i in range(x.shape[1])])
            self.ordered_fr_2 = self.ordered_fr_2.to(self.device)

            # Top K forget rate
            rank = self.node_order(x)
            self.topK_fr_2 = torch.tensor(rank)
            self.topK_fr_2 = self.topK_fr_2.to(self.device)

            # Random forget rate
            random_numbers = random.sample(range(0, x.shape[1] - 1), random.randint(0, x.shape[1] - 1))
            rank = self.node_order(x)
            for rn in random_numbers:
                try:
                    rank[rn] = 1
                except:
                    continue
            self.random_fr_2 = torch.tensor(rank)
            self.random_fr_2 = self.random_fr_2.to(self.device)

            output = F.log_softmax(x, dim=1)
            return output
        else:
            if self.unlearning_type == 'rank_fr':
                rank_1 = self.rank_fr_1
                rank_2 = self.rank_fr_2
            elif self.unlearning_type == 'ordered_fr':
                rank_1 = self.ordered_fr_1
                rank_2 = self.ordered_fr_2
            elif self.unlearning_type == 'topK_fr':
                rank_1 = self.topK_fr_1
                rank_2 = self.topK_fr_2
            elif self.unlearning_type == 'random_fr':
                rank_1 = self.random_fr_1
                rank_2 = self.random_fr_2

            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = x * torch.exp(-(self.epoch / rank_1))
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = x * torch.exp(-(self.epoch / rank_2))
            output = F.log_softmax(x, dim=1)
            return output
