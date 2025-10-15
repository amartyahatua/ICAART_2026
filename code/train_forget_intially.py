import os
import copy
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from aaai.mnist.mia_score import calculate_mia, get_mia_ibm
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import random
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.autograd.set_detect_anomaly(True)

rank_fr_1 = torch.tensor([19, 62, 35, 116, 81, 118, 45, 48, 121, 88, 64, 125, 75, 36,
             102, 87, 41, 53, 59, 7, 74, 120, 57, 5, 4, 55, 108, 34,
             82, 10, 51, 70, 99, 1, 8, 98, 89, 33, 58, 29, 101, 47,
             26, 90, 123, 42, 79, 39, 127, 106, 6, 93, 25, 31, 80, 14,
             17, 71, 20, 32, 69, 37, 77, 86, 50, 30, 96, 65, 114, 78,
             122, 73, 117, 22, 38, 46, 40, 3, 15, 68, 115, 43, 97, 63,
             44, 92, 18, 100, 94, 9, 109, 85, 67, 124, 105, 49, 16, 0,
             60, 61, 12, 76, 103, 112, 104, 111, 52, 110, 66, 83, 11, 28,
             107, 126, 119, 27, 95, 91, 2, 56, 72, 23, 84, 13, 54, 24,
             21, 113])
rank_fr_2 = torch.tensor([9, 8, 1, 4, 3, 2, 7, 6, 0, 5])

ordered_fr_1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                126, 127])
ordered_fr_2 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

topK_fr_1 = torch.tensor([19, 62, 35, 116, 81, 118, 45, 48, 121, 88, 64, 125, 75, 36,
             102, 87, 41, 53, 59, 7, 74, 120, 57, 5, 4, 55, 108, 34,
             82, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1])
topK_fr_2 = torch.tensor([9, 8, 1, 4, 3, 2, 7, 6, 0, 5])

random_fr_1 = torch.tensor([37, 83, 2, 113, 122, 9, 32, 7, 70, 103, 52, 81, 41, 50,
               123, 85, 79, 80, 105, 24, 73, 111, 67, 10, 57, 38, 45, 5,
               91, 99, 40, 96, 59, 42, 30, 107, 39, 60, 86, 15, 6, 114,
               17, 34, 98, 117, 95, 88, 72, 68, 53, 4, 62, 0, 71, 36,
               61, 27, 19, 106, 1, 75, 94, 13, 44, 47, 21, 66, 26, 76,
               127, 120, 92, 116, 97, 77, 115, 64, 93, 11, 56, 112, 55, 125,
               49, 28, 90, 110, 65, 43, 3, 58, 63, 126, 101, 108, 89, 54,
               51, 109, 121, 22, 69, 124, 74, 48, 14, 118, 87, 31, 33, 46,
               20, 35, 8, 16, 84, 23, 25, 100, 78, 119, 29, 102, 104, 82,
               18, 12])
random_fr_2 = torch.tensor([3, 1, 2, 0, 6, 9, 5, 4, 8, 7])


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
            K=29
            rank = self.node_order(x)
            for i in range(len(rank)-K):
                rank[i+K] = 1
            self.topK_fr_1 = torch.tensor(rank)
            self.topK_fr_1 = self.topK_fr_1.to(self.device)

            # Random forget rate
            rank = self.node_order(x)
            np.random.shuffle(rank)
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
            K=29
            rank = self.node_order(x)
            for i in range(len(rank)-K):
                rank[i+K] = 1
            self.topK_fr_2 = torch.tensor(rank)
            self.topK_fr_2 = self.topK_fr_2.to(self.device)

            # Random forget rate
            rank = self.node_order(x)
            np.random.shuffle(rank)
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



def train(args, model, device, train_loader, optimizer, epoch, type, unlearning_type=''):
    model.train()
    model.type = type
    model.epoch = epoch
    model.device = device
    model.unlearning_type = unlearning_type
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f})]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


accuracy = []
accuracy_num = []
mia_score_list = []


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    print(f'Learning type: {type}; droupout 1: {model.dropout1};  droupout 2: {model.dropout2}')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f})\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracy.append(correct / len(test_loader.dataset))
    accuracy_num.append(correct)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.07, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    # elif use_mps:
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [50000, 10000],
                                                                generator=torch.Generator().manual_seed(42))
    dataset_retain, dataset_forget = torch.utils.data.random_split(dataset_train, [40000, 10000],
                                                                   generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    forget_loader = torch.utils.data.DataLoader(dataset_forget, **train_kwargs)
    retain_loader = torch.utils.data.DataLoader(dataset_retain, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)


    model_forget = Net().to(device)
    optimizer = torch.optim.Adam(model_forget.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # print(f'------------------------Check weights for forget dataset-----------------------------')
    # for epoch in range(1, args.epochs + 2):
    #     train(args, model_forget, device, forget_loader, optimizer, epoch, 'learning', '')
    #     test(model_forget, device, test_loader)
    #     scheduler.step()
    #     #get_mia_ibm(model)
    #     mia_temp = calculate_mia(model_forget, test_loader, forget_loader)
    #     print('Forget MIA =', mia_temp)
    #     mia_score_list.append(mia_temp)
    #
    # print('model_forget.rank_fr_1', model_forget.rank_fr_1)
    # print('model_forget.rank_fr_2', model_forget.rank_fr_2)
    #
    # print('model_forget.ordered_fr_1', model_forget.ordered_fr_1)
    # print('model_forget.ordered_fr_2', model_forget.ordered_fr_2)
    #
    # print('model_forget.topK_fr_1', model_forget.topK_fr_1)
    # print('model_forget.topK_fr_2', model_forget.topK_fr_2)
    #
    # print('model_forget.random_fr_1', model_forget.random_fr_1)
    # print('model_forget.random_fr_2', model_forget.random_fr_2)

    print(f'------------------------Learning-----------------------------')

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 5):
        train(args, model, device, train_loader, optimizer, epoch, 'learning', '')
        test(model, device, test_loader)
        scheduler.step()
        #get_mia_ibm(model)
        mia_temp = calculate_mia(model, test_loader, forget_loader)
        print('Learn MIA =', mia_temp)
        mia_score_list.append(mia_temp)



    print(f'------------------------Unlearning-----------------------------')
    for epoch in range(1, args.epochs + 15):
        # Rank forget rate=rank_fr [0.9581, 0.9761, 0.9766, 0.8962, 0.8526, 0.8254, 0.8094, 0.7991, 0.793]
        # Ordered forget rate=ordered_fr [0.9581, 0.9761, 0.9766, 0.9347, 0.9239, 0.9038, 0.875, 0.8478, 0.8236]
        # Top K forget rate=topK_fr [0.9581, 0.9761, 0.9766, 0.8962, 0.8526, 0.8254, 0.8094, 0.7991, 0.793]
        # Random forget rate=random_fr [0.9581, 0.9761, 0.9766, 0.9433, 0.7824, 0.617, 0.3688, 0.323, 0.3099]

        model.rank_fr_1 = rank_fr_1
        model.rank_fr_2 = rank_fr_2

        model.ordered_fr_1 = ordered_fr_1
        model.ordered_fr_2 = ordered_fr_2

        model.topK_fr_1 = topK_fr_1
        model.topK_fr_2 = topK_fr_2

        model.random_fr_1 = random_fr_1
        model.random_fr_2 = random_fr_2

        train(args, model, device, forget_loader, optimizer, epoch, 'unlearning', 'rank_fr')
        #test(model, device, test_loader)
        test(model, device, retain_loader)

        scheduler.step()
        #get_mia_ibm(model)
        mia_temp = calculate_mia(model, test_loader, forget_loader)
        print('MIA =', mia_temp)
        mia_score_list.append(mia_temp)
    # if args.save_model:
    # torch.save(model.state_dict(), "../model/top_30_L1_epoch_variable.pt")


if __name__ == '__main__':
    main()
    print(accuracy)
    print(mia_score_list)

    accuracy = pd.DataFrame(accuracy)
    mia_score_df = pd.DataFrame(mia_score_list)
    # accuracy.to_csv('plots/with_mia/5_epoch_1_Layers/Result_Rank_top_30_all_epoch_all_layer.csv', index=False)
    # mia_score_df.to_csv('plots/with_mia/5_epoch_1_Layers/MIA_Rank_top_30_all_epoch_all_layer.csv', index=False)



# ------------------------Learning-----------------------------
# Train Epoch: 1 [0/50000 (0)]	Loss: 2.308682
# Train Epoch: 1 [32000/50000 (64)]	Loss: 0.478705
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0001, Accuracy: 9622/10000 (96)
#
# Learn MIA = 0.50435
# Train Epoch: 2 [0/50000 (0)]	Loss: 0.205500
# Train Epoch: 2 [32000/50000 (64)]	Loss: 0.188399
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0001, Accuracy: 9737/10000 (97)
#
# Learn MIA = 0.50495
# Train Epoch: 3 [0/50000 (0)]	Loss: 0.221711
# Train Epoch: 3 [32000/50000 (64)]	Loss: 0.086004
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0001, Accuracy: 9735/10000 (97)
#
# Learn MIA = 0.5037499999999999
# Train Epoch: 4 [0/50000 (0)]	Loss: 0.155906
# Train Epoch: 4 [32000/50000 (64)]	Loss: 0.103759
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0001, Accuracy: 9739/10000 (97)
#
# Learn MIA = 0.5039
# Train Epoch: 5 [0/50000 (0)]	Loss: 0.151670
# Train Epoch: 5 [32000/50000 (64)]	Loss: 0.351536
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0001, Accuracy: 9739/10000 (97)
#
# Learn MIA = 0.50405
# Train Epoch: 6 [0/50000 (0)]	Loss: 0.133251
# Train Epoch: 6 [32000/50000 (64)]	Loss: 0.309790
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0001, Accuracy: 9739/10000 (97)
#
# Learn MIA = 0.50345
# ------------------------Unlearning-----------------------------
# Train Epoch: 1 [0/10000 (0)]	Loss: 0.435046
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0003, Accuracy: 34905/40000 (87)
#
# MIA = 0.5054000000000001
# Train Epoch: 2 [0/10000 (0)]	Loss: 0.532738
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0004, Accuracy: 33438/40000 (84)
#
# MIA = 0.5036499999999999
# Train Epoch: 3 [0/10000 (0)]	Loss: 0.520380
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0005, Accuracy: 32470/40000 (81)
#
# MIA = 0.5054000000000001
# Train Epoch: 4 [0/10000 (0)]	Loss: 0.680416
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0006, Accuracy: 31584/40000 (79)
#
# C:\Users\amart\anaconda3\envs\ldk\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:549: ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
#
# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
#   self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
# MIA = 0.5037999999999999
# Train Epoch: 5 [0/10000 (0)]	Loss: 0.790314
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0007, Accuracy: 30787/40000 (77)
#
# MIA = 0.5106
# Train Epoch: 6 [0/10000 (0)]	Loss: 0.807663
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0008, Accuracy: 29963/40000 (75)
#
# MIA = 0.5102
# Train Epoch: 7 [0/10000 (0)]	Loss: 0.927170
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0010, Accuracy: 29293/40000 (73)
#
# MIA = 0.50785
# Train Epoch: 8 [0/10000 (0)]	Loss: 1.073359
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0011, Accuracy: 28670/40000 (72)
#
# MIA = 0.5096999999999999
# Train Epoch: 9 [0/10000 (0)]	Loss: 1.201924
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0012, Accuracy: 28203/40000 (71)
#
# MIA = 0.509
# Train Epoch: 10 [0/10000 (0)]	Loss: 1.240397
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0013, Accuracy: 27849/40000 (70)
#
# MIA = 0.50945
# Train Epoch: 11 [0/10000 (0)]	Loss: 1.388378
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0014, Accuracy: 27539/40000 (69)
#
# MIA = 0.5112
# Train Epoch: 12 [0/10000 (0)]	Loss: 1.552236
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0015, Accuracy: 27239/40000 (68)
#
# MIA = 0.51
# Train Epoch: 13 [0/10000 (0)]	Loss: 1.545609
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0016, Accuracy: 26959/40000 (67)
#
# MIA = 0.50995
# Train Epoch: 14 [0/10000 (0)]	Loss: 1.703604
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0017, Accuracy: 26722/40000 (67)
#
# MIA = 0.5077999999999999
# Train Epoch: 15 [0/10000 (0)]	Loss: 1.757679
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0018, Accuracy: 26534/40000 (66)
#
# MIA = 0.5079
# Train Epoch: 16 [0/10000 (0)]	Loss: 1.816970
# Learning type: <class 'type'>; droupout 1: Dropout(p=0.25, inplace=False);  droupout 2: Dropout(p=0.5, inplace=False)
#
# Test set: Average loss: 0.0018, Accuracy: 26342/40000 (66)
#
# MIA = 0.5071999999999999
# [0.9622, 0.9737, 0.9735, 0.9739, 0.9739, 0.9739, 0.872625, 0.83595, 0.81175, 0.7896, 0.769675, 0.749075, 0.732325, 0.71675, 0.705075, 0.696225, 0.688475, 0.680975, 0.673975, 0.66805, 0.66335, 0.65855]
# [0.50435, 0.50495, 0.5037499999999999, 0.5039, 0.50405, 0.50345, 0.5054000000000001, 0.5036499999999999, 0.5054000000000001, 0.5037999999999999, 0.5106, 0.5102, 0.50785, 0.5096999999999999, 0.509, 0.50945, 0.5112, 0.51, 0.50995, 0.5077999999999999, 0.5079, 0.5071999999999999]
