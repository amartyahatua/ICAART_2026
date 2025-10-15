import os
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from attack_membership_inference import calculate_mia, get_mia_ibm
from model import Net

from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from art.utils import load_mnist
from art.estimators.classification.pytorch import PyTorchClassifier
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.autograd.set_detect_anomaly(True)




def train(args, model, device, train_loader, optimizer, epoch, type, unlearning_type='', turn=0):
    model.train()
    model.type = type
    model.epoch = epoch
    model.turn = turn
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

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    # train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=64)
    # test_loader = torch.utils.data.DataLoader(list(zip(x_test, y_test)), shuffle=True, batch_size=64)
    # dataset_retain, dataset_forget = torch.utils.data.random_split(dataset_train, [40000, 10000],
    #                                                                generator=torch.Generator().manual_seed(42))
    #
    # # train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    # forget_loader = torch.utils.data.DataLoader(dataset_forget, **train_kwargs)
    # retain_loader = torch.utils.data.DataLoader(dataset_retain, **test_kwargs)
    # # test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for turn in range(10):
        print(f'------------------------Turn = {turn}-----------------------------')
        print(f'------------------------Learning-----------------------------')
        for epoch in range(1, 2):
            train(args, model, device, train_loader, optimizer, epoch, 'learning', '', turn)
            test(model, device, test_loader)
            scheduler.step()
            get_mia_ibm(model)
            mia_temp = calculate_mia(model, test_loader, forget_loader)
            print('MIA =', mia_temp)
            mia_score_list.append(mia_temp)

        # mlp_art_model = PyTorchClassifier(
        #     model=model,
        #     loss=criterion,
        #     optimizer=optimizer,
        #     input_shape=(1, 28, 28),
        #     nb_classes=10,
        # )

        # train_pred = np.array([np.argmax(arr) for arr in mlp_art_model.predict(x_train.astype(np.float32))])
        # print('Base model Train accuracy: ', np.sum(train_pred == y_train) / len(y_train))
        #
        # test_pred = np.array([np.argmax(arr) for arr in mlp_art_model.predict(x_test.astype(np.float32))])
        # print('Base model Test accuracy: ', np.sum(test_pred == y_test) / len(y_test))

        print(f'------------------------Unlearning-----------------------------')
        for epoch in range(1, args.epochs + 1):
            # Rank forget rate=rank_fr [0.9581, 0.9761, 0.9766, 0.8962, 0.8526, 0.8254, 0.8094, 0.7991, 0.793]
            # Ordered forget rate=ordered_fr [0.9581, 0.9761, 0.9766, 0.9347, 0.9239, 0.9038, 0.875, 0.8478, 0.8236]
            # Top K forget rate=topK_fr [0.9581, 0.9761, 0.9766, 0.8962, 0.8526, 0.8254, 0.8094, 0.7991, 0.793]
            # Random forget rate=random_fr [0.9581, 0.9761, 0.9766, 0.9433, 0.7824, 0.617, 0.3688, 0.323, 0.3099]

            train(args, model, device, forget_loader, optimizer, epoch, 'unlearning', 'random_fr', turn)
            test(model, device, test_loader)
            scheduler.step()
            # get_mia_ibm(model)
            mia_temp = calculate_mia(model, test_loader, forget_loader)
            print('MIA =', mia_temp)
            mia_score_list.append(mia_temp)
    # if args.save_model:
    # torch.save(model.state_dict(), "../model/top_30_L1_epoch_variable.pt")


if __name__ == '__main__':
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time:.4f} seconds")

    print(accuracy)
    print(mia_score_list)

    accuracy = pd.DataFrame(accuracy)
    mia_score_df = pd.DataFrame(mia_score_list)

    accuracy.to_csv('results/mnist_digit/Result_Random_Forgetting_all_layer_Accuracy.csv', index=False)
    mia_score_df.to_csv('results/mnist_digit/Result_Random_Forgetting_all_layer_MIA.csv', index=False)
