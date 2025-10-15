import torch
import numpy as np
from torch import nn
from sklearn import svm
from sklearn import linear_model, model_selection
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.attacks.inference import membership_inference
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs)
        losses = criterion(logits, targets)
        losses = losses.detach().cpu().numpy()
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    #attack_model = svm.SVC()
    #attack_model = RandomForestClassifier(n_estimators=100,  criterion='gini', max_depth=50, min_samples_split=2, bootstrap=True)
    attack_model = MLPClassifier(hidden_layer_sizes=(500,), activation='relu',
                                 solver='adam', alpha=0.0001, batch_size='auto',
                                 learning_rate='constant', learning_rate_init=0.001,
                                 power_t=0.5, max_iter=200, shuffle=True, random_state=42,
                                 tol=0.0001, verbose=False, warm_start=True, momentum=0.9,
                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                 beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def calculate_mia(model, test_loader, forget_loader):
    mia_score = 1
    forget_losses = compute_losses(model, forget_loader)
    test_losses = compute_losses(model, test_loader)
    rt_samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
    mia_score = simple_mia(rt_samples_mia, labels_mia)
    return mia_score.mean()


def get_mia_ibm(model):
    # Step 1: Load the MNIST dataset

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    # Step 1a: Swap axes to PyTorch's NCHW format

    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    # Step 2a: Define the loss function and the optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Step 3: Create the ART classifier

    mlp_art_model = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    mlp_attack_bb = MembershipInferenceBlackBox(mlp_art_model, attack_model_type='rf')

    attack_train_ratio = 0.5
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    # train attack model
    mlp_attack_bb.fit(x_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size],
                      x_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size])

    # infer
    mlp_inferred_train_bb = mlp_attack_bb.infer(x_train[attack_train_size:].astype(np.float32),
                                                y_train[attack_train_size:])
    mlp_inferred_test_bb = mlp_attack_bb.infer(x_test[attack_test_size:].astype(np.float32), y_test[attack_test_size:])

    # check accuracy
    mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
    mlp_test_acc_bb = 1 - (np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
    mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (
                len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))

    print(f"Members Accuracy: {mlp_train_acc_bb:.4f}")
    print(f"Non Members Accuracy {mlp_test_acc_bb:.4f}")
    print(f"Attack Accuracy {mlp_acc_bb:.4f}")

    return 0