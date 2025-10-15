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
        losses = losses.detach().numpy()
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
    attack_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 10), random_state=1)

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
    cal_mia_bb(model, x_train, y_train, x_test, y_test)

    # # Step 1a: Swap axes to PyTorch's NCHW format
    #
    # x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    # x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
    #
    # # Step 2a: Define the loss function and the optimizer
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    #
    # # Step 3: Create the ART classifier
    #
    # classifier = PyTorchClassifier(
    #     model=model,
    #     clip_values=(min_pixel_value, max_pixel_value),
    #     loss=criterion,
    #     optimizer=optimizer,
    #     input_shape=(1, 28, 28),
    #     nb_classes=10,
    # )
    #
    # # Step 4: Train the ART classifier
    #
    # # classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
    #
    # # Step 5: Evaluate the ART classifier on benign test examples
    #
    # predictions = classifier.predict(x_test)
    # accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    # print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    #
    # # Step 6: Generate adversarial test examples
    # attack = FastGradientMethod(estimator=classifier, eps=0.2)
    # x_test_adv = attack.generate(x=x_test)
    #
    # # Step 7: Evaluate the ART classifier on adversarial test examples
    #
    # predictions = classifier.predict(x_test_adv)
    # accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    # print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    #
    return 0


def cal_mia_bb(model, x_train, y_train, x_test, y_test):
    attack_train_ratio = 0.5
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    bb_attack = MembershipInferenceBlackBox(model)

    # train attack model
    # bb_attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
    #               x_test[:attack_test_size], y_test[:attack_test_size])

    # get inferred values
    inferred_train_bb = bb_attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
    inferred_test_bb = bb_attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
    # check accuracy
    train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
    test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
    acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (
                len(inferred_train_bb) + len(inferred_test_bb))
    print(f"Members Accuracy: {train_acc:.4f}")
    print(f"Non Members Accuracy {test_acc:.4f}")
    print(f"Attack Accuracy {acc:.4f}")
