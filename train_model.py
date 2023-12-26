import torch
import torch.nn as nn
import torch.nn.functional as F


from models import CNN
from common_utils import train_model, get_sets
from metrics import MSE
from floopers import FuncTrain, Processor, identity

loss = FuncTrain([MSE, identity])


def save_func(input, args):
    pass


processor = Processor()


def main():
    device = torch.device("cuda")

    def loss(predict, yhat, data):
        return F.cross_entropy(predict, yhat)

    model = CNN()
    train_loader, test_loader = get_sets()
    train_model(
        model, device, train_loader, test_loader, loss, exp_n=f"./history/debug/"
    )


if __name__ == "__main__":
    main()
