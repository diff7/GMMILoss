import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

PARAMS = {'lr': 1e-3,
          'log_interval': 10,
          'seed': 1,
          'gamma': 0.8,
          'epochs': 10,
          'test_batch_size': 500,
          'batch_size': 256}


def train(model, device, train_loader, optimizer, epoch, loss_fn):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # B = data.shape[0]
        # data = data.reshape(B, -1)
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_fn(output, target, data)
        loss.backward()
        optimizer.step()
        if batch_idx % PARAMS['log_interval'] == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # B = data.shape[0]
            # data = data.reshape(B, -1)
            output = model(data)
            test_loss += F.nll_loss(output, target).detach().cpu().numpy()
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return 100.0 * correct / len(test_loader.dataset), test_loss



def train_model(model, 
                    device, 
                    train_loader, 
                    test_loader, 
                    loss_fn, 
                    metrics_processor, 
                    output_processor,  
                    exp_n=''):

    torch.manual_seed(PARAMS['seed'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['lr'])
    scheduler = StepLR(optimizer, step_size=1, gamma=PARAMS['gamma'])

    correct_list = []
    test_loss = []

    for epoch in range(1, PARAMS['epochs'] + 1):
        train(model, device, train_loader, optimizer, epoch, loss_fn)
        output = test(model, device, test_loader)
        metrics = metrics(output)


        scheduler.step()

    with open(f'{exp_n}_acc.txt', 'w') as f:
        for c in correct_list:
            f.write(str(c)+'\n')

    with open(f'{exp_n}_loss.txt', 'w') as f:
        for c in test_loss:
            f.write(str(c)+'\n')


def get_sets():
    train_kwargs = {"batch_size": PARAMS['batch_size']}
    test_kwargs = {"batch_size": PARAMS['test_batch_size']}

    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.FashionMNIST("../data/fashionMnist", train=True,
                                     download=True, transform=transform)
    dataset2 = datasets.FashionMNIST(
        "../data/fashionMnist", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader
