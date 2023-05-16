import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
from resnet_improve import ResNet18
from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from utils.data_process import get_dataloader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(3))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader, val_loader, test_loader = get_dataloader(batch_size=128)




# This function is used to calculate the accuracy of model classification
def get_accuracy(model, loader=val_loader):
    number = 0
    pos = 0
    with torch.no_grad():
        for item in loader:
            x, y = item
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            _, y_pred = torch.max(out.data, 1)
            number += y.size(0)
            pos += (y_pred==y).sum().item()
    print("Accuracy: %.2f"%(pos/number))
    return pos/number
# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve
        number = 0
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            losses_list.append(loss.cpu().detach())
            _, y_pred = torch.max(prediction.data,1)
            learning_curve[epoch] += (y_pred==y).sum().item()
            number+=y.size(0)

            loss.backward()
            optimizer.step()

        learning_curve[epoch] /= number
    model.eval()
    val_acc = get_accuracy(model)
    return losses_list, learning_curve

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(step, VGG_max, VGG_min, VGG_BN_max, VGG_BN_min):
    fig = plt.figure()
    plt.style.use('ggplot')
    plt.plot(step, VGG_max,c='green')
    plt.plot(step, VGG_min,c="green")
    plt.fill_between(step, VGG_max, VGG_min, color="lightgreen", label="VGG")
    # plt.plot(step, VGG_BN_max, c='red')
    # plt.plot(step, VGG_BN_min, c="red")
    # plt.fill_between(step, VGG_BN_max, VGG_BN_min, color="lightcoral", label="VGGm with BN")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Figure')
    plt.legend(loc='best')
    plt.savefig('Train-loss-res.png', dpi=300)

def plot_accuracy(step, VGG_acc, VGG_BN_acc):
    fig = plt.figure()
    plt.style.use('ggplot')
    plt.plot(step, VGG_acc, c='green', label="VGG")
    plt.plot(step, VGG_BN_acc, c='red', label="VGG with BN")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Figure')
    plt.legend(loc='best')
    plt.xticks(range(0, 23))
    plt.savefig('Train-acc.png', dpi=300)



# Train your model
# feel free to modify
if __name__ == '__main__':
    VGG_loss = []
    VGG_BN_loss = []
    VGG_acc = []
    VGG_BN_acc = []
    lr_list = [1e-4, 5e-4, 1e-3, 1.5e-3]  # 这里选择2e-3会导致loss不下降
    span = 20
    # lr_list=[2e-3, 1e-4,5e-4]
    batch_size = 128
    EPOCH = 50
    set_random_seeds(seed_value=2022, device=device)

    for lr in lr_list:
        model = ResNet18()
        model.fc = nn.Linear(512,10)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss, acc = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=EPOCH)
        VGG_loss.append(loss)
        VGG_acc.append(acc)

        # model = VGG_A_BatchNorm()
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # criterion = nn.CrossEntropyLoss()
        # loss_, acc_ = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=EPOCH)
        # VGG_BN_loss.append(loss_)
        # VGG_BN_acc.append(acc_)

    VGG_loss = np.array(VGG_loss)
    # VGG_BN_loss = np.array(VGG_BN_loss)
    VGG_acc = np.array(VGG_acc)
    # VGG_BN_acc = np.array(VGG_BN_acc)

    step = []
    curve_min = []
    curve_max = []
    curve_min_BN = []
    curve_max_BN = []

    VGG_min = np.min(VGG_loss, axis=0).astype(float)
    VGG_max = np.max(VGG_loss, axis=0).astype(float)
    # VGG_BN_min = np.min(VGG_BN_loss, axis=0).astype(float)
    # VGG_BN_max = np.max(VGG_BN_loss, axis=0).astype(float)
    for i in range(len(VGG_min)):
        if i % span == 0:
            curve_min.append(VGG_min[i])
            curve_max.append(VGG_max[i])
            # curve_min_BN.append(VGG_BN_min[i])
            # curve_max_BN.append(VGG_BN_max[i])
            step.append(i)

    # plot_accuracy(np.arange(1, 21), VGG_acc[0], VGG_BN_acc[0])

    plot_loss_landscape(step, curve_max,
                        curve_min)


