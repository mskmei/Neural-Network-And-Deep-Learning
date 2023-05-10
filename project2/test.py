import torch
from torch import nn
from torch import optim
from utils.data_process import get_dataloader
from models.resnet_improve import *
from tqdm import tqdm

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_class = 10
batch_size = 100
train_loader, validation_loader, test_loader = get_dataloader(batch_size=batch_size)
model = ResNet18()
model.fc = torch.nn.Linear(512, n_class)  # 将最后的全连接层修改

model.load_state_dict(torch.load('resnet18_best.pt'))
model = model.to(device)

total_sample = 0
right_sample = 0
model.eval()


def pos(y, y_pred):
    count = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            count += 1
    return count, len(y)


positive = 0
length = 0
for x, y in validation_loader:
    x = x.to(device)
    y = y.to(device)
    res = model(x).to(device)
    pred_values, pred_labels = torch.max(res, 1)
    temp1, temp2 = pos(y, pred_labels)
    positive += temp1
    length += temp2
acc = positive / length

print("Accuracy:", acc)
