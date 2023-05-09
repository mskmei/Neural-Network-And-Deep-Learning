import torch
from torch import nn
from torch import optim
from utils.data_process import get_dataloader
from models.resnet_improve import *
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else 'cpu'

batch_size = 128
n_class = 10
EPOCH = 250
loss_inf = -float("inf")
learning_rate = 0.1
decay = 0.5

train_loader, validation_loader, test_loader = get_dataloader(batch_size=batch_size)
model = ResNet18()

model.fc = nn.Linear(512, n_class)
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)


# Begin training
train_loss_hist = []
valid_loss_hist = []
acc_hist = []
count = 0

def pos(y, y_pred):
    count = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            count+=1
    return count, len(y)

for epoch in tqdm(1, range(EPOCH)+1):
    train_loss = 0
    val_loss = 0
    if count/10 == 1:
        learning_rate*=decay
        count = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # Train process
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        res = model(x).to(device)
        loss = criterion(res, y)
        loss.backward()
        optimizer.step()
        train_loss += (loss.item()*x.size(0))

    model.eval()
    positive = 0
    length = 0
    for x, y in validation_loader:
        x = x.to(device)
        y = y.to(device)
        res = model(x).to(device)
        loss = criterion(res, y)
        val_loss += (loss.item()*x.size(0))
        pred_values, pred_labels = torch.max(res, 1)
        temp1, temp2 = pos(y, pred_labels)
        positive+=temp1
        length+=temp2
    acc = positive/length
    train_loss = train_loss/len(train_loader.sampler)
    val_loss = val_loss/len(validation_loader.sampler)
    print('Epoch: {} \tAcc:{:.6f}\t Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, acc, train_loss, valid_loss))
    acc_hist.append(acc)
    train_loss_hist.append(train_loss)
    valid_loss_hist.append(val_loss)

    if val_loss <= loss_inf:
        torch.save(model.state_dict(), 'checkpoint/resnet18_cifar10.pt')
        loss_inf = val_loss
        count = 0
    else:
        count += 1

print(acc_hist, train_loss_hist, valid_loss_hist)

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(np.arange(EPOCH), train_loss_hist, label="train")
ax.plot(np.arange(EPOCH), valid_loss_hist, label="val")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
plt.savefig("loss.png")

fig, ax = plt.subplots()
ax.plot(np.arange(EPOCH), acc_hist, label="accuracy")
# ax.plot(np.arange(EPOCH), valid_loss_hist, label="val")
ax.set_xlabel("epoch")
ax.set_ylabel("acc")
ax.legend()
plt.savefig("acc.png")


