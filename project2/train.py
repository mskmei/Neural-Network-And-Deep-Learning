import torch
from torch import nn
from torch import optim
from utils.data_process import get_dataloader
from models.resnet_improve import *
from tqdm import tqdm
import torch
import numpy as np

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
train_loader, valid_loader, test_loader = get_dataloader(batch_size=batch_size)
n_class = 10
model = ResNet18()

model.fc = torch.nn.Linear(512, n_class)
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)

EPOCH = 20
lower = np.Inf
accuracy = []
train_loss_hist = []
val_loss_hist = []
lr = 0.1
count = 0
for epoch in tqdm(range(1, EPOCH + 1)):

    train_loss = 0.0
    val_loss = 0.0
    total_sample = 0
    right_sample = 0

    if count / 10 == 1:
        count = 0
        lr = lr * 0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x).to(device)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)


    model.eval()
    for x, y in valid_loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x).to(device)
        loss = criterion(output, y)
        val_loss += loss.item() * x.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)
        # compare predictions to true label(将预测与真实标签进行比较)
        prediction = pred.eq(y.x.view_as(pred))
        # correct = np.squeeze(prediction.to(device).numpy())
        total_sample += batch_size
        for i in prediction:
            if i:
                right_sample += 1
    print("Accuracy:", 100 * right_sample / total_sample, "%")
    accuracy.append(right_sample / total_sample)

    train_loss_hist.append(train_loss)
    val_loss_hist.append(val_loss)
    train_loss = train_loss / len(train_loader.sampler)
    val_loss = val_loss / len(valid_loader.sampler)

    # 显示训练集与验证集的损失函数
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, val_loss))

    # 如果验证集损失函数减少，就保存模型。
    if val_loss <= lower:
        print('Model updated...')
        torch.save(model.state_dict(), 'resnet18_best.pt')
        lower = val_loss
        count = 0
    else:
        count += 1

print(accuracy)
print(train_loss_hist)
print(val_loss_hist)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(np.arange(EPOCH), train_loss_hist, label="train")
ax.plot(np.arange(EPOCH), val_loss_hist, label="val")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
plt.savefig("loss.png")

fig, ax = plt.subplots()
ax.plot(np.arange(EPOCH), accuracy, label="acc")
# ax.plot(np.arange(EPOCH), val_loss_hist, label="val")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.legend()
plt.savefig("acc.png")
