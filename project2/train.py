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
train_loader,valid_loader,test_loader = get_dataloader(batch_size=batch_size)
n_class = 10
model = ResNet18()

model.fc = torch.nn.Linear(512, n_class) 
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)


n_epochs = 250
valid_loss_min = np.Inf 
accuracy = []
lr = 0.1
counter = 0
for epoch in tqdm(range(1, n_epochs+1)):

    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    
    if counter/10 ==1:
        counter = 0
        lr = lr*0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
 
    model.train() 
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # (正向传递：通过向模型传递输入来计算预测输出)
        output = model(data).to(device)  #（等价于output = model.forward(data).to(device) ）
        # calculate the batch loss（计算损失值）
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        # （反向传递：计算损失相对于模型参数的梯度）
        loss.backward()
        # perform a single optimization step (parameter update)
        # 执行单个优化步骤（参数更新）
        optimizer.step()
        # update training loss（更新损失）
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # 验证集的模型#
    ######################

    model.eval()  # 验证模型
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)    
        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
    print("Accuracy:",100*right_sample/total_sample,"%")
    accuracy.append(right_sample/total_sample)
 
    # 计算平均损失
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # 显示训练集与验证集的损失函数 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # 如果验证集损失函数减少，就保存模型。
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'checkpoint/resnet18_cifar10.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1
