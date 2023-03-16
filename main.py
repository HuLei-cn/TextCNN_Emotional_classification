import time
from datetime import timedelta
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from data_preprocess import load_imdb
from utils import set_seed
from model import TextCNN

# 加快运算速度，需要GPU
set_seed()

# 设置参数
BATCH_SIZE = 50
LEARNING_RATE = 0.0009
NUM_EPOCHS = 8


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


start_time = time.time()

train_data, test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU训练
model = TextCNN(vocab).to(device)
# print(model)
criterion = nn.CrossEntropyLoss()  # 获取损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)  # 获取优化器

for epoch in range(1, NUM_EPOCHS + 1):
    print(f'Epoch {epoch}\n' + '-' * 32)
    avg_train_loss = 0
    for batch_idx, (text, label) in enumerate(train_loader):
        # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
        text, label = text.to(device), label.to(device)
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        avg_train_loss += loss
        optimizer.zero_grad()  # 清空梯度（可以不写）
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新网络参数

        if (batch_idx + 1) % 5 == 0:
            print(f"[{(batch_idx + 1) * BATCH_SIZE:>5}/{len(train_loader.dataset):>5}] train loss: {loss:.4f}")
            # get_epoch = epoch
            # get_step = (batch_idx + 1) * BATCH_SIZE
            # get_loss = loss
            # list = [get_epoch, get_step, get_loss]
            # data = pd.DataFrame([list])
            # data.to_csv('train_acc.csv', mode='a', header=False, index=False)

    print(f"Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}\n")

time_dif = get_time_dif(start_time)
print("Time Usage:", time_dif)

# 保存模型
# torch.save(model, 'textcnn.pt')

# 进行测试
acc = 0
for X, y in test_loader:
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        pred = model(X)
        acc += (pred.argmax(1) == y).sum().item()

print(f"Accuracy: {acc / len(test_loader.dataset):.4f}")
