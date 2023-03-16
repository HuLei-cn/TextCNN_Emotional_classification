import pandas as pd
import matplotlib.pyplot as plt

# 创建文件
# df = pd.DataFrame(columns=['epoch', 'step', 'train_loss', 'test_loss'])
# df.to_csv('train_acc.csv', index=False)

data = pd.read_csv('train_acc.csv')
epoch = 1
x = data.loc[data['epoch'] == epoch, 'step']
y1 = data.loc[data['epoch'] == epoch, 'train_loss']
plt.plot(x, y1, 'g-', label=u'epoch_1')

y3 = data.loc[data['epoch'] == 3, 'train_loss']
plt.plot(x, y3, 'b-', label=u'epoch_3')

y6 = data.loc[data['epoch'] == 6, 'train_loss']
plt.plot(x, y6, 'y-', label=u'epoch_6')

y8 = data.loc[data['epoch'] == 8, 'train_loss']
plt.plot(x, y8, 'r-', label=u'epoch_8')

plt.title(u'train_loss')
plt.legend()
plt.xlabel('iter')
plt.ylabel('loss')
plt.show()
