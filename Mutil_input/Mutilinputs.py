import numpy as np
import torch


xy = np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32)

x = torch.from_numpy(xy[:,: -1])
y = torch.from_numpy(xy[:,[-1]])
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.activate = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x
model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    print(epoch,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
