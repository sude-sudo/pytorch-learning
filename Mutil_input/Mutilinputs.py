import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Datasets(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x = torch.from_numpy(xy[:, : -1])
        self.y = torch.from_numpy(xy[:, [-1]])
    def __getitem__(self, item):
        return self.x[item],self.y[item]
    def __len__(self):
        return self.len

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
def main():
    dataset = Datasets('diabetes.csv.gz')
    train_load = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
    model = Model()
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

    for epoch in range(1000):
        for i,data in enumerate(train_load):
            inputs,labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            print(epoch,i,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
if __name__ == '__main__':
    main()