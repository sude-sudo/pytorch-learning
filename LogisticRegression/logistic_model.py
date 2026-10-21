import torch

x_data = torch.tensor([[1.0],[2.0],[3.0]
])
y_data = torch.tensor([[0.0],[0.0],[1.0]])
class Logistic(torch.nn.Module):
    def __init__(self):
        super(Logistic,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred = torch.nn.functional.sigmoid(self.linear(x))
        return y_pred
model = Logistic()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('w:',model.linear.weight.detach())
print('b:',model.linear.bias.item())
test = torch.Tensor([100.0])
print(model(test).item())

