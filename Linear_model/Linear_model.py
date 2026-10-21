import torch

x_data = torch.tensor([
    [1.0, 1.0],
    [1.0, 2.0],
    [2.0, 1.0],
    [2.0, 2.0],
    [3.0, 1.0],
    [3.0, 3.0],
    [4.0, 1.0],
    [4.0, 3.0]
])

y_data = torch.tensor([
    [6.0],
    [8.0],
    [9.0],
    [11.0],
    [12.0],
    [16.0],
    [15.0],
    [19.0]
])
class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear,self).__init__()
        self.linear = torch.nn.Linear(2,1)
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = Linear()
criterion = torch.nn.MSELoss(reduction='mean')
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

test = torch.Tensor([[5.0,2.0]])
test_pred = model(test)
print('test:',test_pred.detach())

