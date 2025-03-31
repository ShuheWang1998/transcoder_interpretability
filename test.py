import torch


x = torch.tensor([1,2],dtype=(float),requires_grad=True) 

y = torch.tensor([[3],[4]],dtype=(float),requires_grad=True)

w = torch.tensor([[1]],dtype=(float),requires_grad=True)

z = x@y + w     
print(z)

z[0].backward() 

print(x.grad)
print(y.grad)
print((x@y).grad)