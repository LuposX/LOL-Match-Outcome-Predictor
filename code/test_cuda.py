import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())

X_train = torch.FloatTensor([0., 1., 2.])
print(X_train.is_cuda)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

X_train = X_train.to(device)
print(X_train.is_cuda)