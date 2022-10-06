# -*- coding: utf-8 -*-
import torch
import math
import numpy as np


class Model1(torch.nn.Module):
    def __init__(self, h=3, w=1, use_cuda=False):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(h, w),
            torch.nn.Flatten(0, 1)
        )
        if use_cuda: self.nn.cuda()

    def forward(self, x):
        return self.nn(x)

    def string(self):
        linear_layer = self.nn[0]
        return f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + ' \
               f'{linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3'


def fit_model1(x, y, learning_rate=1e-3, use_cuda=False):
    x = torch.from_numpy(x.astype('float32'))
    y = torch.from_numpy(y.astype('float32'))
    # Prepare the input tensor (x, x^2, x^3).
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)
    if use_cuda: xx, y = xx.cuda(), y.cuda()

    # Use the nn package to define our model and loss function.
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(3, 1),
    #     torch.nn.Flatten(0, 1)
    # )
    model = Model1(3, 1, use_cuda=use_cuda)

    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use RMSprop; the optim package contains many other
    # optimization algorithms. The first argument to the RMSprop constructor tells the
    # optimizer which Tensors it should update.

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    for t in range(2000):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(xx)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

    print(model.string())


def demo1():
    # Create Tensors to hold input and outputs.
    x = np.linspace(-math.pi, math.pi, 2000)
    y = np.sin(x)
    fit_model1(x, y, use_cuda=True)


class Model2(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=1, use_cuda=False):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(8, output_dim),
        )
        if use_cuda: self.nn.cuda()

    def forward(self, x):
        return self.nn(x)

    def string(self):
        return f'bias: {self.nn[0].bias}, weight: {self.nn[0].weight}'


def fit_model2(X, Y, model, learning_rate=1e-3, train_rate=0.8, use_cuda=False):
    num_sample = len(X)
    X = torch.from_numpy(X.astype('float32'))
    Y = torch.from_numpy(Y.astype('float32'))
    X, Y = X.unsqueeze(0), Y.unsqueeze(0)
    if use_cuda: X, Y = X.cuda(), Y.cuda()

    ind = int(num_sample*train_rate)
    X_train,  Y_train = X[:,:ind, :],  Y[:,:ind,:]
    X_test,  Y_test = X[:,ind:,:],  Y[:,ind:,:]

    # Use the nn package to define our model and loss function.
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(3, 1),
    #     torch.nn.Flatten(0, 1)
    # )
    # model = Model2(4, 4, use_cuda=use_cuda)

    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use RMSprop; the optim package contains many other
    # optimization algorithms. The first argument to the RMSprop constructor tells the
    # optimizer which Tensors it should update.

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(20000):
        # Forward pass: compute predicted y by passing x to the model.
        Y_pred = model(X_train)

        # Compute and print loss.
        loss = loss_fn(Y_pred[:,:,:3], Y_train[:,:,:3])   # remove last line
        if t % 1000 == 99:
            print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

    Y_pred = model(X_test)

    # Compute and print loss.
    loss = loss_fn(Y_pred[:, :, :3], Y_test[:, :, :3])  # remove last line
    print(f'test error: {loss.item()}')


    torch.save(model.state_dict(), 'test_model.pth')
    # print(model.string())

def demo2():
    # image pixel values
    Xi, Yi = np.meshgrid(range(600), range(400))
    Xi, Yi = Xi.reshape((1, -1)), Yi.reshape((1, -1))
    Zi = 900 * np.ones(Xi.shape, 'int')

    Oi = np.concatenate((Xi,Yi,Zi, np.ones(Xi.shape, 'int')), axis=0)

    # Transformation matrix
    # Translate
    tx, ty, tz = 10, 20, 30
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    #scale
    sx, sy, sz = 0.5, 2, 4
    S = np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])
    # rotate
    thetax, thetay, thetaz = np.pi / 3, np.pi / 4, np.pi / 6
    # thetax, thetay, thetaz = 0, 0, 0
    sinx, cosx = np.sin(thetax), np.cos(thetax)
    Rx = np.array([[1, 0, 0, 0], [0, cosx, sinx, 0], [0, -sinx, cosx, 0], [0, 0, 0, 1]])
    siny, cosy = np.sin(thetay), np.cos(thetay)
    Ry = np.array([[cosy, 0, -siny, 0], [0, 1, 0, 0], [siny, 0, cosy, 0], [0, 0, 0, 1]])
    sinz, cosz = np.sin(thetaz), np.cos(thetaz)
    Rz = np.array([[cosz, -sinz, 0, 0], [sinz, cosz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # shear
    Sh = np.array([[1, np.cos(np.pi/4), 0,0], [0,0,0,0], [0,0,1,0], [0,0,0,1]])

    M = np.dot(T,np.dot(R,S))

    # calc output
    Oc = np.dot(M, Oi)
    Oc[2,:] += np.random.randn(Oc.shape[1]) * 10

    model = Model2(4, 4, use_cuda=True)

    #
    fit_model2(Oi.transpose(), Oc.transpose(), model, use_cuda=True, learning_rate=10e-4)

    #
    model.load_state_dict(torch.load('test_model.pth'))

    print(f'M: {M}')
    print(f'model weight: {model.nn[0].weight}')
    print(f'model bias: {model.nn[0].bias}')






    aa = 1


if __name__ == '__main__':
    demo2()

