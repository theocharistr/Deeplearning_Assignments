import os
import numpy as np
import numpy.testing as npt

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def plot_images(images, n_rows=1):
    fig, axs = plt.subplots(n_rows, images.size(0) // n_rows)
    for ax, img in zip(axs.flat, images):
        ax.matshow(img[0].cpu().numpy(), cmap=plt.cm.Greys)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tight_layout(w_pad=0)


def set_weights_lenet(net):
    for param in net.parameters():
        if param.shape == torch.Size([16, 1, 5, 5]):
            param.data[:8] = 1/25
            param.data[8:] = -1/25
        elif param.shape == torch.Size([16]):
            param.data[:] = 0
        elif param.shape == torch.Size([32, 16, 5, 5]):
            param.data[:16] = -1/200
            param.data[16:] = 1/200
        elif param.shape == torch.Size([32]):
            param.data[:] = 0
        elif param.shape == torch.Size([120, 512]):
            param.data[:60] = 1/235.52
            param.data[60:] = -1/235.52
        elif param.shape == torch.Size([120]):
            param.data[:] = 0
        elif param.shape == torch.Size([84, 120]):
            param.data[:42] = 1/60
            param.data[42:] = -1/60
        elif param.shape == torch.Size([84]):
            param.data[:] = 0
        elif param.shape == torch.Size([10, 84]):
            param.data[:5] = 1/42
            param.data[5:] = -1/42
        elif param.shape == torch.Size([10]):
            param.data[:] = 0


def test_LeNet5(LeNet5):
    x = torch.ones(1, 1, 28, 28)
    x[0, 0, :14] = -1

    net = LeNet5()
    set_weights_lenet(net)
    y = net(x)
    expected = torch.Tensor([1., 1., 1., 1., 1., -1., -1., -1., -1., -1.])
    
    print('y:', y)
    print('expected:', expected)
    assert torch.allclose(y, expected), "y does not match expected value."
    
    print('Success')


def set_weights(module, weight):
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        if module.bias is not None:
            module.bias.data.fill_(0)
        module.weight.data.fill_(weight)
        return

    for child in module.children():
        set_weights(child, weight)


def disable_batch_norm(module):
    if isinstance(module, nn.BatchNorm2d):
        module.momentum = 0
        module.weight.data.fill_(1)
        module.bias.data.fill_(0)
        module.running_var.fill_(1)
        module.running_mean.fill_(0)
        return

    for child in module.children():
        disable_batch_norm(child)


def set_batch_norm(block):
    for child in block.children():
        if isinstance(child, nn.BatchNorm2d):
            child.momentum = 0
            child.weight.data.fill_(1)
            child.bias.data.fill_(0)
            child.running_var.fill_(0.25)
            child.running_mean.fill_(0)
        elif isinstance(child, nn.Sequential):
            set_batch_norm(child)


def set_weights_vgg(net, n_channels):
    for param in net.parameters():
        if param.shape == torch.Size([n_channels, 1, 3, 3]):
            param.data[:1] = 1/9
            param.data[1:] = -1/9
        elif param.shape == torch.Size([n_channels]):
            param.data[:] = 0
        elif param.shape == torch.Size([n_channels, n_channels, 3, 3]):
            param.data[:1] = 1/9
            param.data[1:] = -1/9
        elif param.shape == torch.Size([2*n_channels, n_channels, 3, 3]):
            param.data[:2] = 1/9
            param.data[2:] = -1/9
        elif param.shape == torch.Size([2*n_channels]):
            param.data[:] = 0
        elif param.shape == torch.Size([2*n_channels, 2*n_channels, 3, 3]):
            param.data[:2] = 1/18
            param.data[2:] = -1/18
        elif param.shape == torch.Size([3*n_channels, 2*n_channels, 3, 3]):
            param.data[:3] = 1/18
            param.data[3:] = -1/18
        elif param.shape == torch.Size([3*n_channels]):
            param.data[:] = 0
        elif param.shape == torch.Size([2*n_channels, 3*n_channels, 1, 1]):
            param.data[:2] = 1/3
            param.data[2:] = -1/3
        elif param.shape == torch.Size([n_channels, 2*n_channels, 1, 1]):
            param.data[:] = 1/2
            param.data[1:] = -1/2
        elif param.shape == torch.Size([10, n_channels]):
            param.data[:5] = 1
            param.data[5:] = -1
        elif param.shape == torch.Size([10]):
            param.data[:] = 0


def test_vgg_net(VGGNet):
    x = torch.ones(1, 1, 28, 28)
    x[0, 0, :14] = -1

    n_channels = 16
    net = VGGNet(n_channels)
    net.eval()
    set_weights_vgg(net, n_channels)
    disable_batch_norm(net)
    y = net(x)
    expected = 8.0032 * torch.ones(10)
    expected[5:] = -8.0032
    
    print('y:', y)
    print('expected:', expected)
    assert torch.allclose(y, expected), "y does not match expected value."
    
    print('Success')


def test_Block(Block):
    # Simplest case
    batch_size = 1
    x = torch.ones(batch_size, 1, 3, 3)
    block = Block(in_channels=1, out_channels=1)
    block.eval()
    disable_batch_norm(block)
    set_weights(block, 1)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 1, 3, 3]), "Bad shape of y: y.shape={}".format(y.shape)
    y = y.cpu().data.numpy()
    expected = np.array([
        [26, 36, 26],
        [36, 50, 36],
        [26, 36, 26]
    ]).reshape((batch_size, 1, 3, 3))
    npt.assert_allclose(y, expected, atol=1e-03, err_msg="y does not match expected value.")

    # Increase the number of channels
    batch_size = 1
    x = torch.ones(batch_size, 1, 3, 3)
    block = Block(in_channels=1, out_channels=2)
    block.eval()
    disable_batch_norm(block)
    set_weights(block, 1)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 2, 3, 3]), "Bad shape of y: y.shape={}".format(y.shape)
    y = y.cpu().data.numpy()
    expected = np.array([
        [51, 71, 51],
        [71, 99, 71],
        [51, 71, 51]
    ]).reshape(1, 1, 3, 3)
    expected = np.tile(expected, (1, 2, 1, 1))
    npt.assert_allclose(y, expected, atol=1e-03, err_msg="y does not match expected value.")

    # stride=2
    batch_size = 1
    x = torch.ones(batch_size, 1, 3, 3)
    block = Block(in_channels=1, out_channels=1, stride=2)
    block.eval()
    disable_batch_norm(block)
    set_weights(block, 1)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 1, 2, 2]), "Bad shape of y: y.shape={}".format(y.shape)
    y = y.cpu().data.numpy()
    expected = np.array([
        [17, 17],
        [17, 17],
    ]).reshape(1, 1, 2, 2)
    npt.assert_allclose(y, expected, atol=1e-03, err_msg="y does not match expected value.")

    # Increase the number of channels and stride=2
    batch_size = 1
    x = torch.ones(batch_size, 1, 3, 3)
    block = Block(in_channels=1, out_channels=2, stride=2)
    block.eval()
    disable_batch_norm(block)
    set_weights(block, 1)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 2, 2, 2]), "Bad shape of y: y.shape={}".format(y.shape)
    y = y.cpu().data.numpy()
    expected = np.array([
        [33, 33],
        [33, 33],
    ]).reshape(1, 1, 2, 2)
    expected = np.tile(expected, (1, 2, 1, 1))
    npt.assert_allclose(y, expected, atol=1e-03, err_msg="y does not match expected value.")

    print('Success')


def test_Block_relu(Block):
    # check relus
    batch_size = 1
    x = torch.tensor([
        [-1., 1., -1.],
        [1., -1.,  1.],
        [-1., 1., -1.],
    ]).view(batch_size, 1, 3, 3)
    block = Block(in_channels=1, out_channels=1)
    block.eval()
    disable_batch_norm(block)
    set_weights(block, 1)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 1, 3, 3]), "Bad shape of y: y.shape={}".format(y.shape)
    y = y.cpu().data.numpy()
    expected = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]).reshape(1, 1, 3, 3)
    npt.assert_allclose(y, expected, err_msg="y does not match expected value.")

    print('Success')


def test_Block_batch_norm(Block):
    # Two batch norms in the non-skip part
    batch_size = 1
    x = torch.ones(batch_size, 1, 3, 3)
    block = Block(in_channels=1, out_channels=1)
    block.eval()
    set_batch_norm(block)
    set_weights(block, 1)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 1, 3, 3]), "Bad shape of y: y.shape={}".format(y.shape)
    y = y.cpu().data.numpy()
    expected = np.array([
        [101, 141, 101],
        [141, 197, 141],
        [101, 141, 101]
    ]).reshape((batch_size, 1, 3, 3))
    npt.assert_allclose(y, expected, atol=1e-02, err_msg="y does not match expected value.")

    print('Success')
