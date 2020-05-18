import os
import numpy as np
import numpy.testing as npt

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_weights_encoder(net):
    for child in net.children():
        if isinstance(child, (nn.GRU, nn.GRUCell)):
            if isinstance(child, nn.GRU):
                weight_ih = child.weight_ih_l0
                weight_hh = child.weight_hh_l0
                bias_ih = child.bias_ih_l0
                bias_hh = child.bias_hh_l0
            else:
                weight_ih = child.weight_ih
                weight_hh = child.weight_hh
                bias_ih = child.bias_ih
                bias_hh = child.bias_hh
            #print('weight_ih', weight_ih.data.shape)  # [6, 2]
            #print('weight_hh', weight_hh.data.shape)  # [6, 2]
            #print('bias_ih', bias_ih.data.shape)  # [6]
            #print('bias_hh', bias_hh.data.shape)  # [6]

            weight_ih.data.fill_(0)
            weight_ih.data[3, :].fill_(0)  # output gate: >0 => old state, <0 => new value
            weight_ih.data[5, :].fill_(0.1)  # new value (tanh is applied after that)

            weight_hh.data.fill_(0.1)
            weight_hh.data[:, 1:].fill_(-0.1)

            bias_ih.data.fill_(0)
            bias_hh.data.fill_(0)

        elif isinstance(child, nn.Embedding):
            #print('emb', child.weight.data.shape)
            child.weight.data = child.weight.data = torch.tensor([
                [  1.,  -1.],
                [ -0.5,  0.2],
                [  0.1, -0.4],
                [ -0.3,  0.4],
                [ -0.1,  0.8],
            ])


def set_weights_decoder(net):
    for child in net.children():
        #print(child)
        if isinstance(child, (nn.GRU, nn.GRUCell)):
            if isinstance(child, nn.GRU):
                weight_ih = child.weight_ih_l0
                weight_hh = child.weight_hh_l0
                bias_ih = child.bias_ih_l0
                bias_hh = child.bias_hh_l0
            else:
                weight_ih = child.weight_ih
                weight_hh = child.weight_hh
                bias_ih = child.bias_ih
                bias_hh = child.bias_hh
            #print('weight_ih', weight_ih.data.shape)  # [6, 2]
            #print('weight_hh', weight_hh.data.shape)  # [6, 2]
            #print('bias_ih', bias_ih.data.shape)  # [6]
            #print('bias_hh', bias_hh.data.shape)  # [6]

            weight_ih.data.fill_(0)
            weight_ih.data[3, :].fill_(0)  # output gate: >0 => old state, <0 => new value
            weight_ih.data[5, :].fill_(0.1)  # new value (tanh is applied after that)

            weight_hh.data.fill_(0.1)
            weight_hh.data[:, 1:].fill_(-0.1)

            bias_ih.data.fill_(0)
            bias_hh.data.fill_(0)

        elif isinstance(child, nn.Embedding):
            #print('emb', child.weight.data.shape)
            child.weight.data = child.weight.data = torch.tensor([
                [  1.,  -1.],
                [ -0.5,  0.2],
                [  0.1, -0.4],
                [ -0.3,  0.4],
                [ -0.1,  0.8],
            ])
        elif isinstance(child, nn.Linear):
            #print('linear', child.weight.data.shape)
            child.weight.data = torch.tensor([
                [ 1.,  0.],
                [ 0.,  1.],
                [ 0.5, 0.],
                [ 0.,  0.5],
                [ 0.5, 0.5],
            ])
            child.bias.data.fill_(0)


def test_Encoder(Encoder):
    with torch.no_grad():
        net = Encoder(src_dictionary_size=5, embed_size=2, hidden_size=2)
        set_weights_encoder(net)

        pad_seqs = torch.tensor([
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 0]
        ])  # (max_seq_length, batch_size)
        seq_lengths = [4, 2]
        hidden = net.init_hidden(batch_size=2)

        outputs, new_hidden = net.forward(pad_seqs, seq_lengths, hidden)

        expected = torch.tensor([
            [ 0.0000, -0.0150],
            [ 0.0004, -0.0221],
            [ 0.0007, -0.0055],
            [ 0.0005,  0.0323]
        ])
        print('outputs[:, 0, :]:\n', outputs[:, 0, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,0,:], expected, atol=1e-4), "outputs do not match expected values"

        expected = torch.tensor([
            [ 0.0000, -0.0150],
            [ 0.0004, -0.0021]
        ])
        print('outputs[:2, 1, :]:\n', outputs[:2, 1, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:2,1,:], expected, atol=1e-4), "outputs do not match expected values"

        expected = torch.tensor([[
            [ 0.0005,  0.0323],
            [ 0.0004, -0.0021]
        ]])
        print('new_hidden:\n', new_hidden)
        print('expected:\n', expected)
        assert torch.allclose(new_hidden, expected, atol=1e-4), "new_hidden does not match expected value"
        print('Success')


def test_Decoder_no_forcing(Decoder):
    # Test without teaching_forcing
    with torch.no_grad():
        net = Decoder(tgt_dictionary_size=5, embed_size=2, hidden_size=2)
        set_weights_decoder(net)

        pad_target_seqs = torch.tensor([
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 0]
        ])

        hidden = torch.tensor([
            [1., -1.],
            [1., -1.],
        ]).view(1, 2, 2)
        outputs, new_hidden = net.forward(hidden, pad_target_seqs, teacher_forcing=False)

        #expected = torch.tensor([
        #    [-1.1246, -2.2242, -1.4241, -1.9740, -1.6744],
        #    [-1.3299, -1.9099, -1.5016, -1.7916, -1.6199],
        #    [-1.4575, -1.7559, -1.5531, -1.7023, -1.6067],
        #    [-1.5294, -1.6809, -1.5817, -1.6574, -1.6052]
        #])
        expected = torch.tensor([
            [-1.1366, -2.1924, -1.4361, -1.9640, -1.6645],
            [-1.3540, -1.8630, -1.5249, -1.7793, -1.6085],
            [-1.4899, -1.7024, -1.5838, -1.6901, -1.5962],
            [-1.5665, -1.6246, -1.6166, -1.6457, -1.5956]
        ])

        print('outputs[:, 0, :]:\n', outputs[:, 0, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,0,:], expected, atol=1e-4), "outputs do not match expected values"

        print('outputs[:, 1, :]:\n', outputs[:, 1, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,1,:], expected, atol=1e-4), "outputs do not match expected values"

        #expected = torch.tensor([[[ 0.1045, -0.0470]]])
        expected = torch.tensor([[
            [0.1003, 0.0421],
            [0.1003, 0.0421]
        ]])
        print('new_hidden:\n', new_hidden)
        print('expected:\n', expected)
        assert torch.allclose(new_hidden, expected, atol=1e-4), "new_hidden does not match expected value"

        print('Success')


def test_Decoder_with_forcing(Decoder):
    # Test with teaching_forcing
    with torch.no_grad():
        net = Decoder(tgt_dictionary_size=5, embed_size=2, hidden_size=2)
        set_weights_decoder(net)

        pad_target_seqs = torch.tensor([
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 0]
        ])

        hidden = torch.tensor([
            [1., -1.],
            [1., -1.],
        ]).view(1, 2, 2)
        outputs, new_hidden = net.forward(hidden, pad_target_seqs, teacher_forcing=True)
        expected = torch.tensor([
            [-1.1366, -2.1924, -1.4361, -1.9640, -1.6645],
            [-1.3414, -1.8877, -1.5123, -1.7854, -1.6146],
            [-1.4662, -1.7419, -1.5607, -1.6986, -1.6040],
            [-1.5418, -1.6619, -1.5932, -1.6532, -1.6019]
        ])
        print('outputs[:, 0, :]:\n', outputs[:, 0, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,0,:], expected, atol=1e-4), "outputs do not match expected values"

        expected = torch.tensor([
            [-1.1366, -2.1924, -1.4361, -1.9640, -1.6645],
            [-1.3398, -1.8909, -1.5107, -1.7862, -1.6154],
            [-1.4706, -1.7343, -1.5652, -1.6971, -1.6024],
            [-1.5557, -1.6402, -1.6069, -1.6492, -1.5979]
        ])
        print('outputs[:, 1, :]:\n', outputs[:, 1, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,1,:], expected, atol=1e-4), "outputs do not match expected values"

        expected = torch.tensor([[
            [ 0.1028, -0.0173],
            [ 0.1025,  0.0180]
        ]])
        print('new_hidden:\n', new_hidden)
        print('expected:\n', expected)
        assert torch.allclose(new_hidden, expected, atol=1e-4), "new_hidden does not match expected value"

        print('Success')


def test_Decoder_generation(Decoder):
    # Test in generation mode
    with torch.no_grad():
        net = Decoder(tgt_dictionary_size=5, embed_size=2, hidden_size=2)
        set_weights_decoder(net)

        hidden = torch.tensor([
            [1., -1.],
            [1., -1.],
        ]).view(1, 2, 2)
        outputs, new_hidden = net.forward(hidden, None, teacher_forcing=False)

        expected = torch.tensor([
            [-1.1366, -2.1924, -1.4361, -1.9640, -1.6645],
            [-1.3540, -1.8630, -1.5249, -1.7793, -1.6085],
            [-1.4899, -1.7024, -1.5838, -1.6901, -1.5962],
            [-1.5665, -1.6246, -1.6166, -1.6457, -1.5956],
            [-1.6074, -1.5869, -1.6333, -1.6230, -1.5972],
            [-1.6125, -1.5923, -1.6252, -1.6151, -1.6024],
            [-1.6151, -1.5950, -1.6212, -1.6111, -1.6050],
            [-1.6164, -1.5963, -1.6192, -1.6091, -1.6064],
            [-1.6170, -1.5970, -1.6182, -1.6081, -1.6070],
            [-1.6173, -1.5973, -1.6177, -1.6077, -1.6073]
        ])
        print('outputs[:, 0, :]:\n', outputs[:, 0, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,0,:], expected, atol=1e-4), "outputs do not match expected values"

        print('outputs[:, 1, :]:\n', outputs[:, 1, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,1,:], expected, atol=1e-4), "outputs do not match expected values"

        expected = torch.tensor([[
            [0.0006, 0.0207],
            [0.0006, 0.0207]
        ]])
        print('new_hidden:\n', new_hidden)
        print('expected:\n', expected)
        assert torch.allclose(new_hidden, expected, atol=1e-4), "new_hidden does not match expected value"

        print('Success')
