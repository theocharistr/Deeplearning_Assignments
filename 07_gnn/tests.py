import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from IPython import display

import torch
import torchvision.utils as utils
import torch.nn.functional as F
import torch.nn as nn


def test_GNN_forward(GNN):
    with torch.no_grad():
        src_ids = torch.LongTensor([0, 1, 0])
        dst_ids = torch.LongTensor([1, 2, 2])
        #src_ids = torch.LongTensor([0, 1])
        #dst_ids = torch.LongTensor([1, 2])
        n_edges = len(src_ids)

        gnn = GNN(n_iters=2, n_node_inputs=1, n_node_features=2, n_edge_features=3, n_node_outputs=2)
        class MyMsgNet(nn.Module):
            def forward(self, msg_net_inputs):
                n_edges = msg_net_inputs.size(0)
                messages = torch.ones(n_edges, gnn.n_edge_features)
                return messages
        gnn.msg_net = MyMsgNet()

        def set_weights(gnn):
            # TODO: Support of nested chidren
            for child in gnn.children():
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
                        
                    #print(weight_ih.data.shape)  # [6, 1+3]
                    #print(weight_hh.data.shape)  # [6, 2]
                    #print(bias_ih.data.shape)  # [6]
                    #print(bias_hh.data.shape)  # [6]
                    weight_ih.data.fill_(0)
                    weight_ih.data[3, :].fill_(0)  # output gate: >0 => old state, <0 => new value
                    weight_ih.data[5, :].fill_(0.1)  # new value (tanh is applied after that)

                    weight_hh.data.fill_(0.1)
                    weight_hh.data[:, 1:].fill_(-0.1)

                    bias_ih.data.fill_(0)
                    bias_hh.data.fill_(0)

                elif isinstance(child, nn.Linear) and (child.in_features == 2) and (child.out_features == 2):
                    child.weight.data = torch.eye(2)
                    child.bias.data.fill_(0)

        set_weights(gnn)
        node_inputs = torch.Tensor([1, 2, 3]).view(3, 1)
        outputs = gnn.forward(node_inputs, src_ids, dst_ids)  # [n_iters, n_nodes, n_node_outputs]
        expected = torch.tensor([
            [[ 0.0000,  0.0498],
             [ 0.0000,  0.2311],
             [ 0.0000,  0.3581]],
            [[-0.0012,  0.0736],
             [-0.0058,  0.3434],
             [-0.0089,  0.5360]]])
        print('outputs:\n', outputs)
        print('expected:\n', expected)

        assert torch.allclose(outputs, expected, atol=1e-04)
        print('Success')
