import torch
import torch.nn as nn
import math

class Block(nn.Module):
    def __init__(self, activation : str, residual : bool, in_channels = 32, out_channels = 32, kernel_size=5, stride=1, padding=2):
        super(Block, self).__init__()
        
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        
        if activation in ['Siren', 'SIREN']:
            self.activation = torch.sin
        elif activation in ['ReLU']:
            self.activation = nn.ReLU(inplace = True)
    
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        
        if self.residual == True:
            x = x + identity
        
        return x

class Network(nn.Module):
    def __init__(self, edge_aware : bool, activation : str, residual : bool, tracking_point_list = ['']):
        super(Network, self).__init__()
        self.edge_aware = edge_aware
        self.residual = residual
        self.activation = activation
        
        if tracking_point_list == 'ALL':
            self.tracking_point_list = ['first', '0', '1', '2', '3', '4', '5', 'last', 'output']
        else:
            self.tracking_point_list = tracking_point_list
        
        if edge_aware == True:
            self.conv_first = nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2)
        else:
            self.conv_first = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
            
        self.block0 = Block(activation = self.activation, residual = False)
        self.block1 = Block(activation = self.activation, residual = self.residual)
        self.block2 = Block(activation = self.activation, residual = self.residual)
        self.block3 = Block(activation = self.activation, residual = self.residual)
        self.block4 = Block(activation = self.activation, residual = self.residual)
        self.block5 = Block(activation = self.activation, residual = self.residual)
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)


    def forward(self, x):
        
        outputs = {}
        
        x = self.conv_first(x)
        if 'first' in self.tracking_point_list: outputs['first'] = x
        
        x = self.block0(x)
        if '0' in self.tracking_point_list: outputs['0'] = x
        
        x = self.block1(x)
        if '1' in self.tracking_point_list: outputs['1'] = x
        
        x = self.block2(x)
        if '2' in self.tracking_point_list: outputs['2'] = x
        
        x = self.block3(x)
        if '3' in self.tracking_point_list: outputs['3'] = x
        
        x = self.block4(x)
        if '4' in self.tracking_point_list: outputs['4'] = x
        
        x = self.block5(x)
        if '5' in self.tracking_point_list: outputs['5'] = x
        
        x = self.conv_last(x)
        if 'last' in self.tracking_point_list: outputs['last'] = x
        
        return x, outputs
    

if __name__=='__main__':
    # create a sample tensor
    x = torch.randn(1, 1, 28, 28)

    # create an instance of the Network class
    network = Network(activation='Siren', residual=True)

    # pass the tensor through the network
    output, _ = network(x)

    # print the shape of the output tensor
    print(output.shape)