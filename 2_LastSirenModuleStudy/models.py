import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomBlock(nn.Module):
    def __init__(self, activations: str, residual: bool, omega: int, in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2):
        super(CustomBlock, self).__init__()
        
        self.activations = activations.split(',')
        self.omega = omega
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.residual = residual
        
        # Check input
        if len(self.activations) != 3:
            raise ValueError(f"Invalid activation setting input: {activations}, maybe invalid length input")
        
        # Define activation functions
        self.activation_functions = []
        for i in range(3):
            if self.activations[i] == 'R':
                self.activation_functions.append(nn.ReLU(inplace=True))
            elif self.activations[i] == 'S':
                self.activation_functions.append(lambda x: torch.sin(self.omega * x))
            elif self.activations[i] == 'SC':  # SIREN_Custom
                self.activation_functions.append(lambda x: torch.sin(self.omega * x))
                eval(f"nn.init.uniform_(self.conv{i+1}.weight, -1 * math.sqrt(6) / math.sqrt(5 ** 2), math.sqrt(6) / math.sqrt(5 ** 2))")
            elif self.activations[i] == 'S01': # SIREN_01
                self.activaion_functions.append(lambda x : 0.5 * torch.sin(self.omega * x) + 0.5)
            else:
                raise ValueError(f"Invalid Activation Type Input: {activations.split(',')[i]}")
        
        print(self.activation_functions)
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.activation_functions[0](x)
        x = self.conv2(x)
        x = self.activation_functions[1](x)
        x = self.conv3(x)
        x = self.activation_functions[2](x)
        
        if self.residual == True:
            x = x + identity
        
        return x
    
# before_siren : SIREN 직전 값의 저장을 위한 임시 변수
class SirenBlock(nn.Module):
    def __init__(self, omega, in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2):
        super(SirenBlock, self).__init__()
        self.omega = omega
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size, stride, padding, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        siren_1 = x
        x = torch.sin(x * self.omega)
        
        x = self.conv2(x)
        siren_2 = x
        x = torch.sin(x * self.omega)
        
        x = self.conv3(x)
        siren_3 = x
        # x = 0.5 * torch.sin(x * self.omega) + 0.5
        
        return x, siren_1, siren_2, siren_3

class CustomNetwork(nn.Module):
    
    # 1. Global Skip을 지움 (옵션상으로는 사용 가능하게끔 남겨두기는 하였으나)
    # 2. 마지막의 single convolution layer를 지움
    # 3. 마지막 block의 activation을 모두 siren으로 바꾸고, Local Skip Connection을 지움
    # 4. 마지막 block의 마지막 activation을 S01로 변경
    def __init__(self, activation_string, omega_string, residual_string, global_skip, tracking_point_list = ['']):
        super(CustomNetwork, self).__init__()
        
        self.tracking_point_list = tracking_point_list
        self.global_skip = global_skip
        
        #################### unpack string #######################
        activation_list = activation_string.split('/')
        omega_list = [int(x) for x in omega_string.split('/')]
        
        residual_list = []
        for x in residual_string.split('/'):
            if x == 'T':
                residual_list.append(True)
            elif x == 'F':
                residual_list.append(False)
            else:
                raise ValueError(f"Unexpected bool type : {x}")
        ###########################################################

        ################## to check if inputs are invalid ##################
        if len(omega_list) != 6:
            raise ValueError(f"Expected 6 omegas, but got {len(omega_list)}.")
        if len(activation_list) != 6:
            raise ValueError(f"Expected 6 activation settings, but got {len(activation_list)}.")
        if len(residual_list) != 6:
            raise ValueError(f"Expected 6 residuals, but got {len(residual_list)}.")
        ####################################################################
        
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        
        self.block0 = CustomBlock(activations=activation_list[0], residual=residual_list[0], omega=omega_list[0])
        self.block1 = CustomBlock(activations=activation_list[1], residual=residual_list[1], omega=omega_list[1])
        self.block2 = CustomBlock(activations=activation_list[2], residual=residual_list[2], omega=omega_list[2])
        self.block3 = CustomBlock(activations=activation_list[3], residual=residual_list[3], omega=omega_list[3])
        self.block4 = CustomBlock(activations=activation_list[4], residual=residual_list[4], omega=omega_list[4])
        self.block5 = SirenBlock(omega=omega_list[5]) # activation_list[5], residual_list[5] are dismissed (임시)

    def forward(self, x):
        
        outputs = {}
            
        identity = x

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
        
        x, outputs['siren_1'], outputs['siren_2'], outputs['siren_3'] = self.block5(x)
        
        if '5' in self.tracking_point_list: outputs['5'] = x
        
        # NO LAST SINGLE CONVOLUTION
        
        # Maybe... No Global Skip
        if self.global_skip:
            x = x + identity
            
        if 'output' in self.tracking_point_list: outputs['output'] = x

        return x, outputs


    
if __name__ == '__main__':

    # create a random input tensor
    
    # Instantiate CustomNetwork
    activation_string = "R,R,R/R,SC,S/R,S,S/R,R,R/S,S,S/R,S,S01"
    omega_string = "100/100/10/100/10/1"
    residual_string = "T/F/F/T/T/T"
    model = CustomNetwork(activation_string, omega_string, residual_string)

    # Create input tensor
    batch_size = 2
    input_shape = (1, 28, 28)
    x = torch.randn(batch_size, *input_shape)

    # Check output value and shape
    output = model(x)
    # print(output)
    output_shape = output.shape
    expected_shape = (batch_size, 1, 28, 28)
    assert output_shape == expected_shape, f"Expected output shape {expected_shape}, but got {output_shape}"
