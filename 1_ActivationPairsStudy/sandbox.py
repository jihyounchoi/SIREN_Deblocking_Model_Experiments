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
            elif self.activations[i] == 'CS':  # Custom_Siren
                self.activation_functions.append(lambda x: torch.sin(self.omega * x))
                eval(f"nn.init.uniform_(self.conv{i+1}.weight, -1 * math.sqrt(6) / math.sqrt(5 ** 2), math.sqrt(6) / math.sqrt(5 ** 2))")
            else:
                raise ValueError(f"Invalid Activation Type Input: {activations.split(',')[i]}")
        
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


class CustomNetwork(nn.Module):
    # Q> Activation이 필요한 Siren과 그렇지 않은 Siren을 나누면 좋을 것 같은데..
    # 설정 가능하도록 하자. CS (Custom Siren)와 S로 나눈 후, 아래와 같이 입력을 받도록
    # "S,CS,R" -> ,를 기준으로 split -> ['S', 'CS', 'R'] -> Siren - Custom_Siren - ReLU
    # CustomNetwork에서는 여러 Activation Setting을 block-by-block으로 다르게 입력받을 수 있도록,
    # 여러 string들을 원소로 하는 list 형태로 activation을 저장하는 것이 좋을듯
    # omega에 대해서도 동일하게 설정 가능하도록 하였음
    # 이때 activation, omega의 설정 개수는 총 6개가 됨 (first, last conv를 제외한 6개의 block들)
    # Residual 도 가능하도록 하자 (overall skipconnection은 변경 불가능, 각 block에서만 가능하도록)
    
    # activation_string의 작성법 : "R,R,R/S,S,S/R,S,S/R,R,R/S,S,S/R,S,S"
    # omega_string의 작성법 : "100/100/10/100/10/1"
    # residual_string의 작성법 : "T/F/F/T/T/F"
    def __init__(self, activation_string, omega_string, residual_string):
        super(CustomNetwork, self).__init__()
        
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
        
        self.blocks = []
        for i in range(6):
            self.blocks.append(CustomBlock(activations=activation_list[i], residual=residual_list[i], omega=omega_list[i]))
        
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
            
        identity = x

        x = self.conv_first(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_last(x)

        x = x + identity

        return x


    
if __name__ == '__main__':

    # create a random input tensor
    
    # Instantiate CustomNetwork
    activation_string = "R,R,R/CS,CS,S/R,S,S/R,R,R/S,S,S/R,S,S"
    omega_string = "100/100/10/100/10/1"
    residual_string = "T/F/F/T/T/F"
    model = CustomNetwork(activation_string, omega_string, residual_string)

    # Create input tensor
    batch_size = 2
    input_shape = (1, 28, 28)
    x = torch.randn(batch_size, *input_shape)

    # Check output shape
    output_shape = model(x).shape
    expected_shape = (batch_size, 1, 28, 28)
    assert output_shape == expected_shape, f"Expected output shape {expected_shape}, but got {output_shape}"