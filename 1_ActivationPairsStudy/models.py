import torch
import torch.nn as nn
import torch.nn.functional as F
import math

######################## LEGACY VERSIONS ############################

class ReLUBlock(nn.Module):
    def __init__(self, in_channels = 32, out_channels = 32, kernel_size=5, stride=1, padding=2):
        super(ReLUBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        
        return x

class ReLUBlock_residual(nn.Module):
    def __init__(self, in_channels = 32, out_channels = 32, kernel_size=5, stride=1, padding=2):
        super(ReLUBlock_residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x + identity
        
        return x

class ReLUNetwork(nn.Module):
    def __init__(self, tracking_point_list = ['']):
        super(ReLUNetwork, self).__init__()
        
        self.tracking_point_list = tracking_point_list
        
        self.relu = nn.ReLU(inplace = True)
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv_block = ReLUBlock()
        self.residualblock1 = ReLUBlock_residual()
        self.residualblock2 = ReLUBlock_residual()
        self.residualblock3 = ReLUBlock_residual()
        self.residualblock4 = ReLUBlock_residual()
        self.residualblock5 = ReLUBlock_residual()
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        
        outputs = {}
        
        identity = x
        
        x = self.conv_first(x)
        if 'first' in self.tracking_point_list: outputs['first'] = x
        
        x = self.relu(x)
        
        x = self.conv_block(x)
        if '0' in self.tracking_point_list: outputs['0'] = x
        
        x = self.residualblock1(x)
        if '1' in self.tracking_point_list: outputs['1'] = x
        
        x = self.residualblock2(x)
        if '2' in self.tracking_point_list: outputs['2'] = x
        
        x = self.residualblock3(x)
        if '3' in self.tracking_point_list: outputs['3'] = x
        
        x = self.residualblock4(x)
        if '4' in self.tracking_point_list: outputs['4'] = x
        
        x = self.residualblock5(x)
        if '5' in self.tracking_point_list: outputs['5'] = x
        
        x = self.conv_last(x)
        if 'last' in self.tracking_point_list: outputs['last'] = x
        
        x = x + identity
        if 'output' in self.tracking_point_list: outputs['output'] = x
        
        return x, outputs

class RRRBlock(nn.Module):
    def __init__(self, residual : bool, in_channels = 32, out_channels = 32, kernel_size=5, stride=1, padding=2):
        super(RRRBlock, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        
        if self.residual == True:
            x = x + identity
        
        return x

class RRRNetwork(nn.Module):
    def __init__(self, tracking_point_list = ['']):
        super(RRRNetwork, self).__init__()
        
        self.tracking_point_list = tracking_point_list
        
        self.relu = nn.ReLU(inplace = True)
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv_block = RRRBlock(residual=False)
        self.residualblock1 = RRRBlock(residual=True)
        self.residualblock2 = RRRBlock(residual=True)
        self.residualblock3 = RRRBlock(residual=True)
        self.residualblock4 = RRRBlock(residual=True)
        self.residualblock5 = RRRBlock(residual=True)
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        
        outputs = {}
        
        identity = x
        
        x = self.conv_first(x)
        if 'first' in self.tracking_point_list: outputs['first'] = x
        
        x = self.conv_block(x)
        if '0' in self.tracking_point_list: outputs['0'] = x
        
        x = self.residualblock1(x)
        if '1' in self.tracking_point_list: outputs['1'] = x
        
        x = self.residualblock2(x)
        if '2' in self.tracking_point_list: outputs['2'] = x
        
        x = self.residualblock3(x)
        if '3' in self.tracking_point_list: outputs['3'] = x
        
        x = self.residualblock4(x)
        if '4' in self.tracking_point_list: outputs['4'] = x
        
        x = self.residualblock5(x)
        if '5' in self.tracking_point_list: outputs['5'] = x
        
        x = self.conv_last(x)
        if 'last' in self.tracking_point_list: outputs['last'] = x
        
        x = x + identity
        if 'output' in self.tracking_point_list: outputs['output'] = x
        
        return x, outputs

class PlainRRRNetwork(nn.Module):
    
    ##########################################
    # PlainRRRNetwork
    ##########################################
    # Activation function of each block : ReLU - ReLU - ReLU
    # NO RESIDUAL & IDENTITY MAPPING IN EACH BLOCK
    # ONLY OVERALL SINGLE SKIP CONNECTION IN BEGINNING EXISTS
    
    def __init__(self, tracking_point_list = ['']):
        super(PlainRRRNetwork, self).__init__()
        
        self.tracking_point_list = tracking_point_list
        
        self.relu = nn.ReLU(inplace = True)
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv_block = RRRBlock(residual=False)
        self.residualblock1 = RRRBlock(residual=False)
        self.residualblock2 = RRRBlock(residual=False)
        self.residualblock3 = RRRBlock(residual=False)
        self.residualblock4 = RRRBlock(residual=False)
        self.residualblock5 = RRRBlock(residual=False)
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        outputs = {}
        
        identity = x
        
        x = self.conv_first(x)
        if 'first' in self.tracking_point_list: outputs['first'] = x
        
        x = self.conv_block(x)
        if '0' in self.tracking_point_list: outputs['0'] = x
        
        x = self.residualblock1(x)
        if '1' in self.tracking_point_list: outputs['1'] = x
        
        x = self.residualblock2(x)
        if '2' in self.tracking_point_list: outputs['2'] = x
        
        x = self.residualblock3(x)
        if '3' in self.tracking_point_list: outputs['3'] = x
        
        x = self.residualblock4(x)
        if '4' in self.tracking_point_list: outputs['4'] = x
        
        x = self.residualblock5(x)
        if '5' in self.tracking_point_list: outputs['5'] = x
        
        x = self.conv_last(x)
        if 'last' in self.tracking_point_list: outputs['last'] = x
        
        x = x + identity
        if 'output' in self.tracking_point_list: outputs['output'] = x
        
        return x, outputs
    
class SSSBlock(nn.Module):
    def __init__(self, residual : bool, omega : int, in_channels = 32, out_channels = 32, kernel_size=5, stride=1, padding=2):
        super(SSSBlock, self).__init__()
        self.residual = residual
        self.omega = omega
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = torch.sin(self.omega * x)
        x = self.conv2(x)
        x = torch.sin(self.omega * x)
        x = self.conv3(x)
        
        if self.residual == True:
            x = x + identity
        
        return x

class SSSNetwork(nn.Module):
    def __init__(self, omega, tracking_point_list = ['']):
        super(SSSNetwork, self).__init__()
        
        self.tracking_point_list = tracking_point_list
        self.omega = omega
        
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        nn.init.uniform_(self.conv_first.weight, -1 * math.sqrt(6) / math.sqrt(5 ** 2), math.sqrt(6) / math.sqrt(5 ** 2))
        
        self.conv_block = SSSBlock(residual = False, omega = self.omega) 
        self.residualblock1 = SSSBlock(residual = True, omega = self.omega)
        self.residualblock2 = SSSBlock(residual = True, omega = self.omega)
        self.residualblock3 = SSSBlock(residual = True, omega = self.omega)
        self.residualblock4 = SSSBlock(residual = True, omega = self.omega)
        self.residualblock5 = SSSBlock(residual = True, omega = self.omega)
        
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)
        # nn.init.uniform_(self.conv_last.weight, -1 * math.sqrt(6) / math.sqrt(32 * 5 ** 2), math.sqrt(6) / math.sqrt(32 * 5 ** 2))

    def forward(self, x):
        outputs = {}
        
        identity = x
        
        x = self.conv_first(x)
        if 'first' in self.tracking_point_list: outputs['first'] = x
        
        x = self.conv_block(x)
        if '0' in self.tracking_point_list: outputs['0'] = x
        
        x = self.residualblock1(x)
        if '1' in self.tracking_point_list: outputs['1'] = x
        
        x = self.residualblock2(x)
        if '2' in self.tracking_point_list: outputs['2'] = x
        
        x = self.residualblock3(x)
        if '3' in self.tracking_point_list: outputs['3'] = x
        
        x = self.residualblock4(x)
        if '4' in self.tracking_point_list: outputs['4'] = x
        
        x = self.residualblock5(x)
        if '5' in self.tracking_point_list: outputs['5'] = x
        
        x = self.conv_last(x)
        if 'last' in self.tracking_point_list: outputs['last'] = x
        
        x = x + identity
        if 'output' in self.tracking_point_list: outputs['output'] = x
        
        return x, outputs
    
class PlainSSSNetwork(nn.Module):

    ##########################################
    # PlainSirenNetwork
    ##########################################
    # Activation function of each block : SIREN - SIREN - SIREN
    # NO RESIDUAL & IDENTITY MAPPING IN EACH BLOCK
    # ONLY OVERALL SINGLE SKIP CONNECTION IN BEGINNING EXISTS
    def __init__(self, omega, tracking_point_list = ['']):
        super(PlainSSSNetwork, self).__init__()
        
        self.tracking_point_list = tracking_point_list
        self.omega = omega
        
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        # nn.init.uniform_(self.conv_first.weight, -1 * math.sqrt(6) / math.sqrt(5 ** 2), math.sqrt(6) / math.sqrt(5 ** 2))
        
        self.conv_block = SSSBlock(residual = False, omega = self.omega) 
        self.residualblock1 = SSSBlock(residual = False, omega = self.omega)
        self.residualblock2 = SSSBlock(residual = False, omega = self.omega)
        self.residualblock3 = SSSBlock(residual = False, omega = self.omega)
        self.residualblock4 = SSSBlock(residual = False, omega = self.omega)
        self.residualblock5 = SSSBlock(residual = False, omega = self.omega)
        
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)
        # nn.init.uniform_(self.conv_last.weight, -1 * math.sqrt(6) / math.sqrt(32 * 5 ** 2), math.sqrt(6) / math.sqrt(32 * 5 ** 2))

    def forward(self, x):
        outputs = {}
        
        identity = x
        
        x = self.conv_first(x)
        if 'first' in self.tracking_point_list: outputs['first'] = x
        
        x = self.conv_block(x)
        if '0' in self.tracking_point_list: outputs['0'] = x
        
        x = self.residualblock1(x)
        if '1' in self.tracking_point_list: outputs['1'] = x
        
        x = self.residualblock2(x)
        if '2' in self.tracking_point_list: outputs['2'] = x
        
        x = self.residualblock3(x)
        if '3' in self.tracking_point_list: outputs['3'] = x
        
        x = self.residualblock4(x)
        if '4' in self.tracking_point_list: outputs['4'] = x
        
        x = self.residualblock5(x)
        if '5' in self.tracking_point_list: outputs['5'] = x
        
        x = self.conv_last(x)
        if 'last' in self.tracking_point_list: outputs['last'] = x
        
        x = x + identity
        if 'output' in self.tracking_point_list: outputs['output'] = x
        
        return x, outputs

class RRSBlock(nn.Module):
    ##########################################
    # RRSNetwork
    ##########################################
    # Activation function of each block : ReLU - ReLU - SIREN
    # Not any operation after skip connection
    # Use default parameter initialization instead of uniform initialization
    def __init__(self, residual : bool, omega : int, in_channels = 32, out_channels = 32, kernel_size=5, stride=1, padding=2):
        super(RRSBlock, self).__init__()
        self.residual = residual
        self.omega = omega
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = torch.sin(self.omega * x) # only last activation of residual path is SIREN
        
        if self.residual == True:
            x = x + identity
        
        return x

class RRSNetwork(nn.Module):

    def __init__(self, omega, tracking_point_list = ['']):
        super(RRSNetwork, self).__init__()
        
        self.tracking_point_list = tracking_point_list
        self.omega = omega
        
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        # nn.init.uniform_(self.conv_first.weight, -1 * math.sqrt(6) / math.sqrt(5 ** 2), math.sqrt(6) / math.sqrt(5 ** 2))
        
        self.conv_block = SSSBlock(residual = False, omega = self.omega) 
        self.residualblock1 = SSSBlock(residual = True, omega = self.omega)
        self.residualblock2 = SSSBlock(residual = True, omega = self.omega)
        self.residualblock3 = SSSBlock(residual = True, omega = self.omega)
        self.residualblock4 = SSSBlock(residual = True, omega = self.omega)
        self.residualblock5 = SSSBlock(residual = True, omega = self.omega)
        
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)
        # nn.init.uniform_(self.conv_last.weight, -1 * math.sqrt(6) / math.sqrt(32 * 5 ** 2), math.sqrt(6) / math.sqrt(32 * 5 ** 2))

    def forward(self, x):
        outputs = {}
        
        identity = x
        
        x = self.conv_first(x)
        if 'first' in self.tracking_point_list: outputs['first'] = x
        
        x = self.conv_block(x)
        if '0' in self.tracking_point_list: outputs['0'] = x
        
        x = self.residualblock1(x)
        if '1' in self.tracking_point_list: outputs['1'] = x
        
        x = self.residualblock2(x)
        if '2' in self.tracking_point_list: outputs['2'] = x
        
        x = self.residualblock3(x)
        if '3' in self.tracking_point_list: outputs['3'] = x
        
        x = self.residualblock4(x)
        if '4' in self.tracking_point_list: outputs['4'] = x
        
        x = self.residualblock5(x)
        if '5' in self.tracking_point_list: outputs['5'] = x
        
        x = self.conv_last(x)
        if 'last' in self.tracking_point_list: outputs['last'] = x
        
        x = x + identity
        if 'output' in self.tracking_point_list: outputs['output'] = x
        
        return x, outputs

class PlainRRSNetwork(nn.Module):

    def __init__(self, omega, tracking_point_list = ['']):
        super(PlainRRSNetwork, self).__init__()
        
        self.tracking_point_list = tracking_point_list
        self.omega = omega
        
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        # nn.init.uniform_(self.conv_first.weight, -1 * math.sqrt(6) / math.sqrt(5 ** 2), math.sqrt(6) / math.sqrt(5 ** 2))
        
        self.conv_block = SSSBlock(residual = False, omega = self.omega) 
        self.residualblock1 = SSSBlock(residual = False, omega = self.omega)
        self.residualblock2 = SSSBlock(residual = False, omega = self.omega)
        self.residualblock3 = SSSBlock(residual = False, omega = self.omega)
        self.residualblock4 = SSSBlock(residual = False, omega = self.omega)
        self.residualblock5 = SSSBlock(residual = False, omega = self.omega)
        
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)
        # nn.init.uniform_(self.conv_last.weight, -1 * math.sqrt(6) / math.sqrt(32 * 5 ** 2), math.sqrt(6) / math.sqrt(32 * 5 ** 2))

    def forward(self, x):
        outputs = {}
        
        identity = x
        
        x = self.conv_first(x)
        if 'first' in self.tracking_point_list: outputs['first'] = x
        
        x = self.conv_block(x)
        if '0' in self.tracking_point_list: outputs['0'] = x
        
        x = self.residualblock1(x)
        if '1' in self.tracking_point_list: outputs['1'] = x
        
        x = self.residualblock2(x)
        if '2' in self.tracking_point_list: outputs['2'] = x
        
        x = self.residualblock3(x)
        if '3' in self.tracking_point_list: outputs['3'] = x
        
        x = self.residualblock4(x)
        if '4' in self.tracking_point_list: outputs['4'] = x
        
        x = self.residualblock5(x)
        if '5' in self.tracking_point_list: outputs['5'] = x
        
        x = self.conv_last(x)
        if 'last' in self.tracking_point_list: outputs['last'] = x
        
        x = x + identity
        if 'output' in self.tracking_point_list: outputs['output'] = x
        
        return x, outputs

####################################################################

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
    def __init__(self, activation_string, omega_string, residual_string, tracking_point_list = ['']):
        super(CustomNetwork, self).__init__()
        
        self.tracking_point_list = tracking_point_list
        
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
        self.block5 = CustomBlock(activations=activation_list[5], residual=residual_list[5], omega=omega_list[5])
        
        self.conv_last = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)

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
        
        x = self.block5(x)
        if '5' in self.tracking_point_list: outputs['5'] = x

        x = self.conv_last(x)
        if 'last' in self.tracking_point_list: outputs['last'] = x

        x = x + identity
        if 'output' in self.tracking_point_list: outputs['output'] = x

        return x, outputs


    
if __name__ == '__main__':

    # create a random input tensor
    
    # Instantiate CustomNetwork
    activation_string = "R,R,R/CS,CS,S/R,S,S/R,R,R/S,S,S/R,S,S"
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
