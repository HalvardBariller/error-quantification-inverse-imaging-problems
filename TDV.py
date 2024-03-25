import torch 
import torch.nn as nn
import torch.nn.functional as F

"""This code is based on the implementation of the Total Deep Variation (TDV) Regularizer
    proposed in the paper "Total Deep Variation for Linear Inverse Problems" by 
    E. Kobler, A. Effland, K. Kunisch, and T. Pock. The code is adapted from the
    implementation proposed by M. Malka in his course (Ben Gurion University, 2022).
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SmoothLogStudentTActivation(torch.nn.Module):
    '''Log Student T Activation'''
    def __init__(self, nu=9):
        super(SmoothLogStudentTActivation, self).__init__()
        self.nu = nu

    def forward(self,x):
        eps = 1e-6
        return torch.log(1 + self.nu * x**2 + eps) / (2 * self.nu)

    
class MicroBlock(nn.Module):
    '''MicroBlock with residual connection.'''
    def __init__(self, num_features, first=False, last=False):
        super(MicroBlock, self).__init__()
        in_channels = 3 if first else num_features
        out_channels = 3 if last else num_features
        self.last = last
        self.act = SmoothLogStudentTActivation()
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=False) if last else None

    def forward(self, x):
        x = self.conv1(x)
        phi_x = self.act(x)
        x = x + self.conv2(phi_x)
        if self.last:
            return self.conv3(x)
        return x
    

class MacroBlock(nn.Module):
    '''MacroBlock composed of 5 Microblocks with residual connections.'''
    def __init__(self, num_features, first_mab=True, last_mab=False):
        super().__init__()
        self.num_features = num_features
        self.conv_2m_m = nn.Conv2d(num_features*2,num_features,kernel_size=(1,1),bias=False)
        self.downsampling = nn.Conv2d(num_features,num_features,kernel_size=(2,2),stride=2,bias=False)
        self.upsampling = nn.ConvTranspose2d(num_features, num_features, kernel_size=(2, 2), stride=2,bias=False)

    def block(self, num_features, first_mab, last_mab):
        mib1 = MicroBlock(num_features,first=first_mab)
        mib2 = MicroBlock(num_features)
        mib3 = MicroBlock(num_features)
        mib4 = MicroBlock(num_features)
        mib5 = MicroBlock(num_features,last=last_mab)
        return nn.Sequential(mib1,mib2,mib3,mib4,mib5)

    def forward_pass(self, x, first_mab, last_mab, block):
        if first_mab:
            x1 = x
        else:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
        mib_out1 = block[0](x1)
        out1 = self.downsampling(mib_out1)
        if not first_mab:
            out1 = x2 + out1
        mib_out2 = block[1](out1)
        out2 = self.downsampling(mib_out2)
        if not first_mab:
            out2 = x3 + out2
        mib_out3 = block[2](out2)
        mib_out3_us = self.upsampling(mib_out3)
        concat_out2_out3 = self.conv_2m_m(torch.cat([mib_out2, mib_out3_us], 1))
        mib_out4 = block[3](concat_out2_out3)
        mib_out4_us = self.upsampling(mib_out4)
        concat_out1_out2 = self.conv_2m_m(torch.cat([mib_out1, mib_out4_us], 1))
        mib_out5 = block[4](concat_out1_out2)
        if last_mab:
            return mib_out5
        else:
            return [mib_out5, mib_out4, mib_out3]
        

class TDV(MacroBlock):
    ''' Construction of the TDV Regularizer, robust to any number of scales '''
    def __init__(self, num_features, num_scales=3):
        super().__init__(num_features)
        self.num_features = num_features
        self.num_scales = num_scales
        self.tdv = self.build_network()

    def build_network(self):
        self.ma_blocks = []
        for scale in range(self.num_scales):
            if scale == 0:
                self.ma_blocks.append(self.block(self.num_features, first_mab=True, last_mab=False))
            elif scale == self.num_scales - 1:
                self.ma_blocks.append(self.block(self.num_features, first_mab=False, last_mab=True))
            else:
                self.ma_blocks.append(self.block(self.num_features, first_mab=False, last_mab=False))

        return nn.ModuleList(self.ma_blocks)

    def forward(self,x):
        for scale in range(self.num_scales):
            if scale == 0:
                x = self.forward_pass(x, True, False, self.tdv[scale])
            elif scale == self.num_scales - 1:
                x = self.forward_pass(x, False, True, self.tdv[scale])
            else:
                x = self.forward_pass(x, False, False, self.tdv[scale])
        return x
    

def load_TDV_network(path):
    """Load the TDV network from a given path."""
    num_features = 32
    tdv_scales = 3
    model = TDV(num_features,tdv_scales).to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().cpu()
    return model

