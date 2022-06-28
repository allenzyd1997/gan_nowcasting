import torch.nn as nn
from CStack import conditioningStack
# from OldconvGRU import ConvGRU
from convGRU import ConvGRU
from outputStack import outputStack
from LCStack import LCStack
import torch
from GBlock import GBlockUp,GBlock
from torch.nn.utils import spectral_norm

def forward(self,CD_input,LCS_input,val=False):
        CD_input=torch.unsqueeze(CD_input,2)
       
        CD_output = self.conditioningStack(CD_input)
        
        self.convGRU1.setHidden(CD_output[3])
        self.convGRU2.setHidden(CD_output[2])
        self.convGRU3.setHidden(CD_output[1])
        self.convGRU4.setHidden(CD_output[0])

        if not val:
            for itr in range(6):
                
                LCS_output = self.LCStack(LCS_input[itr])

                output=[]
                for i in range(18):
                    x=self.conv1(self.convGRU1(LCS_output))
                    x=self.GBlock1(x)
                    x=self.GBlockUp1(x)

                    x=self.conv2(self.convGRU2(x))
                    x=self.GBlock2(x)
                    x=self.GBlockUp2(x)

                    x=self.conv3(self.convGRU3(x))
                    x=self.GBlock3(x)
                    x=self.GBlockUp3(x)

                    x=self.conv4(self.convGRU4(x))
                    x=self.GBlock4(x)
                    x=self.GBlockUp4(x)

                    final_result=self.outputStack(x)
                    output.append(final_result)
                gen_images=self.sigmoid(torch.cat(output,1))
                if itr ==0:
                    gen_sequences=gen_images
                else:
                    gen_sequences = gen_sequences +  gen_images
            return gen_sequences / 6.0
        else:
            LCS_output = self.LCStack(LCS_input)

            output=[]
            for i in range(18):
                x=self.conv1(self.convGRU1(LCS_output))
                x=self.GBlock1(x)
                x=self.GBlockUp1(x)

                x=self.conv2(self.convGRU2(x))
                x=self.GBlock2(x)
                x=self.GBlockUp2(x)

                x=self.conv3(self.convGRU3(x))
                x=self.GBlock3(x)
                x=self.GBlockUp3(x)

                x=self.conv4(self.convGRU4(x))
                x=self.GBlock4(x)
                x=self.GBlockUp4(x)

                final_result=self.outputStack(x)
                output.append(final_result)
            gen_images=self.sigmoid(torch.cat(output,1))
            return gen_images

if __name__ == "__main__":
        CD_input = torch.randn(8, 4, 256, 256)
        li = torch.randn((8, 8, 8))
        LCS_input = li.repeat(8,1,1,1)
        g = generator(24)
        RadarPreds=g(CD_input,LCS_input)