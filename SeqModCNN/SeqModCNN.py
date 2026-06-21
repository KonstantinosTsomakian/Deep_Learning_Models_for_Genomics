import torch
import torch.nn as nn
import torch.nn.functional as F

#### A simple cnn model with a module to add one hot encoded sequence
class seqCNN(nn.Module):
    def __init__(self,input_len, channels, hist_param_list, seq_param_list, seq_len, seq_channels, return_embedding = False):
        super(seqCNN, self).__init__()
        self.input_len = input_len
        self.channels = channels
        self.return_embedding = return_embedding
        self.hist_param_list = hist_param_list
        self.seq_param_list = seq_param_list
        self.seq_len = seq_len
        self.seq_channels = seq_channels


        kh, sh, ph = self.hist_param_list
        ks, ss, ps = self.seq_param_list


        def compute_tensor_dim_after_convolution(width, kernel_size, stride, padding):
            import math
            W_out = math.floor((width + 2 * padding - kernel_size) / stride + 1)
            return (W_out)

        def x_after_conv(len, param_list, pool_param_list):
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(len, *param_list), *pool_param_list)
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(xl_ac,param_list[0] // 2, param_list[1], param_list[2]), *pool_param_list)
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(xl_ac, param_list[0] // 2, param_list[1], param_list[2]), *pool_param_list)
            return(xl_ac)
        
        xl_ac = x_after_conv(len = self.input_len, param_list=self.hist_param_list, pool_param_list=[4,2,0])
        sl_ac = x_after_conv(len = self.seq_len, param_list=self.seq_param_list, pool_param_list=[4,2,0])
        print(f'After the convolutiopn block x has length {xl_ac}')
        print(f'After the convolutiopn block sequence has length {sl_ac}')

# Define the layers for the histone modification data
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels * 2, kernel_size=kh, stride=sh, padding=ph) # In [bs, 5, 5000]
        self.conv2 = nn.Conv1d(in_channels=channels * 2, out_channels=channels * 4, kernel_size=kh // 2, stride=sh, padding=ph)
        self.conv3 = nn.Conv1d(in_channels=channels * 4, out_channels=channels * 8, kernel_size=kh // 2, stride=sh, padding=ph) # Out [bs, 40, 5000]

        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)#
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)#


        ### Batch normalization layers
        # For convolution layers for histone modifications
        self.convbn1 = nn.BatchNorm1d(self.channels * 2)
        self.convbn2 = nn.BatchNorm1d(self.channels * 4)
        self.convbn3 = nn.BatchNorm1d(self.channels * 8)   

        
# Define the layers for the sequence data
        self.seq_conv1 = nn.Conv1d(in_channels=seq_channels, out_channels=seq_channels * 2, kernel_size=ks, stride=ss, padding=ps) # In [bs, 5, 5000]
        self.seq_conv2 = nn.Conv1d(in_channels=seq_channels * 2, out_channels=seq_channels * 4, kernel_size=ks // 2, stride=ss, padding=ps)
        self.seq_conv3 = nn.Conv1d(in_channels=seq_channels * 4, out_channels=seq_channels * 8, kernel_size=ks // 2, stride=ss, padding=ps) # Out [bs, 40, 5000]

        # Max pooling layer
        self.seq_pool = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)
        self.seq_pool2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)#
        self.seq_pool3 = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)#


        ### Batch normalization layers
        # For convolution layers for histone modifications
        self.seq_convbn1 = nn.BatchNorm1d(self.seq_channels * 2)
        self.seq_convbn2 = nn.BatchNorm1d(self.seq_channels * 4)
        self.seq_convbn3 = nn.BatchNorm1d(self.seq_channels * 8)

        cat_dim = (xl_ac * channels * 8) + (sl_ac * seq_channels * 8)
        print(f'Dim after concatenation {cat_dim}')

        self.fc1 = nn.Linear(cat_dim, cat_dim // 2)
        self.fc2 = nn.Linear(cat_dim // 2, cat_dim // 4)
        self.fc3 = nn.Linear(cat_dim // 4, 1)

        self.fcbn1 = nn.BatchNorm1d( cat_dim // 2 )
        self.fcbn2 = nn.BatchNorm1d( cat_dim // 4 )
        
                
    def forward(self, x, seq):

        def wide_block(x):
            x = self.pool( F.relu( self.convbn1( self.conv1(x) ) ) )  
            x = self.pool2( F.relu( self.convbn2( self.conv2(x) ) ) )
            x = self.pool3( F.relu( self.convbn3(self.conv3(x)) ) )  
            return(x)
        
        def wide_block_seq(seq):
            seq = self.seq_pool( F.relu( self.seq_convbn1( self.seq_conv1(seq) ) ) )  
            seq = self.seq_pool2( F.relu( self.seq_convbn2( self.seq_conv2(seq) ) ) )
            seq = self.seq_pool3( F.relu( self.seq_convbn3(self.seq_conv3(seq)) ) )  
            return(seq)
        x = wide_block(x)
        seq = wide_block_seq(seq)
        
        x = x.view(x.size(0), -1)
        seq = seq.view(seq.size(0), -1)
        
        x_seq = torch.cat([x, seq], dim = 1)

        x_seq = F.relu( self.fcbn1( self.fc1( x_seq ) ) )
        
        x_seq = F.relu( self.fcbn2( self.fc2( x_seq ) ) )
        x_embed = x_seq.clone()
        x_seq = self.fc3( x_seq )
        if self.return_embedding:

            return(x_seq, x_embed)
        else:
            return(x_seq)