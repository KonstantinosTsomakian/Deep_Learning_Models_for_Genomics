import torch.nn as nn
import torch.nn.functional as F
import torch


#CNN model with fine module and gru in architecture but no sequence moule
class histonCNN_with_fine_no_sequence_module(nn.Module):
    def __init__(self, input_channels,
                  batch_size, histon_len,
                  with_fine = True,
                  fine_len = 20,
                    hist_param_list = [10,2,2],
                    fine_hist_param_list = [3, 1, 1]):
        super(histonCNN_with_fine_no_sequence_module, self).__init__()
        # For convolution layers the shapes for histone modifications are [batch size, channels, width]
        #batch size represents the number of genes processed at a time
        # channels represent different histone modifications
        # width reprezents the bins of the genome around the tss

        self.histon_len = histon_len
        self.hist_param_list = hist_param_list
        self.fine_hist_param_list = fine_hist_param_list
        self.fine_len = fine_len
        self.with_fine = with_fine


        kh, sh, ph = self.hist_param_list
        kfh, sfh, pfh = self.fine_hist_param_list

        hist_out_channels = 32


        def compute_tensor_dim_after_convolution(width, kernel_size, stride, padding):
            import math
            W_out = math.floor((width + 2 * padding - kernel_size) / stride + 1)
            return (W_out)
        
        def x_after_conv(len, param_list, pool_param_list):
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(len, *param_list), *pool_param_list)
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(xl_ac,param_list[0] // 2, param_list[1], param_list[2]), *pool_param_list)
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(xl_ac, param_list[0] // 2, param_list[1], param_list[2]), *pool_param_list)
            return(xl_ac)
        
        def x_fine_after_conv(len, param_list, pool_param_list):
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(len, *param_list), *pool_param_list)
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(xl_ac,param_list[0], param_list[1], param_list[2]), *pool_param_list)
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(xl_ac, param_list[0], param_list[1], param_list[2]), *pool_param_list)
            return(xl_ac)
        


        xl_ac = x_after_conv(len = self.histon_len, param_list=self.hist_param_list, pool_param_list=[4,2,0])
        fine_xl_ac = x_fine_after_conv(len = self.fine_len * 2, param_list=self.fine_hist_param_list, pool_param_list=[2,2,0])
        
        # print(xl_ac, fine_xl_ac)
        # print(int(1 * xl_ac * 32) , int(1 * fine_xl_ac * 32))

        
        self.dropout = nn.Dropout(p=0.1)

        self.batch_size = batch_size
        self.input_channels = input_channels
        

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=input_channels * 2, kernel_size=kh, stride=sh, padding=ph) # In [bs, 5, 5000]
        self.conv2 = nn.Conv1d(in_channels=input_channels * 2, out_channels=input_channels * 4, kernel_size=kh // 2, stride=sh, padding=ph)
        self.conv3 = nn.Conv1d(in_channels=input_channels * 4, out_channels=hist_out_channels, kernel_size=kh // 2, stride=sh, padding=ph) # Out [bs, 40, 5000]
        

        self.fine_conv1 = nn.Conv1d(in_channels=input_channels, out_channels=input_channels * 2, kernel_size = kfh, stride=sfh, padding=pfh) # In [bs, 5, 5000]
        self.fine_conv2 = nn.Conv1d(in_channels=input_channels * 2, out_channels=input_channels * 4, kernel_size=kfh, stride=sfh, padding=pfh)
        self.fine_conv3 = nn.Conv1d(in_channels=input_channels * 4, out_channels=hist_out_channels, kernel_size=kfh, stride=sfh, padding=pfh) # Out [bs, 40, 5000]
      

        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)
        self.fine_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)


        ### Batch normalization layers
        # For convolution layers for histone modifications
        self.convbn1 = nn.BatchNorm1d(input_channels * 2)
        self.convbn2 = nn.BatchNorm1d(input_channels * 4)
        self.convbn3 = nn.BatchNorm1d(hist_out_channels)




        # For linear layers
        self.fcbn1 = nn.BatchNorm1d(((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4) 
        self.fcbn2 = nn.BatchNorm1d(((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4)
        self.fcbn3 = nn.BatchNorm1d(((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4)
        self.fcbn4 = nn.BatchNorm1d(((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 2)



        # Fully connected layers
        self.fc1_x_x_fine = nn.Linear(in_features= int(1 * xl_ac * hist_out_channels) + int(1 * fine_xl_ac * hist_out_channels), out_features=((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4)
        self.fc1_x_only = nn.Linear(in_features= int(1 * xl_ac * 1 * hist_out_channels), out_features=((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4) 

 
        self.fc2 = nn.Linear(in_features=((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4, out_features= ((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4)
        self.fc3 = nn.Linear(((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4, out_features= ((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4)
        self.fc4 = nn.Linear(in_features=((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 4, out_features=((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 2)

        self.fc5 = nn.Linear(in_features=((1 * xl_ac * 1 * 2) + (1 * fine_xl_ac * 1 * 2)) * 2, out_features=1)


    
    def forward(self, x):

        center = x.shape[2] // 2
        # print(center)
        x_fine = x[:, :, (center - self.fine_len):(center + self.fine_len)]
        # Input shape: (batch_size, 5 channels(histone modifications), heigh = 1, width = 20(features))
        
        center = x.shape[2] // 2
        # print(center)
        # x_fine = x[:, :, (center - self.fine_len):(center + self.fine_len)]
        # Input shape: (batch_size, 5 channels(histone modifications), heigh = 1, width = 20(features))
        
        # Convolutional layers with ReLU and pooling for the histone modification module
        def wide_block(x):
            x = self.dropout(self.pool( F.relu( self.convbn1( self.conv1(x) ) ) ) )  
            x = self.dropout(self.pool( F.relu( self.convbn2( self.conv2(x) ) ) ) )
            x = self.pool( F.relu( self.convbn3(self.conv3(x)) ) )  
            return(x)
        
        
        # print(f'After covolution x has shape {x.shape}')
        def fine_block(x_fine):
            # Convolutional layers with ReLU and pooling for the histone modification module using the fine filter
            x_fine = self.fine_pool( F.relu( self.convbn1( self.fine_conv1(x_fine) ) ) )
            x_fine = self.fine_pool( F.relu( self.convbn2( self.fine_conv2(x_fine) ) ) )
            x_fine = self.fine_pool(F.relu( self.convbn3(self.fine_conv3(x_fine)) ))
            return(x_fine)
              

        
        
        x_fine = fine_block(x_fine)
        x = wide_block(x)
        # print(x.shape, x_fine.shape, 'step1')
        
    




        # Flatten the output for the fully connected layer
        bs = x.shape[0]
        x = x.reshape(bs, -1)  # Flatten
        x_fine = x_fine.reshape(bs, -1)
        # print(x.shape, x_fine.shape, 'step 2')


            
        if self.with_fine:
            
            x = torch.cat((x, x_fine), dim = 1)
            # print(x.shape, 'step 3')
            x_embed = x
            x = F.relu( self.fcbn1(self.fc1_x_x_fine(x)) )
            # print(x.shape, 'step 3')
        elif not self.with_fine:
            x_embed = x
            x = F.relu( self.fcbn1(self.fc1_x_only(x)) )
       

            



        
        
             
        # Fully connected layers with ReLU
        # print(x.shape)
        x = self.dropout(F.relu( self.fcbn2(self.fc2(x)) )  )# Output shape: (batch_size, 10)
        x = self.dropout(F.relu( self.fcbn3(self.fc3(x)) ) )
        x = self.dropout(F.relu( self.fcbn4(self.fc4(x)) ) )
        x = self.fc5(x)
        
        return( x, x_embed)
