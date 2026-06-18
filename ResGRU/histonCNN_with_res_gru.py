import torch
import torch.nn as nn
import torch.nn.functional as F

# The following model takes as input a multichannel tensor of histone modifications and outputs a prediction tensor of width [batch, target_dim]
# The input is of shape [batch,channels,width] and the outputs are of shape [batch,target_dim]. This model can be paired with the histonCNN_signal_regression_trainer trainer.

class histonCNN_with_res_gru(nn.Module):
    def __init__(self, input_channels,
                  batch_size, histon_len,
                  target_dim,
                    hist_param_list = [10,2,2],
                    averaging_window = False,
                    res_blocks = False,
                    use_gru = False,
                    return_embeddings = False):
        super(histonCNN_with_res_gru, self).__init__()
        
        # For convolution layers the shapes for histone modifications are [batch size, channels, width]
        #batch size reprezents the number of genes processed at a time
        # channels represent different histone modifications
        # width reprezents the bins of the genome around the tss

     
        self.histon_len = histon_len
        self.hist_param_list = hist_param_list
        self.res_blocks = res_blocks
        self.use_gru = use_gru
        self.batch_size = batch_size
        self.averaging_window = averaging_window
        self.input_channels = input_channels
        self.target_dim = target_dim
        self.return_embeddings = return_embeddings


        kh, sh, ph = self.hist_param_list

        hist_out_channels = 32

        # This fucntion is to make the model more flexible so the user can use different inout sizes and different convolution parameters
        def compute_tensor_dim_after_convolution(width, kernel_size, stride, padding):
            import math
            W_out = math.floor((width + 2 * padding - kernel_size) / stride + 1)
            return (W_out)
        
        # This function takes as input the length of the input genomix region (in bins/length of the vector) and the list of the kernel parameters.
        # Returns the length of the vector after the convolution operation.
        def x_after_conv(len, param_list, pool_param_list):
            # sl_ac stands for sequence length after the convolution layers and flattening
            #sequence length after convolution layers and flattening
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(len, *param_list), *pool_param_list)
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(xl_ac,param_list[0] // 2, param_list[1], param_list[2]), *pool_param_list)
            xl_ac = compute_tensor_dim_after_convolution(compute_tensor_dim_after_convolution(xl_ac, param_list[0] // 2, param_list[1], param_list[2]), *pool_param_list)
            return(xl_ac)

        #Compute the the size of the tensor after average pooling. Average pooling is an extra option set to the model
        # to perform an average pooloing after all the convolution operations
        tensor_size_after_average_pooling = compute_tensor_dim_after_convolution(width = self.target_dim,
                                                                                  kernel_size = averaging_window,
                                                                                    stride = 1,
                                                                                      padding = 0)
        # Cmpute the lenth of the tensor after all the convolution operations
        # This is computed here so the parameters of the downstream gru can be defined/
        xl_ac = x_after_conv(len = self.histon_len, param_list=self.hist_param_list, pool_param_list=[4,2,0])

        # Wether to use dropout or not.
        self.dropout = nn.Dropout(p=0.0)
        



        # Convolutional layers for the histone modifications.
        # Here the architecture of the model is defined. 
        # The architecture is constant and cannot be changed.
        # However the parameters of the layers can be user defined.

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=input_channels * 2, kernel_size=kh, stride=sh, padding=ph) # In [bs, 5, 5000]
        self.conv2 = nn.Conv1d(in_channels=input_channels * 2, out_channels=input_channels * 4, kernel_size=kh // 2, stride=sh, padding=ph)
        self.conv3 = nn.Conv1d(in_channels=input_channels * 4, out_channels=hist_out_channels, kernel_size=kh // 2, stride=sh, padding=ph) # Out [bs, 40, 5000]
        
        # For the residual connections of the wide block
        # The residual connections are yet to be fully operable since the kernel sizes should be fixed.
        # When running try run kernel sizes of 3, 5, 7
        self.res_conv1 = nn.Conv1d(in_channels=input_channels, out_channels=input_channels * 2, kernel_size=kh, stride=sh, padding="same") # In [bs, 5, 5000]
        self.res_conv2 = nn.Conv1d(in_channels=input_channels * 2, out_channels=input_channels * 4, kernel_size=kh, stride=sh, padding="same")
        self.res_conv3 = nn.Conv1d(in_channels=input_channels * 4, out_channels=input_channels, kernel_size=kh, stride=sh, padding="same") # Out [bs, 40, 5000]
        
        
        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)    



        ### Batch normalization layers
        ### Batch normalization is used to solve the internal covariate shift problem :
        # https://medium.com/analytics-vidhya/internal-covariate-shift-an-overview-of-how-to-speed-up-neural-network-training-3e2a3dcdd5cc

        # For convolution layers for histone modifications
        self.convbn1 = nn.BatchNorm1d(input_channels * 2)
        self.convbn2 = nn.BatchNorm1d(input_channels * 4)
        self.convbn3 = nn.BatchNorm1d(hist_out_channels)

        # For linear layers
        self.fcbn1 = nn.BatchNorm1d((1 * xl_ac * 1 * 2)  * 4) 
        self.fcbn2 = nn.BatchNorm1d((1 * xl_ac * 1 * 2)  * 32)
        self.fcbn3 = nn.BatchNorm1d((1 * xl_ac * 1 * 2)  * 64)
        self.fcbn4 = nn.BatchNorm1d((1 * xl_ac * 1 * 2)  * 128)

        # Fully connected layers
        self.fc1_cnn_no_gru_no_res = nn.Linear(in_features= int(xl_ac * hist_out_channels), out_features=(1 * xl_ac * 1 * 2)  * 4)
        self.fc1_cnn_no_gru_res = nn.Linear(in_features= int(self.histon_len * self.input_channels), out_features=(1 * xl_ac * 1 * 2)  * 4)
        self.fc1_cnn_gru_no_res = nn.Linear(in_features= int(xl_ac  * 2), out_features=(1 * xl_ac * 1 * 2)  * 4)
        self.fc1_cnn_gru_res = nn.Linear(in_features= int(self.histon_len * 2), out_features=(1 * xl_ac * 1 * 2)  * 4)



        self.fc2 = nn.Linear(in_features=(1 * xl_ac * 1 * 2)  * 4, out_features= (1 * xl_ac * 1 * 2)  * 32)
        self.fc3 = nn.Linear((1 * xl_ac * 1 * 2)  * 32, out_features= (1 * xl_ac * 1 * 2)  * 64)
        self.fc4 = nn.Linear(in_features=(1 * xl_ac * 1 * 2)  * 64, out_features=(1 * xl_ac * 1 * 2)  * 128)

        self.fc5 = nn.Linear(in_features=(1 * xl_ac * 1 * 2)  * 128, out_features=self.target_dim)


        # If the GRU option is selected the following GRU is implemented on top of the output of the convolution operations
        # Two GRU implementations are defined. One if residual layers are used and one if not.
        # This is done because the dimensions of the output change.

        # GRU layer
        if self.res_blocks:
            self.gru = nn.GRU(input_size = self.input_channels, # This is the number of features of each element of the sequence. Here each channel outputed from the final convolution layer is considered as a feature. Because the output cahnnels are 40 from the convolution layer here I have input_channelsthat are 5 * 8 to make it 40
                            hidden_size = 1, # This is the number of units in the rnn layer
                            num_layers = 2, # This is the number of stacked layers that the rnn layer has. #Note that this is not the same with the hidden layers of the network.
                                batch_first=True,
                                bidirectional = True) # So the output will be of size input_size * 8 * 16 * 2. input_size * 8 referes to the number of the features the vector initally had. 16 to the number of the nodes of the layer, and the 2 because I set it to be bidirectional.
            #  GRU layer
            self.lstm = nn.LSTM(input_size = self.input_channels, # This is the number of features of each element of the sequence. Here each channel outputed from the final convolution layer is considered as afeature
                            hidden_size = 16, # This is the number of units in the rnn layer
                            num_layers = 2, # This is the number of layers that the rnn layer has. #Note that this is not the same with the hidden layers of the network
                                batch_first=True,
                                bidirectional = True) 
        elif not self.res_blocks:
            self.gru = nn.GRU(input_size =hist_out_channels, # This is the number of features of each element of the sequence. Here each channel outputed from the final convolution layer is considered as a feature. Because the output cahnnels are 40 from the convolution layer here I have input_channelsthat are 5 * 8 to make it 40
                            hidden_size = 1, # This is the number of units in the rnn layer
                            num_layers = 2, # This is the number of stacked layers that the rnn layer has. #Note that this is not the same with the hidden layers of the network.
                                batch_first=True,
                                bidirectional = True) # So the output will be of size input_size * 8 * 16 * 2. input_size * 8 referes to the number of the features the vector initally had. 16 to the number of the nodes of the layer, and the 2 because I set it to be bidirectional.
            #  GRU layer
            self.lstm = nn.LSTM(input_size = input_channels, # This is the number of features of each element of the sequence. Here each channel outputed from the final convolution layer is considered as afeature
                            hidden_size = 16, # This is the number of units in the rnn layer
                            num_layers = 2, # This is the number of layers that the rnn layer has. #Note that this is not the same with the hidden layers of the network
                                batch_first=True,
                                bidirectional = True) 

    
    def forward(self, x):
        
  
        # Convolutional layers with ReLU and pooling for the histone modification module
        def wide_block(x):
            x = self.dropout(self.pool( F.relu( self.convbn1( self.conv1(x) ) ) ) )  
            x = self.dropout(self.pool( F.relu( self.convbn2( self.conv2(x) ) ) ) )
            x = self.pool( F.relu( self.convbn3(self.conv3(x)) ) )  
            return(x)
        

        def residual_wide_block(x):
            res_x = x
            x = self.dropout(F.relu( self.convbn1( self.res_conv1(x) ) ) )  
            x = self.dropout(F.relu( self.convbn2( self.res_conv2(x) ) ) )
            x = F.relu(self.res_conv3(x))
            x = F.relu(x + res_x)
            return(x)
        
        def average_smoothing(tensor, window_size):

                x = tensor.unsqueeze(1)

                # Compute padding to preserve input length
                total_pad = window_size - 1
                pad_left = total_pad // 2
                pad_right = total_pad - pad_left
                x_padded = F.pad(x, (pad_left, pad_right), mode='replicate')

                # Apply average pooling
                out = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)

                return out.squeeze(1)  # Return shape [B, L]


        # From this point and on the forward pass of the input tensor is described.
        if self.res_blocks:
            x = residual_wide_block(x)
        else:
            x = wide_block(x)


        #Permute the cnn output to fit the LSTM layers. 
        ### SOS ### The CNN output is in the form of [batch_size, channels, width] while the LSTM input should be in the from of [batch_size, sequence_length, input_size].
        if self.use_gru:
            x = x.permute(0, 2, 1) #This swaps the channels and the width
            # Pass through GRU layer
            x, x_hidden = self.gru(x)
        


        

        # Flatten the output for the fully connected layer.
        bs = x.shape[0]
        x = x.reshape(bs, -1)  # Flatten


        if not self.res_blocks and not self.use_gru:

            x_embed = x
            x = F.relu( self.fcbn1(self.fc1_cnn_no_gru_no_res(x)) )


        elif not self.res_blocks and self.use_gru:
            x_embed = x
            x = F.relu( self.fcbn1(self.fc1_cnn_gru_no_res(x)) )        

        elif self.res_blocks and not self.use_gru:
            x_embed = x
            x = F.relu( self.fc1_cnn_no_gru_res(x))


        elif self.res_blocks and self.use_gru:
            x_embed = x
            x = F.relu( self.fc1_cnn_gru_res(x))

            


        # Fully connected layers with ReLU
        # print(x.shape)
        x = self.dropout(F.relu( self.fcbn2(self.fc2(x)) )  )
        
        x = self.dropout(F.relu( self.fcbn3(self.fc3(x)) ) )
        x = self.dropout(F.relu( self.fcbn4(self.fc4(x)) ) )
        x = self.fc5(x)
        
        if self.averaging_window:
            x = average_smoothing(x, self.averaging_window)
        
       
        #x_embed is the last layer representation before the output.
        #Note the output logit can be used for both regression and classification tasks
        # depending on the trainer used
        if self.return_embeddings:
            return( x, x_embed)
        else:
            return(x)
