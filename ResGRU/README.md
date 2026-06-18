## <ins> ResConvGRU

**<ins>ResConvGRU** is a deep learning network that can be used for both classification and regression tasks. It's architecture implements a residual CNN block followed by a gated neural network (**GRU**). The hidden state of the GRU is finally passed to a feed forword layer (**FFN**) where the output of defined shape is computed. The output can be a tensor or a scalar for regression tasks or a scalar that can be used to perform sample classification. The residual block is optional. If the parameter ```res_blocks``` that controls it is set to ```False``` instead of a residual block a simple CNN is used. The GRU option ```use_gru``` is optional as well. In the case where the option is set to ```False``` the tensor that comes out of the convolution block is directed to the final feed forward layer.

<figure>
  <img src="./images/Screenshot 2026-06-18 212126.png" alt="A sunset" width="400">
</figure>


The model comes along with the trainer where the type of the task is defined. 
