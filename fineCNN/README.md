## <ins> fineCNN

**<ins>fineCNN** is a deep learning network that can be used for both classification and regression tasks. It's architecture implements a dual module **CNN** block followed by a feed forword layer (**FFN**) where finally a single logit is outputed. The output logit can be used for regression tasks as well as to perform sample classification. The model was structured to work with genomic data where **1D** signal tracks are used as input to the model. Each signal track can be used as an additional channel to the model input tensor. 

Genomic signal can appear as long distance peaks that spann long genomic regions or as more narrow peaks that spann shorter regions but follow a very specific distribution even at the nucleosome level. Taking this into acount fineCNN was implented to process a single input with two different convolution operations with user defined parameters. The input once is processed through a convolution block with a big kernel size to capture the wide distributed signals and once with a smaller kernel size to capture finer signal distribution patterns.


The residual block is optional. If the parameter ```res_blocks``` that controls it is set to ```False``` instead of a residual block a simple CNN is used. The GRU option ```use_gru``` is optional as well. In the case where the option is set to ```False``` the tensor that comes out of the convolution block is directed to the final feed forward layer.


The model comes along with the trainer where the type of the task is defined. 

<ins> The parameters of the model are:

```-input_channels```   The number of the channels of the input tensor.

```-batch_size``` The batch size that is used for training the model.

```-histon_len``` The length of the histone modification signal.

```-with_fine``` Boolean. Weather to use the fine filter module or not.

```-fine_len``` If ```with_fine = True``` define the the length of the input region from the center of the input that will be scanned with the fine filter. This is implemented this way because the model was initially implemented for genomic input signals where the gene TSS was in the middle.

```-hist_param_list``` The size, the stride and the padding that will be used for the default convolution operation.

```-fine_hist_param_list``` The size, the stride and the padding that will be used for the convolution operation with the fine kernel.
