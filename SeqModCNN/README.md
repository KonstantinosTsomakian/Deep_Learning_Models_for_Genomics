## <ins> SeqModeCNN

**<ins>SeqModeCNN** is a deep learning network that can be used for both classification and regression tasks. It's architecture implements a dual module **CNN** block followed by a feed forward layer (**FFN**) where finally a single logit is outputed. The output logit can be used for regression tasks as well as to perform sample classification. The model was structured to work with genomic data where **1D** signal tracks are used as input to the model along with the corresponding sequence information. Each signal track can be used as an additional channel to the model input tensor. 

The dual module structure of the model allows the simultanous processing  of multiple genomic information that can be represented as signal tracks and sequence information in the form of one hot encoded tesnors. The convolution parameters that operate on the genomic signals and the sequence information are user defined. The convoluted inputs are then concatenated to a single tensor that holds information about both the signal and the sequence. The unified tensor is finally passed to an FFN head that generates the final model output.


<figure style="text-align: center;">
  <img src="./images/Screenshot 2026-06-21 100713.png" alt="A sunset" width="400">
</figure>


The model comes along with the trainer where the type of the task is defined. 

<ins> The parameters of the model are:

```-channels```   The number of the channels of the input tensor.

```-return_embedding``` Boolean. Whether to return last layer embeddings or not.

```-input_len``` The length of the histone modification signal.

```-fine_len``` If ```with_fine = True``` define the the length of the input region from the center of the input that will be scanned with the fine filter. This is implemented this way because the model was initially implemented for genomic input signals where the gene TSS was in the middle.

```-hist_param_list``` The size, the stride and the padding that will be used for the default convolution operation.

```-seq_param_list``` The size, the stride and the padding that will be used for the convolution operation for the sequence input.

```-seq_len``` The length of the input sequences.

```-seq_channels``` The number of channels of the tensor that stores the one hot encoded sequences..
