import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.utils import clone_module_list


class ConvMixerLayer(Module):
    """
    <a id="ConvMixerLayer"></a>
    ## ConvMixer layer
    This is a single ConvMixer layer. The model will have a series of these.
    """

    def __init__(self, d_model: int, kernel_size: int):
        """
        * `d_model` is the number of channels in patch embeddings, $h$
        * `kernel_size` is the size of the kernel of spatial convolution, $k$
        """
        super().__init__()
        # Depth-wise convolution is separate convolution for each channel.
        # We do this with a convolution layer with the number of groups equal to the number of channels.
        # So that each channel is it's own group.
        self.depth_wise_conv = nn.Conv2d(d_model, d_model,
                                         kernel_size=kernel_size,
                                         groups=d_model,
                                         padding=(kernel_size - 1) // 2)
        # Activation after depth-wise convolution
        self.act1 = nn.GELU()
        # Normalization after depth-wise convolution
        self.norm1 = nn.BatchNorm2d(d_model)

        # Point-wise convolution is a $1 \times 1$ convolution.
        # i.e. a linear transformation of patch embeddings
        self.point_wise_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        # Activation after point-wise convolution
        self.act2 = nn.GELU()
        # Normalization after point-wise convolution
        self.norm2 = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor):
        # For the residual connection around the depth-wise convolution
        residual = x

        # Depth-wise convolution, activation and normalization
        x = self.depth_wise_conv(x)
        x = self.act1(x)
        x = self.norm1(x)

        # Add residual connection
        x += residual

        # Point-wise convolution, activation and normalization
        x = self.point_wise_conv(x)
        x = self.act2(x)
        x = self.norm2(x)
        #
        return x


class PatchEmbeddings(Module):
    """
    <a id="PatchEmbeddings"></a>
    ## Get patch embeddings
    This splits the image into patches of size $p \times p$ and gives an embedding for each patch.
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        """
        * `d_model` is the number of channels in patch embeddings $h$
        * `patch_size` is the size of the patch, $p$
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch.
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        # Activation function
        self.act = nn.GELU()
        # Batch normalization
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Apply convolution layer
        x = self.conv(x)
        # Activation and normalization
        x = self.act(x)
        x = self.norm(x)

        #
        return x


class ClassificationHead(Module):
    """
    <a id="ClassificationHead"></a>
    ## Classification Head
    They do average pooling (taking the mean of all patch embeddings) and a final linear transformation
    to predict the log-probabilities of the image classes.
    """

    def __init__(self, d_model: int):
        """
        * `d_model` is the number of channels in patch embeddings, $h$
        * `n_classes` is the number of classes in the classification task
        """
        super().__init__()
        # Average Pool
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.convtrans = nn.ConvTranspose2d(d_model, d_model, kernel_size=10, stride=8, padding=1)
        self.act = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(d_model)
        #这里尽量不要考虑上采用函数，因为这个线性插值的纯粹的数值计算是不能学习的，反卷积可以做到上采样
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        #1*1卷积这里，无论前面输出多好channel， 这里直接拿来作为输入就行了
        self.adjust = nn.Conv2d(d_model, 3, kernel_size=1, stride=1, padding=0)
        #由于目前用的交叉商函数自带softmax， 所以这里就不需要加入softmax了
        # Linear layer
        # self.linear = nn.Linear(d_model, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # Average pooling
        # x = self.pool(x)
        x = self.convtrans(x)
        x = self.act(x)
        x = self.batchnorm(x)
        # x = self.upsample(x)
        # Get the embedding, `x` will have shape `[batch_size, d_model, 1, 1]`
        # x = x[:, :, 0, 0]
        # Linear layer
        # x = self.linear(x)

        # print(x)
        # print('*'*25)
        x = self.adjust(x)
        # x = self.softmax(x)
        # print(x.shape)
        # print(x)

        #
        return x


class ConvMixer(Module):
    """
    ## ConvMixer
    This combines the patch embeddings block, a number of ConvMixer layers and a classification head.
    """
    def __init__(self, conv_mixer_layer: ConvMixerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings,
                 classification: ClassificationHead):
        """
        * `conv_mixer_layer` is a copy of a single [ConvMixer layer](#ConvMixerLayer).
         We make copies of it to make ConvMixer with `n_layers`.
        * `n_layers` is the number of ConvMixer layers (or depth), $d$.
        * `patch_emb` is the [patch embeddings layer](#PatchEmbeddings).
        * `classification` is the [classification head](#ClassificationHead).
        """
        super().__init__()
        # Patch embeddings
        self.patch_emb = patch_emb
        # Classification head
        self.classification = classification
        # Make copies of the [ConvMixer layer](#ConvMixerLayer)
        self.conv_mixer_layers = clone_module_list(conv_mixer_layer, n_layers)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Get patch embeddings. This gives a tensor of shape `[batch_size, d_model, height / patch_size, width / patch_size]`.
        x = self.patch_emb(x)

        # Pass through [ConvMixer layers](#ConvMixerLayer)
        for layer in self.conv_mixer_layers:
            x = layer(x)

        # Classification head, to get logits
        x = self.classification(x)

        #
        return x