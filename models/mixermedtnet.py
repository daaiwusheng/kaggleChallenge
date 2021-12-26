import torch
from torch import nn

from labml_helpers.module import Module
import torch.nn.functional as F
from blocks import *
from labml_nn.utils import clone_module_list

import torch.nn as nn

from models.modules import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


class ConvMixerLayer(Module):
    """
    <a id="ConvMixerLayer"></a>
    ## ConvMixer layer
    This is a single ConvMixerLayer layer. The model will have a series of these.
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


class UNet2D(nn.Module):
    def __init__(self, in_channels, conv_depths=(64, 128, 256, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet2D, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last2D(conv_depths[1], conv_depths[0]))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec


def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape
    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)



class ResUnetPlusPlus(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512, 1024]):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        # self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        out = self.aspp_out(x8)
        # out = self.output_layer(x9)

        return out




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
        self.convtrans1 = nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.GELU()
        self.batchnorm1 = nn.BatchNorm2d(d_model)

        self.convtrans2 = nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.act2 = nn.GELU()
        self.batchnorm2 = nn.BatchNorm2d(d_model)

        self.convtrans3 = nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.act3 = nn.GELU()
        self.batchnorm3 = nn.BatchNorm2d(d_model)
        #这里尽量不要考虑上采用函数，因为这个线性插值的纯粹的数值计算是不能学习的，反卷积可以做到上采样
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        #1*1卷积这里，无论前面输出多好channel， 这里直接拿来作为输入就行了
        self.conv = nn.Conv2d(d_model, 64, kernel_size=1, stride=1, padding=0)
        #由于目前用的交叉商函数自带softmax， 所以这里就不需要加入softmax了
        # Linear layer
        # self.linear = nn.Linear(d_model, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # Average pooling
        # x = self.pool(x)
        x = self.convtrans1(x)
        x = self.act1(x)
        x = self.batchnorm1(x)

        x = self.convtrans2(x)
        x = self.act2(x)
        x = self.batchnorm2(x)

        x = self.convtrans3(x)
        x = self.act3(x)
        x = self.batchnorm3(x)

        x = self.conv(x)
        # x = self.upsample(x)
        # Get the embedding, `x` will have shape `[batch_size, d_model, 1, 1]`
        # x = x[:, :, 0, 0]
        # Linear layer
        # x = self.linear(x)

        # print(x)
        # print('*'*25)
        # x = self.adjust(x)
        # x = self.softmax(x)
        # print(x.shape)
        # print(x)

        #
        return x


class Segmentation(Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.adjust = nn.Conv2d(d_model, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.adjust(x)

        return x


class ConvUnetMixer(Module):
    """
    ## ConvMixer
    This combines the patch embeddings block, a number of ConvMixer layers and a classification head.
    """
    def __init__(self, conv_mixer_layer: ConvMixerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings,
                 unet: ResUnetPlusPlus,
                 classification: ClassificationHead,
                 segmentation: Segmentation):
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
        # Unet layer
        self.unet = unet
        # Segmentation layer
        self.classification = classification
        self.segmentation = segmentation
        # Make copies of the [ConvMixer layer](#ConvMixerLayer)
        self.conv_mixer_layers = clone_module_list(conv_mixer_layer, n_layers)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        xin = x.clone()
        # Get patch embeddings. This gives a tensor of shape `[batch_size, d_model, height / patch_size, width / patch_size]`.
        x = self.patch_emb(x)

        # Pass through [ConvMixer layers](#ConvMixerLayer)
        for layer in self.conv_mixer_layers:
            x = layer(x)

        x = self.classification(x)

        x_u = self.unet(xin)

        x = x + x_u

        # Segmentation output
        x = self.segmentation(x)

        #
        return x