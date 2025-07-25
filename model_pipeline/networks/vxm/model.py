import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from .layers import ConvBlock, InitWeights_He


class VoxelMorph(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 ndims=3,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 predict_residual=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if isinstance(nb_features, int):
            nb_features = None
            
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],             # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ]
        elif nb_features == [16,32,32]:
            nb_features = [
                [16, 32, 32],             # encoder
                [32, 32, 32, 32, 16, 16]  # decoder
            ]
        elif nb_features == [32,64,64]:
            nb_features = [
                [32, 64, 64],             # encoder
                [64, 64, 64, 64, 32, 32]  # decoder
            ]
        elif nb_features == [32,64,64,64]:
            nb_features = [
                [32, 64, 64, 64],             # encoder
                [64, 64, 64, 64, 64, 32, 32]  # decoder
            ]
        elif nb_features == [32,64,128,256]:
            nb_features = [
                [32, 64, 128, 256],             # encoder
                [512, 256, 128, 64, 32, 16, 16]  # decoder
            ]
        elif nb_features == [32,64,128]:
            nb_features = [
                [32, 64, 128],             # encoder
                [256, 128, 64, 32, 16, 16]  # decoder
            ]
        elif nb_features == [64,128,256,512]:
            nb_features = [
                [64,128,256,512],             # encoder
                [1024, 512, 256, 128, 64, 32, 32]  # decoder
            ]
        elif nb_features == [64,128,256]:
            nb_features = [
                [64,128,256],             # encoder
                [512, 256, 128, 64, 32, 32]  # decoder
            ]


        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [lambda x, ref: F.interpolate(x, size=ref.shape[2:], mode='nearest') for _ in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        self.apply(InitWeights_He())

        self.final = nn.Conv3d(prev_nf, ndims, kernel_size=3, padding=1)
        self.final.weight = nn.Parameter(Normal(0, 1e-5).sample(self.final.weight.shape))
        self.final.bias = nn.Parameter(torch.zeros(self.final.bias.shape))

        self.predict_residual = predict_residual

    def forward(self, x):
        # slice initial ddf to learn residual
        if self.predict_residual:
            init_ddf = x[:,0:3].clone() 

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                skip = x_history.pop()
                x = self.upsampling[level](x, skip)
                x = torch.cat([x, skip], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        if self.predict_residual:
            return init_ddf + self.final(x)
        else:
            return self.final(x)
    