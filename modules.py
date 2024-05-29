import torch
import torch.nn as nn
import functools

class KeypointGenerator(nn.Module):
    ''' Defines a UNet-based generator. '''

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        '''
        Init function.

        Parameters:
            input_nc (int):  The number of input image channels.
            output_nc (int): The number of output channels.
            num_downs (int): The number of down-sampling layers in the UNet.
            ngf (int):       The base feature space of the network.
            norm_layer:      The type of normalization layer to use.
        '''

        super(KeypointGenerator, self).__init__()
        n_residual_blocks = 8

        self.input = UnetDownBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.down1 = UnetDownBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.down2 = UnetDownBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.down3 = UnetDownBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)

        # n_keyp = 0 if self.use_keypoint else 0
        res_ext_encoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_encoder += [ResidualBlock(ngf * 8)]
        self.ext_f1 = nn.Sequential(*res_ext_encoder)

        self.keypoint_layer = nn.Linear(2, 256)

        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(ngf * 8)]
        self.ext_f2 = nn.Sequential(*res_ext_decoder)

        self.up1 = UnetUpBlock(ngf * 4, (ngf * 8) + (ngf * 8), input_nc=None, norm_layer=norm_layer, innermost=True)
        self.up2 = UnetUpBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.up3 = UnetUpBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.output = UnetUpBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(ngf * 8)]
        self.kp_ext_f2 = nn.Sequential(*res_ext_decoder)

        self.kp_up1 = UnetUpBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)
        self.kp_up2 = UnetUpBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.kp_up3 = UnetUpBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.kp_output = UnetUpBlock(3, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        # Encode the inputs.
        x0 = self.input(input)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        features = self.ext_f1(x3)

        out_dict = {}

        # Main decoder branch. Facilitates ,the style transfer task.
        x = self.ext_f2(features)
        x = self.up1(torch.cat((x, x3), 1))
        x = self.up2(torch.cat((x, x2), 1))
        x = self.up3(torch.cat((x, x1), 1))
        out_dict['output'] = self.output(torch.cat((x, x0), 1)).tanh()

        # Secondary decoder branch. Enables learning of quantifiable information.
        # NOTE: We detach the gradients of the encoder outputs here. We found
        # that the detaching of gradients resembles a sort of task decomposition,
        # because the encoder only needs to optimize for the main branch's task.
        x = self.kp_ext_f2(features.detach())
        x = self.kp_up1(torch.cat((x, x3), 1))
        x = self.kp_up2(torch.cat((x, x2), 1))
        x = self.kp_up3(torch.cat((x, x1), 1))
        out_dict['keypoints'] = self.kp_output(torch.cat((x, x0), 1)).tanh()

        return out_dict


class UnetDownBlock(nn.Module):
    ''' Defines a UNet down-sampling block with skip connections. '''

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        '''
        Init function.

        Parameters:
            outer_nc (int):     The number of filters in the outer conv layer.
            inner_nc (int):     The number of filters in the inner conv layer.
            input_nc (int):     The number of channels in input image.
            outermost (bool):   Whether the submodule is the first down-sampling block.
            innermost (bool):   Whether the submodule is the last down-sampling block.
            norm_layer:         The type of normalization layer to use.
            use_dropout (bool): Whether to use a dropout layer.
        '''
        super(UnetDownBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        if outermost: model = [downconv]
        elif innermost: model = [downrelu, downconv]
        else:
            model = [downrelu, downconv, downnorm]
            if use_dropout: model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    

class UnetUpBlock(nn.Module):
    ''' Defines a UNet up-sampling block with skip connections. '''

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        '''
        Init function.

        Parameters:
            outer_nc (int):     The number of filters in the outer conv layer.
            inner_nc (int):     The number of filters in the inner conv layer.
            input_nc (int):     The number of channels in input image.
            outermost (bool):   Whether the submodule is the first down-sampling block.
            innermost (bool):   Whether the submodule is the last down-sampling block.
            norm_layer:         The type of normalization layer to use.
            use_dropout (bool): Whether to use a dropout layer.
        '''
        super(UnetUpBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [uprelu, upconv]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [uprelu, upconv, upnorm]
            if use_dropout: model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    ''' Defines a residual block. '''

    def __init__(self, in_features, alt_leak=False, neg_slope=1e-2):
        '''
        Init function.

        Parameters:
            in_features (int):  The number of input features.
            alt_leak (bool):    Whether to use LeakyReLU (True) or ReLU (False).
            neg_slope (float):  The negative slope of the LeakyReLU.
        '''
        
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x): return x + self.conv_block(x)


class NLayerDiscriminator(nn.Module):
    ''' Defines a PatchGAN discriminator. '''

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        '''
        Init function.

        Parameters:
            input_nc (int):  The number of input image channels.
            ngf (int):       The base feature space of the network.
            n_layers (int):  The number of down-sampling layers.
            norm_layer:      The type of normalization layer to use.
        '''

        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input): return self.model(input)