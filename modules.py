import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class KeypointLayer(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(KeypointLayer, self).__init__()

        num_classes = 2

        self.input = UnetDownBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.down1 = UnetDownBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.down2 = UnetDownBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.down3 = UnetDownBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)
        self.ext_f1 = nn.Conv2d(ngf * 8, 3 + num_classes, kernel_size=1, stride=1, padding=0)
    
    def forward(self, input):
        x0 = self.input(input)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        features = self.ext_f1(x3)

        features = features.permute(0, 2, 3, 1)
        _, h, w, _ = features.shape

        range_y, range_x = torch.meshgrid(
            torch.arange(h).cuda(),
            torch.arange(w).cuda(),
        )

        output = torch.cat([
            (features[:,:,:,0:1].sigmoid() + range_x[None, :, :, None]) / w,
            (features[:,:,:,1:2].sigmoid() + range_y[None, :, :, None]) / h,
            features[:,:,:,2:3].sigmoid(),
            features[:,:,:,3:].softmax(-1),
        ], -1)

        return output
    

    def filter(self, keyp, threshold=0.2):
        kp = keyp[:,:,:,0:2]
        conf = keyp[:,:,:,2]
        scores, _ = torch.max(keyp[:,:,:,3:], -1)
        one_hot = torch.where(keyp[:,:,:,3:]>threshold, 1, 0)
        scores = scores * conf

        filter = torch.where(scores[:,:,:,None].repeat(1, 1, 1, 5) > threshold,
                             torch.cat((kp, scores[:,:,:,None], one_hot), -1),
                             torch.zeros(size=(1, 16, 16, 5)).cuda())
        return filter
    
    def create_keyp_mask(self, keyp, size=None):
        mask = torch.zeros(size=(1, 2, 256, 256)).cuda()
        keyp[:,:,:2] = torch.round(keyp[:,:,:2] * 255)
        mask[:, keyp[:,:,3].to(int), keyp[:,:,0].to(int), keyp[:,:,1].to(int)] = 1
        return mask
    
    def create_keyp_masks(self, keyp, size=None):
        masks = []
        for kp in keyp[0]:
            x, y, conf, clf = kp
            mask = torch.zeros(size=(1, 2, 256, 256)).cuda()
            x, y, clf = torch.round(x * 255).to(int), torch.round(y * 255).to(int), clf.to(int)
            mask[:, clf, x, y] = 1
            masks.append(mask[None,:,:,:,:])
        return torch.cat(masks)
    
    def visualize_keyp(self, img, mask, keyp):
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np

        fig = plt.figure(figsize=(8, 8))

        def get_color(val): return (255, 0, 0) if int(val)==0 else (255, 255, 0)

        ax = fig.add_subplot(2, 1, 1)
        overlay = np.ascontiguousarray((img[0]*255).permute(1, 2, 0).cpu(), dtype=np.uint8)
        for kp in keyp[0]:
            overlay = cv2.circle(overlay, tuple([int(v) for v in kp[:2]]), 2, get_color(kp[-1]), -1)
        ax.imshow(overlay)

        ax = fig.add_subplot(2, 1, 2)
        ax.imshow(mask[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

        return overlay
    
    
class KeypointGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(KeypointGenerator, self).__init__()
        # self.use_keypoint = use_keypoint
        n_residual_blocks = 8
        alt_leak = False
        neg_slope = 1e-2

        # if use_keypoint: self.keyA2B = KeypointLayer(input_nc=3, output_nc=3, ngf=ngf, norm_layer=norm_layer)

        self.input = UnetDownBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.down1 = UnetDownBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.down2 = UnetDownBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.down3 = UnetDownBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)

        # n_keyp = 0 if self.use_keypoint else 0
        res_ext_encoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_encoder += [ResidualBlock(ngf * 8, alt_leak, neg_slope)]
        self.ext_f1 = nn.Sequential(*res_ext_encoder)

        self.keypoint_layer = nn.Linear(2, 256)

        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(ngf * 8, alt_leak, neg_slope)]
        self.ext_f2 = nn.Sequential(*res_ext_decoder)

        self.up1 = UnetUpBlock(ngf * 4, (ngf * 8) + (ngf * 8), input_nc=None, norm_layer=norm_layer, innermost=True)
        self.up2 = UnetUpBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.up3 = UnetUpBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.output = UnetUpBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(ngf * 8, alt_leak, neg_slope)]
        self.kp_ext_f2 = nn.Sequential(*res_ext_decoder)

        self.kp_up1 = UnetUpBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)
        self.kp_up2 = UnetUpBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.kp_up3 = UnetUpBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.kp_output = UnetUpBlock(3, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        x0 = self.input(input)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        features = self.ext_f1(x3)

        out_dict = {}

        x = self.ext_f2(features)
        x = self.up1(torch.cat((x, x3), 1))
        x = self.up2(torch.cat((x, x2), 1))
        x = self.up3(torch.cat((x, x1), 1))
        out_dict['output'] = self.output(torch.cat((x, x0), 1)).tanh()

        x = self.kp_ext_f2(features.detach())
        x = self.kp_up1(torch.cat((x, x3), 1))
        x = self.kp_up2(torch.cat((x, x2), 1))
        x = self.kp_up3(torch.cat((x, x1), 1))
        out_dict['keypoints'] = self.kp_output(torch.cat((x, x0), 1)).tanh()

        out_dict['both'] = torch.cat((out_dict['output'], out_dict['keypoints']), 1)

        return out_dict

def denorm(x): return x * 0.5 + 0.5

def gen_grid_dist(center_point, grid_size): return (center_point - gen_grid2d(grid_size, 0, 1))**2

def gen_grid2d(grid_size, left_end=0, right_end=1):
    x = torch.linspace(left_end, right_end, grid_size).cuda()
    y, x = torch.meshgrid([x, x])
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(grid_size, grid_size, 2)
    return grid

    
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(UnetGenerator, self).__init__()

        n_residual_blocks = 8
        alt_leak = False
        neg_slope = 1e-2

        self.input = UnetDownBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.down1 = UnetDownBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.down2 = UnetDownBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.down3 = UnetDownBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)

        res_ext_encoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_encoder += [ResidualBlock(ngf * 8, alt_leak, neg_slope)]
        self.ext_f1 = nn.Sequential(*res_ext_encoder)

        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(ngf * 8, alt_leak, neg_slope)]
        self.ext_f2 = nn.Sequential(*res_ext_decoder)

        self.up1 = UnetUpBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)
        self.up2 = UnetUpBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        self.up3 = UnetUpBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        self.output = UnetUpBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        x0 = self.input(input)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        features = self.ext_f1(x3)

        x = self.ext_f2(features)
        x = self.up1(torch.cat((x, x3), 1))
        x = self.up2(torch.cat((x, x2), 1))
        x = self.up3(torch.cat((x, x1), 1))
        outputs = self.output(torch.cat((x, x0), 1)).tanh()

        return outputs

        
class UnetDownBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetDownBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        if outermost:
            model = [downconv]
        elif innermost:
            model = [downrelu, downconv]
        else:
            model = [downrelu, downconv, downnorm]

            if use_dropout: model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class UnetUpBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
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
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            model = [uprelu, upconv]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            model = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            model = [uprelu, upconv, upnorm]

            if use_dropout: model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_features, alt_leak=False, neg_slope=1e-2):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
    
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)