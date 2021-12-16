import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from correlation_package import Correlation

from package_core.net_basics import *
from forward_warp_package import *

def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 

    return grid

class Pred_image(nn.Module):
    def __init__(self, nc_in, nc_out=3):
        super(Pred_image, self).__init__()
        self.conv1 = nn.Conv2d(nc_in, nc_out, kernel_size=3, stride=1, padding=1)
        
        self.leakyRELU2 = nn.LeakyReLU(0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        x = self.leakyRELU2(x)

        return x

class Image_encoder(nn.Module):
    def __init__(self, nc_in, nc_init):
        super(Image_encoder, self).__init__()
        
        self.conv_e = conv2d(in_planes=nc_in, 
                            out_planes=nc_init//2, 
                            batch_norm=False, 
                            activation=nn.ReLU(), 
                            kernel_size=7,
                            stride=1)###kernel_size=7
        self.resnet_block_e = Cascade_resnet_blocks(in_planes=nc_init//2, n_blocks=3)
        #nc_init//2,
        
        self.conv_0 = conv2d(in_planes=nc_init//2, 
                            out_planes=nc_init, 
                            batch_norm=False, 
                            activation=nn.ReLU(), 
                            kernel_size=3,
                            stride=2)
        self.resnet_block_0 = Cascade_resnet_blocks(in_planes=nc_init, n_blocks=3)

        self.conv_1 = conv2d(in_planes=nc_init, 
                            out_planes=nc_init*2, 
                            batch_norm=False, 
                            activation=nn.ReLU(), 
                            kernel_size=3, 
                            stride=2)
        self.resnet_block_1 = Cascade_resnet_blocks(in_planes=nc_init*2, n_blocks=3)

        self.conv_2 = conv2d(in_planes=nc_init*2, 
                            out_planes=nc_init*3, 
                            batch_norm=False, 
                            activation=nn.ReLU(), 
                            kernel_size=3, 
                            stride=2)
        self.resnet_block_2 = Cascade_resnet_blocks(in_planes=nc_init*3, n_blocks=3)
        
        self.conv_3 = conv2d(in_planes=nc_init*3, 
                            out_planes=nc_init*4, 
                            batch_norm=False, 
                            activation=nn.ReLU(), 
                            kernel_size=3, 
                            stride=2)
        self.resnet_block_3 = Cascade_resnet_blocks(in_planes=nc_init*4, n_blocks=3)        
                
    def forward(self, x):
        xe = self.resnet_block_e(self.conv_e(x ))
        x0 = self.resnet_block_0(self.conv_0(xe))
        #x0 = self.resnet_block_0(self.conv_0(x))
        
        x1 = self.resnet_block_1(self.conv_1(x0))
        x2 = self.resnet_block_2(self.conv_2(x1))
        x3 = self.resnet_block_3(self.conv_3(x2))
        return x3, x2, x1, x0

class Corr_decoder(nn.Module):
    def __init__(self, md=4):
        super(Corr_decoder, self).__init__()
        
        self.corr1    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.corr2    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU2 = nn.LeakyReLU(0.1)

    def forward(self, in0, in1):#Interactive cost volume
        #corr = torch.cat([self.corr1(in0, in1),self.corr2(in1, in0)], dim=1)
        #corr = self.leakyRELU2(corr)
        corr = self.leakyRELU2(self.corr1(in0, in1)) 
        return corr

class Flow_decoder(nn.Module):
    def __init__(self, filters, channel_c, md=4):
        super(Flow_decoder, self).__init__()

        self.filters = filters
        self.channel_c = channel_c #Sum of concatting channels of (l)-th and (l-1)-th

        # level l
        dd = np.cumsum(filters)        
        #nd = 2*(2*md+1)**2
        nd = (2*md+1)**2
        od = nd + channel_c
        self.conv0 = conv2d(od,      filters[0], batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv1 = conv2d(od+dd[0],filters[1], batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv2 = conv2d(od+dd[1],filters[2],  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv3 = conv2d(od+dd[2],filters[3],  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv4 = conv2d(od+dd[3],filters[4],  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.predict_flow0 = self.predict_flow(od+dd[4])
        
        self.leakyRELU2 = nn.LeakyReLU(0.1)
        
    def predict_flow(self, in_planes):
        return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

    def forward(self, corr, c_l, c_l_1):
        if c_l_1 is not None: #last layer is set as None
            corr_ = torch.cat([corr , c_l  ], dim=1)
            corr_ = torch.cat([corr_, c_l_1], dim=1)
        else:
            corr_ = torch.cat([corr , c_l  ], dim=1)

        x = torch.cat((self.conv0(corr_), corr_),1)
        x = torch.cat((self.conv1(x), x),1)
        x = torch.cat((self.conv2(x), x),1)
        x = torch.cat((self.conv3(x), x),1)
        x = torch.cat((self.conv4(x), x),1)
        flow = self.predict_flow0(x)

        flow = self.leakyRELU2(flow)

        return flow

class Flow_sup_decoder(nn.Module):
    def __init__(self):
        super(Flow_sup_decoder, self).__init__()

        self.deconv2 = self.deconv(2, 2, kernel_size=4, stride=2, padding=1) 

    def deconv(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1):
        return nn.ConvTranspose2d(int(in_planes), out_planes, kernel_size, stride, padding, bias=True)

    def forward(self, flow):
        upflow = self.deconv2(flow)*2.0

        #upflow = F.interpolate(flow, scale_factor=2, mode='nearest')*2.0

        return upflow

class Feature_map_decoder(nn.Module):
    def __init__(self, est_vel):
        super(Feature_map_decoder, self).__init__()

        self.est_vel = est_vel      

    def forward(self, x, upflow):

        B,C,H,W = x.size()
        warper2 = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
        
        upflow_dis = upflow
        
        '''
        if self.est_vel:#1-th frame RS  True
            grid_rows=generate_2D_grid(H, W)[1]
            t_flow_ref_to_row0=grid_rows.unsqueeze(0).unsqueeze(0)
            t_flow_ref_to_row0=-(t_flow_ref_to_row0-H+1)/H
            upflow_dis=upflow*t_flow_ref_to_row0
        else:#2-th frame RS  False
            grid_rows=generate_2D_grid(H, W)[1]
            t_flow_ref_to_row0=grid_rows.unsqueeze(0).unsqueeze(0)/H
            upflow_dis=upflow*t_flow_ref_to_row0
        '''
        
        x_warped, mask = warper2(x, upflow_dis)
        
        return x_warped, mask, upflow_dis


class Image_Pro(nn.Module):
    def __init__(self, f_channel):
        super(Image_Pro, self).__init__()

        self.pred_im2 = Pred_image(nc_in=f_channel, nc_out=3) 
        self.resnet_block_2 = Cascade_resnet_blocks(in_planes=f_channel, n_blocks=3)

    def forward(self, x_warped):
        x2 = self.resnet_block_2(x_warped)
        im = self.pred_im2(x2)
        
        return im


class encoder_decoder(nn.Module):
    def __init__(self, nc_in, est_vel, filters, channel_c, md=4):
        super(encoder_decoder, self).__init__()

        self.flow_decoder = Flow_decoder(filters, channel_c, md)   
        self.flow_sup_decoder = Flow_sup_decoder()   
        self.feature_map_decoder = Feature_map_decoder(est_vel)   

    def forward(self, corr, c_l, c_l_1, flow_pre):
        
        flow = self.flow_decoder(corr, c_l, flow_pre)
        upflow = self.flow_sup_decoder(flow)
        f_warped, mask, upflow_dis = self.feature_map_decoder(c_l_1, upflow)

        return flow, upflow_dis, f_warped, mask


class Image_decoder(nn.Module):
    def __init__(self, nc_init, layer, nc_out):
        super(Image_decoder, self).__init__()
        
        nc=nc_out*2
        if layer == 3:
            self.pred_im_0 = Pred_image(nc_in=nc_init*layer  , nc_out=nc_out)
            self.pred_im_1 = Pred_image(nc_in=nc_init*layer  , nc_out=nc_out)
            self.pred_im_m = Pred_image(nc_in=nc_init*layer*2+nc_out*2, nc_out=nc_out)
            self.resnet_block_0 = Cascade_resnet_blocks(in_planes=nc_init*layer  , n_blocks=3)
            self.resnet_block_1 = Cascade_resnet_blocks(in_planes=nc_init*layer  , n_blocks=3)
            self.resnet_block_m = Cascade_resnet_blocks(in_planes=nc_init*layer*2+nc_out*2, n_blocks=3)
            self.upconv_0 = deconv2d(in_planes=nc_init*layer, out_planes=nc_out)
            self.upconv_1 = deconv2d(in_planes=nc_init*layer, out_planes=nc_out)
            self.upconv_m = deconv2d(in_planes=nc_init*layer*2+nc_out*2, out_planes=nc_out)
        else:
            self.pred_im_0 = Pred_image(nc_in=nc_init*layer+nc, nc_out=nc_out)
            self.pred_im_1 = Pred_image(nc_in=nc_init*layer+nc, nc_out=nc_out)
            self.pred_im_m = Pred_image(nc_in=nc_init*layer*2+nc_out*4, nc_out=nc_out)
            self.resnet_block_0 = Cascade_resnet_blocks(in_planes=nc_init*layer+nc, n_blocks=3)
            self.resnet_block_1 = Cascade_resnet_blocks(in_planes=nc_init*layer+nc, n_blocks=3)
            self.resnet_block_m = Cascade_resnet_blocks(in_planes=nc_init*layer*2+nc_out*4, n_blocks=3)
            self.upconv_0 = deconv2d(in_planes=nc_init*layer+nc, out_planes=nc_out)
            self.upconv_1 = deconv2d(in_planes=nc_init*layer+nc, out_planes=nc_out)
            self.upconv_m = deconv2d(in_planes=nc_init*layer*2+nc_out*4, out_planes=nc_out)

        self.deconv_0 = self.deconv(3, 3, kernel_size=4, stride=2, padding=1)
        self.deconv_1 = self.deconv(3, 3, kernel_size=4, stride=2, padding=1)
        self.deconv_m = self.deconv(3, 3, kernel_size=4, stride=2, padding=1)
    
    def deconv(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1):
        return nn.ConvTranspose2d(int(in_planes), out_planes, kernel_size, stride, padding, bias=True)
        
    def forward(self, f_warped_0, f_warped_1, f_warped_m_m, f_warped_0_m, f_warped_1_m):
        
        f_warped_m = torch.cat([f_warped_0, f_warped_1], dim=1)
        if f_warped_m_m is not None:
            f_warped_0 = torch.cat([f_warped_0, f_warped_0_m], dim=1)
            f_warped_1 = torch.cat([f_warped_1, f_warped_1_m], dim=1)
            f_warped_m = torch.cat([f_warped_m, f_warped_m_m], dim=1)
        
        x0 = self.resnet_block_0(f_warped_0)
        im_0 = self.pred_im_0(x0)
        #im_0_ = F.interpolate(im_0, scale_factor=2, mode='nearest')#
        im_0_ = self.deconv_0(im_0)
        up_x0 = self.upconv_0(x0)
        f_warped_0_output = torch.cat([up_x0, im_0_], dim=1)
        
        #Weight sharing
        x1 = self.resnet_block_0(f_warped_1)
        im_1 = self.pred_im_0(x1)
        #im_1_ = F.interpolate(im_1, scale_factor=2, mode='nearest')#
        im_1_ = self.deconv_1(im_1)
        up_x1 = self.upconv_1(x1)
        f_warped_1_output = torch.cat([up_x1, im_1_], dim=1)
        
        f_warped_m = torch.cat([ torch.cat([im_0, im_1], dim=1), f_warped_m], dim=1)
        
        xm = self.resnet_block_m(f_warped_m)
        im_m = self.pred_im_m(xm)
        #f_warped_m_output = F.interpolate(im_m, scale_factor=2, mode='nearest')#
        im_m_ = self.deconv_m(im_m)
        up_xm = self.upconv_m(xm)
        f_warped_m_output = torch.cat([up_xm, im_m_], dim=1)
        
        return [f_warped_m_output, f_warped_0_output, f_warped_1_output, im_m, im_0, im_1]        



# main network
class Net_flow_symmetry(nn.Module):
    def __init__(self, nc_in, nc_init, est_vel, md=[4,4,4,4]):
        super(Net_flow_symmetry, self).__init__()
        self.nc_in = nc_in
        self.nc_init = nc_init
        self.est_vel = est_vel
        self.md = md
        
        est_vel_  = False
        est_vel = True   ###

        filters_1 = [128,128,96,64,32]
        filters_2 = [64,64,48,32,16]
        filters_3 = [32,32,24,16,8]
        filters_4 = [32,32,24,16,8]

        self.encoder = Image_encoder(nc_in, nc_init)
        
        self.corr_decoder_0_1 = Corr_decoder(md[0])
        self.corr_decoder_1_1 = Corr_decoder(md[0])
        self.encoder_decoder_0_1 = encoder_decoder(nc_in, est_vel , filters_1, nc_init*4  , md[0])
        self.encoder_decoder_1_1 = encoder_decoder(nc_in, est_vel_, filters_1, nc_init*4  , md[0])

        self.corr_decoder_0_2 = Corr_decoder(md[1])
        self.corr_decoder_1_2 = Corr_decoder(md[1])
        self.encoder_decoder_0_2 = encoder_decoder(nc_in, est_vel , filters_2, nc_init*3+2, md[1])
        self.encoder_decoder_1_2 = encoder_decoder(nc_in, est_vel_, filters_2, nc_init*3+2, md[1])
        
        self.corr_decoder_0_3 = Corr_decoder(md[2])
        self.corr_decoder_1_3 = Corr_decoder(md[2])
        self.encoder_decoder_0_3 = encoder_decoder(nc_in, est_vel , filters_3, nc_init*2+2, md[2])
        self.encoder_decoder_1_3 = encoder_decoder(nc_in, est_vel_, filters_3, nc_init*2+2, md[2])
        
        self.corr_decoder_0_4 = Corr_decoder(md[3])
        self.corr_decoder_1_4 = Corr_decoder(md[3])
        self.encoder_decoder_0_4 = encoder_decoder(nc_in, est_vel , filters_4, nc_init + 2, md[3])
        self.encoder_decoder_1_4 = encoder_decoder(nc_in, est_vel_, filters_4, nc_init + 2, md[3])
        
        
        self.image_decoder_1 = Image_decoder(nc_init, 3, nc_in)
        self.image_decoder_2 = Image_decoder(nc_init, 2, nc_in)
        self.image_decoder_3 = Image_decoder(nc_init, 1, nc_in)
          
        self.deconv2 = self.deconv(3, 3, kernel_size=4, stride=2, padding=1)
        self.resizeX2 = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1)
        
    def deconv(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1):
        return nn.ConvTranspose2d(int(in_planes), out_planes, kernel_size, stride, padding, bias=True)
            
    def forward(self, im_rs):
        im_rs0 = im_rs[:,0:self.nc_in,:,:].clone()
        im_rs1 = im_rs[:,self.nc_in:self.nc_in*2,:,:].clone()
        
        im_rs0_clone=[im_rs0]
        im_rs1_clone=[im_rs1]
        for i in range(3):
            im_rs0_clone.append(F.interpolate(im_rs0_clone[-1],scale_factor=0.5))
            im_rs1_clone.append(F.interpolate(im_rs1_clone[-1],scale_factor=0.5))

        # Pyramid
        x0_0 = im_rs0
        x1_0 = im_rs1
        x0_4, x0_3, x0_2, x0_1 = self.encoder(im_rs0)
        x1_4, x1_3, x1_2, x1_1 = self.encoder(im_rs1)
        
        #Backbone of dual-stream network
        corr0_1 = self.corr_decoder_0_1(x0_4, x1_4)
        corr1_1 = self.corr_decoder_1_1(x1_4, x0_4)
        flow_0_1, upflow_dis_0_1, f_warped_0_1, mask_0_1 = self.encoder_decoder_0_1(corr0_1, x0_4, x0_3, None)
        flow_1_1, upflow_dis_1_1, f_warped_1_1, mask_1_1 = self.encoder_decoder_1_1(corr1_1, x1_4, x1_3, None)

        corr0_2 = self.corr_decoder_0_2(f_warped_0_1, f_warped_1_1)
        corr1_2 = self.corr_decoder_1_2(f_warped_1_1, f_warped_0_1)
        flow_0_2, upflow_dis_0_2, f_warped_0_2, mask_0_2 = self.encoder_decoder_0_2(corr0_2, x0_3, x0_2, upflow_dis_0_1)
        flow_1_2, upflow_dis_1_2, f_warped_1_2, mask_1_2 = self.encoder_decoder_1_2(corr1_2, x1_3, x1_2, upflow_dis_1_1)
        
        corr0_3 = self.corr_decoder_0_3(f_warped_0_2, f_warped_1_2)
        corr1_3 = self.corr_decoder_1_3(f_warped_1_2, f_warped_0_2)
        flow_0_3, upflow_dis_0_3, f_warped_0_3, mask_0_3 = self.encoder_decoder_0_3(corr0_3, x0_2, x0_1, upflow_dis_0_2)
        flow_1_3, upflow_dis_1_3, f_warped_1_3, mask_1_3 = self.encoder_decoder_1_3(corr1_3, x1_2, x1_1, upflow_dis_1_2)
        
        # Predict GS images
        img1 = self.image_decoder_1(f_warped_0_1, f_warped_1_1, None, None, None)
        img2 = self.image_decoder_2(f_warped_0_2, f_warped_1_2, img1[0], img1[1], img1[2])
        img3 = self.image_decoder_3(f_warped_0_3, f_warped_1_3, img2[0], img2[1], img2[2])
        
        im_pre_ = F.interpolate(img3[3], scale_factor=2, mode='bilinear')
        im_pre = self.resizeX2(im_pre_)
        
        ims = [img3[3], img3[4], img3[5], img2[3], img2[4], img2[5], img1[3], img1[4], img1[5], im_pre]
        masks = [mask_0_3, mask_1_3, mask_0_2, mask_1_2, mask_0_1, mask_1_1]
        flows = [upflow_dis_0_3, upflow_dis_1_3, upflow_dis_0_2, upflow_dis_1_2, upflow_dis_0_1, upflow_dis_1_1]


        return ims, masks, flows

