import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from .gauss_pyramid import Gauss_Pyramid_Conv
from .gradcam_utils import GradCAM
import util.util as util

from util.losses import MS_SSIM_Loss
from .PALS import MLPA_LOSS
from .PCLS import UNet_pro, CTPC_LOSS



# Sobel edge operator
def sobel_edges(x):
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], device=x.device).float().view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    edges = 0
    for c in range(x.shape[1]):
        gx = F.conv2d(x[:, c:c+1], sobel_x, padding=1)
        gy = F.conv2d(x[:, c:c+1], sobel_y, padding=1)
        edges += torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return edges



class GradPSPStainModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.add_argument('--CUT_mode', type=str, default='FastCUT',
                            choices=['CUT', 'FastCUT'])
        
        parser.add_argument('--lambda_GAN', type=float, default=1.0)
        parser.add_argument('--lambda_NCE', type=float, default=1.0)
        parser.add_argument('--lambda_gp', type=float, default=1.0)
        parser.add_argument('--lambda_gc_G', type=float, default=0.15)
        parser.add_argument('--lambda_edge', type=float, default=0.5)
        parser.add_argument('--lambda_SSIM', type=float, default=0.5)
        parser.add_argument('--lambda_MLPA', type=float, default=1.0)
        parser.add_argument('--lambda_CTPC', type=float, default=2.5, help='weight for GAN loss: GAN(G(X))')

        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16')
        parser.add_argument('--num_patches', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07)
        parser.add_argument('--netF', type=str, default='mlp_sample')
        parser.add_argument('--netF_nc', type=int, default=256)

        parser.add_argument( '--nce_includes_all_negatives_from_minibatch',action='store_true')
        parser.add_argument('--gp_weights', type=str, default='uniform', help='weights for reconstruction pyramids.')
        parser.add_argument('--flip_equivariance',type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'NCE','MLPA', 'CTPC', 'ssim', 'GP', 'GC_G']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'F', 'D'] if self.isTrain else ['G']

        self.nce_layers = [int(i) for i in opt.nce_layers.split(',')]

        # define networks 
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,opt.netG, opt.normG, not opt.no_dropout,opt.init_type, opt.init_gain,opt.no_antialias, opt.no_antialias_up,self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,self.gpu_ids, opt)
        self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netSeg = UNet_pro(in_chns=3, class_num=2).to(self.device)
        self.netSeg.load_state_dict(torch.load(f'pretrain/{opt.unet_seg}.pth'))
        # Grad-CAM (visual guidance only)
        self.gradcam_viz = GradCAM(self.netD, target_layer='final_conv', detach=False)

        # define loss functions
        gan_mode = getattr(opt, 'gan_mode', 'lsgan')
        self.criterionGAN = networks.GANLoss(gan_mode).to(self.device)
        self.criterionNCE = PatchNCELoss(opt).to(self.device)
        self.criterion_ssim = MS_SSIM_Loss(data_range=1.0, channel=3)
        self.criterionMLPA = MLPA_LOSS().to(self.device)
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)


        if self.opt.lambda_gp > 0:
            self.P = Gauss_Pyramid_Conv(num_high=5)
            self.criterionGP = torch.nn.L1Loss().to(self.device)

            if self.opt.gp_weights == 'uniform':
                self.gp_weights = [1.0] * 6
            else:
                self.gp_weights = eval(self.opt.gp_weights)

            self.loss_names += ['GP']


        self.optimizers = []
        self.optimizer_G = None
        self.optimizer_D = None
        self.optimizer_F = None

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.lr,betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=opt.lr / 10,betas=(opt.beta1, opt.beta2))
            self.optimizers += [self.optimizer_G, self.optimizer_D]


    def data_dependent_initialize(self, data):

        if not self.isTrain:
            return

        self.set_input(data)
        self.forward()
        feat = self.netG(self.real_A, self.nce_layers, encode_only=True)
        _ = self.netF(feat, self.opt.num_patches, None)
        self.optimizer_F = torch.optim.Adam(self.netF.parameters(),lr=self.opt.lr,betas=(self.opt.beta1, self.opt.beta2))
        self.optimizers.append(self.optimizer_F)

    def set_input(self, input, data_init=0):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        self.fake_B = self.netG(self.real_A)


    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        loss_D = self.compute_D_loss()
        loss_D.backward()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()

        if self.optimizer_F:
            self.optimizer_F.zero_grad()

        loss_G = self.compute_G_loss()
        loss_G.backward()

        self.optimizer_G.step()
        if self.optimizer_F:
            self.optimizer_F.step()


    def compute_D_loss(self):

        pred_fake = self.netD(self.fake_B.detach())
        pred_real = self.netD(self.real_B)

        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        ###alpha in {0,1}
        alpha = torch.randint(0, 2, (1,), device=self.real_B.device).item()
        loss_D = alpha * self.loss_D_real + (1 - alpha) * self.loss_D_fake
        return loss_D


    def compute_G_loss(self):

        """Calculate GAN and NCE loss for the generator"""
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean()
        feat_real = self.netG(self.real_A, self.nce_layers, encode_only=True)
        feat_fake = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        self.loss_NCE = self.calculate_NCE_loss(feat_real, feat_fake)

        # SSIM: Structural Similarity Loss
        self.loss_ssim = self.opt.lambda_SSIM * self.criterion_ssim(self.fake_B, self.real_B)

        # MLPA: Multi-Level Protein Awareness Loss
        self.loss_MLPA, mask_A, mask_B = self.criterionMLPA(self.fake_B, self.real_B)
        
        # CTPC: Cross-image Tumor Prototype Consistency Loss
        self.loss_CTPC = 0.0
        for i in range(self.real_B.size(0)):
            imgs = torch.cat([self.fake_B[i:i+1], self.real_B[i:i+1]], 0)
            _, cross = self.netSeg(imgs)
            target = torch.cat([mask_A[i:i+1], mask_B[i:i+1]], 0).long().view(2, 1, -1)
            self.loss_CTPC += torch.mean(
                CTPC_LOSS(cross, target, self.ce_loss))
            self.loss_CTPC *= self.opt.lambda_CTPC

        # GP: Gaussian Pyramid Loss
        if self.opt.lambda_gp > 0:
            pf = self.P(self.fake_B)
            pr = self.P(self.real_B)
            self.loss_GP = sum(w * self.criterionGP(a, b)for a, b, w in zip(pf, pr, self.gp_weights))


        # GC: Grad-CAM Attention-Guided Loss
        req = [p.requires_grad for p in self.netD.parameters()]
        for p in self.netD.parameters():
            p.requires_grad_(False)

        real_cam = self.real_B.detach().requires_grad_(True)
        cam = self.gradcam_viz(real_cam)

        for p, r in zip(self.netD.parameters(), req):
            p.requires_grad_(r)

        cam = F.interpolate(cam, self.real_B.shape[-2:], mode='bilinear')
        cam = cam / (cam.max() + 1e-6)

        cam_l1 = torch.mean(cam * torch.abs(self.fake_B - self.real_B))
        edge_loss = F.l1_loss(sobel_edges(self.fake_B),sobel_edges(self.real_B))
        self.loss_GC_G = self.opt.lambda_gc_G * (cam_l1 + self.opt.lambda_edge * edge_loss)  
        self.loss_G = (self.loss_G_GAN+ self.loss_NCE+ self.opt.lambda_gp * self.loss_GP + self.loss_ssim+ self.loss_MLPA+ self.loss_CTPC+ self.loss_GC_G)
        return self.loss_G


    def calculate_NCE_loss(self, feat_src, feat_tgt):

        feat_k_pool, ids = self.netF(feat_src, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_tgt, self.opt.num_patches, ids)

        loss = 0
        for q, k in zip(feat_q_pool, feat_k_pool):
            loss += self.criterionNCE(q, k).mean()

        return loss / len(feat_q_pool)
