
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from collections import deque


from networks import get_network
from networks.pro_D import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, DataConsistencyInKspace_I, DataConsistencyInKspace_K, fft2, complex_abs_eval



class RecurrentModel(nn.Module):
    def __init__(self, opts):
        super(RecurrentModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        self.n_recurrent = opts.n_recurrent
        self.upscale = opts.upscale


        # set default loss flags
        loss_flags = ("w_img_L1")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G_I = get_network(opts)
        self.networks.append(self.net_G_I)

        if self.is_train:
            self.loss_names += ['loss_G_L1']
            param = list(self.net_G_I.parameters())
            self.optimizer_G = torch.optim.Adam(param,
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.opts = opts

        # data consistency layers in image space & k-space
        dcs_I = []
        for i in range(self.n_recurrent):
            dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        self.dcs_I = dcs_I

        dcs_K = []
        for i in range(self.n_recurrent):
            dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        self.dcs_K = dcs_K

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        """
        Handle both single-frame and temporal inputs with frame-specific references.
        
        Expected shapes:
        Single-frame mode:
            ref_image_full: [B, Nref, 2, H, W]
            tag_*:          [B, 2, H, W]
        
        Temporal mode:
            ref_image_full: [B, T, Nref, 2, H, W]  ← Frame-specific!
            tag_*:          [B, T, 2, H, W]
        """
        ref_full = data['ref_image_full'].to(self.device)
        ref_sub = data['ref_image_sub'].to(self.device)
        
        # Handle both single-frame and temporal inputs, ensuring refs have a "view" dimension
        if ref_full.dim() == 5:
            # Single-frame: [B, Nref, 2, H, W] → [B, 1, Nref, 2, H, W]
            self.ref_image_full = ref_full.unsqueeze(1)
            self.ref_image_sub = ref_sub.unsqueeze(1)
        elif ref_full.dim() == 6:
            # Temporal with frame-specific refs: [B, T, Nref, 2, H, W]
            self.ref_image_full = ref_full
            self.ref_image_sub = ref_sub

        self.tag_kspace_full = data['tag_kspace_full'].to(self.device)
        self.tag_image_full = data['tag_image_full'].to(self.device)
        self.tag_image_sub = data['tag_image_sub'].to(self.device)

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        """
        Forward with frame-specific references.
        
        References: [B, T, Nref, 2, H, W] (frame-specific)
        Tag images: [B, T, 2, H, W]
        """
        # ========= Get inputs =========
        x_sub  = self.tag_image_sub      # [B, T, 2, H, W]
        x_full = self.tag_image_full     # [B, T, 2, H, W]
        k_full = self.tag_kspace_full    # [B, T, 2, H, W]

        assert x_sub.dim() == 5, f"Expected tag_image_sub [B,T,C,H,W], got {x_sub.shape}"
        assert x_full.dim() == 5, f"Expected tag_image_full [B,T,C,H,W], got {x_full.shape}"
        assert k_full.dim() == 5, f"Expected tag_kspace_full [B,T,C,H,W], got {k_full.shape}"

        B, T, C, H, W = x_sub.shape

        # ========= Frame-specific references =========
        # After set_input, these are [B, T, Nref, 2, H, W]
        ref_sub_all  = self.ref_image_sub   # [B, T, Nref, 2, H, W]
        ref_full_all = self.ref_image_full  # [B, T, Nref, 2, H, W]

        # ========= Temporal memory =========
        M = getattr(self.opts, 'temporal_memory', 2)
        mem_lr = deque(maxlen=M)
        mem_hr = deque(maxlen=M)

        net = {}
        pred_list = []

        for t in range(T):
            I    = x_sub[:, t]      # [B, 2, H, W]
            k_t  = k_full[:, t]     # [B, 2, H, W]
            gt   = x_full[:, t]     # [B, 2, H, W]
            
            #  Get frame-specific references for this timestep
            base_reflr = ref_sub_all[:, t]   # [B, Nref, 2, H, W]
            base_ref   = ref_full_all[:, t]  # [B, Nref, 2, H, W]

            #  Build temporal views: concat frame-specific refs + memory refs
            if len(mem_lr) > 0:
                mem_reflr = torch.stack(list(mem_lr), dim=1)  # [B, M, 2, H, W]
                mem_ref   = torch.stack(list(mem_hr), dim=1)  # [B, M, 2, H, W]
                # Concat along view dimension: [B, Nref+M, 2, H, W]
                reflr_all = torch.cat([base_reflr, mem_reflr], dim=1)
                ref_all   = torch.cat([base_ref,   mem_ref],   dim=1)
            else:
                reflr_all = base_reflr  # [B, Nref, 2, H, W]
                ref_all   = base_ref    # [B, Nref, 2, H, W]

            # Recurrent DC unrolling
            for i in range(1, self.n_recurrent + 1):
                pred = self.net_G_I(I, reflr_all, ref_all)
                k_pred = fft2(pred)
                I = pred

                net[f't{t}_r{i}_img_pred'] = pred
                net[f't{t}_r{i}_kspc_img_dc'] = k_pred
            
            pred_list.append(pred)

            # Update memory (not detach - allows gradient flow)
            mem_lr.append(x_sub[:, t])
            mem_hr.append(pred)

        self.net = net
        self.pred_out = torch.stack(pred_list, dim=1)  # [B, T, 2, H, W]

    def update_G(self):
        self.optimizer_G.zero_grad()

        # ========= assume temporal input already =========
        x_full = self.tag_image_full
        k_full = self.tag_kspace_full

        assert x_full.dim() == 5, "Expected tag_image_full to be [B,T,C,H,W] (temporal)."
        assert k_full.dim() == 5, "Expected tag_kspace_full to be [B,T,C,H,W] (temporal)."

        B, T, C, H, W = x_full.shape

        loss_img_l1 = 0.0
        loss_kspc   = 0.0

        for t in range(T):
            gt_t = x_full[:, t]
            k_t  = k_full[:, t]

            for i in range(1, self.n_recurrent + 1):
                loss_img_l1 += self.criterion(self.net[f't{t}_r{i}_img_pred'], gt_t)
                loss_kspc   += self.mse(self.net[f't{t}_r{i}_kspc_img_dc'], k_t) * 0.001

        loss_G_L1 = loss_img_l1 + loss_kspc

        self.loss_G_L1  = loss_G_L1.item()
        self.loss_img_l1 = loss_img_l1.item()
        self.loss_kspc   = loss_kspc.item()

        loss_G_L1.backward()
        torch.nn.utils.clip_grad_norm_(self.net_G_I.parameters(), max_norm=1.0)
        self.optimizer_G.step()


    def optimize(self):
        self.loss_G_L1 = 0

        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4f} Img_L1: {:.4f} dc_loss: {:.4f}'.format(self.loss_G_L1, self.loss_img_l1,self.loss_kspc)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        if self.opts.wr_L1 > 0:
            state['net_G_I'] = self.net_G_I.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        if self.opts.wr_L1 > 0:
            self.net_G_I.module.load_state_dict(checkpoint['net_G_I'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()

        recon_images = []
        gt_images = []
        input_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            if self.opts.wr_L1 > 0:
                # Handle temporal sequences [B,T,2,H,W]
                if self.pred_out.dim() == 5:
                    B, T, C, H, W = self.pred_out.shape
                    
                    # Compute metrics per frame, then average
                    psnr_list = []
                    ssim_list = []
                    
                    for t in range(T):
                        # Extract single frame [B,2,H,W]
                        pred_out_t = self.pred_out[:, t]  # [B,2,H,W]
                        gt_t = self.tag_image_full[:, t]  # [B,2,H,W]
                        
                        # Compute magnitude
                        abs_rec_t = complex_abs_eval(pred_out_t)  # [B,1,H,W]
                        abs_gt_t = complex_abs_eval(gt_t)  # [B,1,H,W]
                        
                        # PSNR
                        psnr_list.append(psnr(abs_rec_t, abs_gt_t))
                        
                        # SSIM (first sample only)
                        ssim_val = ssim(
                            abs_rec_t[0, 0, :, :].detach().cpu().numpy(),
                            abs_gt_t[0, 0, :, :].detach().cpu().numpy(),
                            data_range=1.0
                        )
                        ssim_list.append(ssim_val)
                    
                    # Average metrics across all frames
                    psnr_recon = sum(psnr_list) / len(psnr_list)
                    ssim_recon = sum(ssim_list) / len(ssim_list)
                    
                    avg_psnr.update(psnr_recon)
                    avg_ssim.update(ssim_recon)
                    
                    # Save only first frame for visualization
                    recon_images.append(self.pred_out[0, 0].detach().cpu())
                    gt_images.append(self.tag_image_full[0, 0].detach().cpu())
                    input_images.append(self.tag_image_sub[0, 0].detach().cpu())
                    
                else:
                    # Single-frame mode [B,2,H,W]
                    abs_rec = complex_abs_eval(self.pred_out)
                    abs_gt = complex_abs_eval(self.tag_image_full)
                    
                    psnr_recon = psnr(abs_rec, abs_gt)
                    avg_psnr.update(psnr_recon)
                    
                    ssim_recon = ssim(
                        abs_rec[0, 0, :, :].detach().cpu().numpy(),
                        abs_gt[0, 0, :, :].detach().cpu().numpy(),
                        data_range=1.0
                    )
                    avg_ssim.update(ssim_recon)
                    
                    recon_images.append(self.pred_out[0].detach().cpu())
                    gt_images.append(self.tag_image_full[0].detach().cpu())
                    input_images.append(self.tag_image_sub[0].detach().cpu())

            message = 'PSNR: {:4f} '.format(avg_psnr.avg)
            message += 'SSIM: {:4f} '.format(avg_ssim.avg)
            val_bar.set_description(desc=message)

        self.psnr_recon = avg_psnr.avg
        self.ssim_recon = avg_ssim.avg

        self.results = {}
        if self.opts.wr_L1 > 0:
            self.results['recon'] = torch.stack(recon_images).squeeze().numpy()
            self.results['gt'] = torch.stack(gt_images).squeeze().numpy()
            self.results['input'] = torch.stack(input_images).squeeze().numpy()

