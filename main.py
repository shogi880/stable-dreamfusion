import torch
import argparse
import os

from nerf.provider import NeRFDataset
from nerf.utils import *
from optimizer import Shampoo

from nerf.gui import NeRFGUI

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --fp16 --dir_text")
    parser.add_argument('-O3', action="store_true", help="3d-to-3d")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='/home/acc12252dc/linked_tmp/logs/dreamfusion')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=64, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='grid', help="nerf backbone, choose from [grid, vanilla]")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true', help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--negative_dir_text', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    parser.add_argument('--lambda_entropy', type=float, default=1e-4, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=0.1, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=0, help="loss scale for orientation")
    # parser.add_argument('--lambda_surface', type=float, default=1e-2, help="loss scale for surface preservation")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    
    ### additional options
    # parser.add_argument('--nerf_transfer', action="store_true", help="transfer nerf from pre-trained nerf")
    parser.add_argument('--gt_dir', type=str, default="dataset/pose_2", help='path to gt data')
    parser.add_argument('--gt_images_path', type=str, default="dataset/elsa", help='path to gt data')
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help="use image guidance instead of text guidance")
    parser.add_argument('--reload_model', action="store_true", help="restart the whole training process")
    parser.add_argument('--tune_sd_at_start', action="store_true", help="restart the whole training process")
    
    # parser.add_argument('--back_view_prompt', type=str, default=None, help="set non-prompt when rendering back view")
    parser.add_argument('--sd_version', type=str, default='CompVis', help="choose from [CompVis, waifu, disney]")
    # parser.add_argument('--surface_threshold', type=float, default=1.0, help="threshold for surface")
    parser.add_argument('--sd_tune_iter', type=int, default=100, help="frequency to tune SD")
    parser.add_argument('--sd_tune_step', type=int, default=1, help="the tuning step per tune sd")
    parser.add_argument('--sd_tune_at_n_iter', type=int, default=1000, help="frequency to tune SD")

    parser.add_argument('--subject_text', type=str, default=None, help="text for the subject")
    # parser.add_argument('--classes', type=str, default=None, help="related classes")
    
    parser.add_argument('--transfer_type', type=str, default=None, help="select from [t_inversion, dream_booth, original]")
    parser.add_argument('--ex_name', type=str, default='test', help="create experiment log folder")
    
    opt = parser.parse_args()

    
    workspace = os.path.join(opt.workspace, opt.ex_name, f'{opt.text.replace(" ", "_")}_subject_{opt.subject_text}_seed_{opt.seed}', f'lambda_entropy_{opt.lambda_entropy}_opacity_{opt.lambda_opacity}_orient{opt.lambda_orient}_smooth_{opt.lambda_smooth}')
    
    if opt.O:
        opt.fp16 = True
        opt.dir_text = True
        opt.negative_dir_text = True
        opt.cuda_ray = True

        # opt.lambda_entropy = 1e-4
        # opt.lambda_opacity = 0
        # opt.h = 128  # get OOM using 256...
        # opt.w = 128
        
    elif opt.O2:
        opt.fp16 = True
        opt.dir_text = True
        opt.negative_dir_text = True

        opt.lambda_entropy = 1e-4 # necessary to keep non-empty
        opt.lambda_opacity = 3e-3 # no occupancy grid, so use a stronger opacity loss.

    elif opt.O3:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.save_mesh = True
        
        opt.h = 256
        opt.w = 256
        opt.iters = 5000  # it's enough for pre-training.
        
        if opt.transfer_type is not None: # during only pretrain.
            opt.iters = 10000
            opt.dir_text = True

            workspace = os.path.join(
                workspace, 
                f'{opt.pretrain_ckpt.split("/")[-1].split(".")[0]}_lr_{opt.lr}_iters_{opt.iters}', 
                f'sd_tune_params_iter_{opt.sd_tune_iter}_step{opt.sd_tune_step}_start_at_{opt.sd_tune_at_n_iter}'
                )
    opt.workspace = workspace
    
 
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)

    seed_everything(opt.seed)

    # get model.
    model = NeRFNetwork(opt)
    
    
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)
            
        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
            trainer.test(test_loader)
            
            if opt.save_mesh:
                trainer.save_mesh(resolution=256)
    
    else:
        
        if opt.guidance == 'stable-diffusion':
            from nerf.sd import StableDiffusion
            # add a stable diffusion guidance model
            guidance = StableDiffusion(device, opt)
        elif opt.guidance == 'clip':
            from nerf.clip import CLIP
            guidance = CLIP(device)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        if opt.tune_sd_at_start and opt.sd_version != 'disney':
            # getting the ground true image to train the dreambooth
            normalize = lambda x : (x - 0.5) * 2
            def load_gt_images(path):
                images = []
                for img in os.listdir(path):
                    img_tensor = transforms.ToTensor()(Image.open(os.path.join(path, img)).resize((256, 256))).unsqueeze(0)
                    if img_tensor.shape[1] != 3:
                        img_tensor = img_tensor[:, :3, :, :]
                    images.append(img_tensor)
                images = normalize(torch.concat(images, dim=0).to(device).permute(0, 2, 3, 1))  # since we apply permute(0, 3, 1, 2) in the forward pass
                
                return images
                
            training_views = load_gt_images(opt.gt_images_path)
            
            os.makedirs(os.path.join(opt.workspace, 'gt_images'), exist_ok=True)
            guidance = train_dreambooth(guidance, opt.subject_text, training_views, 
                                        tmp_logs=opt.workspace, max_train_steps=opt.sd_tune_step)
            images = guidance.prompt_to_img("modern disney style, sks elsa")

            for i, image in enumerate(images):
                Image.fromarray(image).save(f'{opt.workspace}/end_tune{i}.png')
                
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        # optimizer = lambda model: Shampoo(model.get_params(opt.lr))

        train_dataset = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100)
        few_shot_views = train_dataset.few_shot_rays
        train_loader = train_dataset.dataloader()

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        # scheduler = lambda optimizer: optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=opt.iters, pct_start=0.1)

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True, few_shot_views=few_shot_views)

        # model.load_state_dict(torch.load(opt.load_model))
        if opt.reload_model:
            trainer.load_checkpoint(opt.pretrain_ckpt, model_only=True)
        
        if opt.gui:
            trainer.train_loader = train_loader # attach dataloader to trainer

            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
            trainer.test(test_loader)

            if opt.save_mesh:
                trainer.save_mesh(resolution=256)
                
