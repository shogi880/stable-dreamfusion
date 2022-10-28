import os
import argparse

parameter_dict = {
    'text' : ['a raccoon astronaut'],
    'subject_text' : ['raccoon'],

    # nerf rendering
    'lambda_entropy' : [0.5, 1, 0.1, 0.001, 0.00001],
    'lambda_opacity' : [0, 0.1, 0.01],
    'lambda_orient' : [0.01, 0.001, 0.1] ,
    'lambda_smooth' : [0, 0.1, 0.01],

    # stable-diffusion-tuning
    'sd_tune_iter' : [100, 10, 50],
    'sd_tune_step' : [1, 5, 10],
    'sd_tune_at_n_iter' : [1000, 500, 100],

    'sd_version' : ['CompVis', 'waifu', 'disney'],
        
    'transfer_type' : ['default', 'dream_booth', 't_inversion'],

    'bound' : [1],
    'dt_gamma' : [0],
    'min_near' : [0.1],
    'radius_range' : [[1.0, 1.5]],
    'fovy_range' : [[40, 70]],
    
    'angle_overhead' : [30],
    'angle_front' : [60],

    # cuda rendering
    'w' : [256],
    'h' : [256],

    # training
    'iters' : [2000, 1000, 3000],
    'lr' : [0.001, 0.0001, 0.00001],
    'albedo_iters' : [1000, 500],
}

def run_command(cmd):
    print('to run: ', cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', default=None, help="select one parameter to tune")
    opt = parser.parse_args()
    
    base_cmd = 'python main.py --pretrain_ckpt ./pretrain_models/pose_1_0050.pth --transfer_type "dream_booth"'
    workspace = f'/home/acc12252dc/linked_tmp/logs/dreamfusion/{opt.tune}'
    
    for num in parameter_dict[opt.tune]:
        cmd_list = [base_cmd + f' --{opt.tune} {str(num)}' + f' --workspace {workspace}']
        for cmd in cmd_list:
            print(cmd)
            # run_command(cmd)