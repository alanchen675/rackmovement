##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
import multiprocessing
from utils.utils import create_logger, copy_all_src
import numpy as np

np.random.seed(100)
from RMPTrainer import RMPTrainer as Trainer
from config_clean import config

##########################################################################################
# parameters

env_params = {
    'problem_size': 10,
    'pomo_size': 10,
    'config': config,
    'periods': 1,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [501,],
        'gamma': 0.1
    }
}

trainer_params = {
    'leader_reward': True,
    'leader_reward_alpha': 4,
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 200,
    'train_episodes': 64,
    #'train_batch_size': 64,
    'train_batch_size': 8,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_rmp_10.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to laod.

    }
}

logger_params = {
    'log_file': {
        'desc': 'train__rmp_n10',
        'desc_base': 'train__rmp_n10',
        'filename': 'run_log'
    }
}

##########################################################################################
# main

def main():
    #print(f'Number of CPUs: {multiprocessing.cpu_count()}')
    #num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    #print(f"Using {num_cpus} CPUs as allocated by SLURM.")
    if DEBUG_MODE:
        _set_debug_mode()

    # Set up lists of hyper-parameters
    # Set config object or trainer_params['leader_reward_alpha'] 
    # Revise logger_params['log_file']['desc'] so that results are saved in the corresponding folder

    lr_alpha_list = [2, 4, 8, 16, 32]
    sp_beta_list = [(20, 10, 4), (40, 20, 8)]
    sscale_res_limit = [(3,6,40,60,80,100)]
    lscale_res_limit = [(4,7,50,70,90,110)]

    for lr_alpha in lr_alpha_list:
        logger_params['log_file']['desc'] = logger_params['log_file']['desc_base']+f'-leader_alpha_{lr_alpha}'
        create_logger(**logger_params)
        _print_config()

        trainer = Trainer(env_params=env_params,
                          model_params=model_params,
                          optimizer_params=optimizer_params,
                          trainer_params=trainer_params)

        copy_all_src(trainer.result_folder)

        trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
