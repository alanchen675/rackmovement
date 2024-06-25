
import torch

import os
from logging import getLogger

from RMPEnv import RMPEnv as Env
from RMPModel import RMPModel as Model
from mip_solvers import SubSolver as MIPSolver

from utils.utils import *


class RMPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        if self.tester_params['algorithm']=='pomo':
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            if self.tester_params['algorithm']=='pomo':
                score, aug_score = self._test_one_batch(batch_size)
            elif self.tester_params['algorithm']=='comp_heuristic':
                score, aug_score = self._test_one_batch_comp_heuristic(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def get_aug_factor(self):
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        return aug_factor

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        aug_factor = self.get_aug_factor()

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        reward_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~self.env.periods)
        state, reward, done = self.env.pre_step()
        for period in range(1,self.env.periods+1):
            while not done:
                selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
            ## TODO-while done, call the self.env.middle_reset()
            reward_list = torch.cat((reward_list, reward[:, :, None]), dim=2)
            if period < self.env.periods:
                self.env.middle_reset(period)

        return self.get_scores(batch_size, aug_factor, reward_list)

    def _test_one_batch_comp_heuristic(self, batch_size):
        # Augmentation
        ###############################################
        aug_factor = self.get_aug_factor()

        # Ready
        ###############################################
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
        # POMO Rollout
        ###############################################
        reward_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~self.env.periods)
        state, reward, done = self.env.pre_step()
        for period in range(1,self.env.periods+1):
            action = 0
            while not done:
                selected = action*torch.ones(batch_size, self.env.pomo_size, dtype=torch.int64)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
                action += 1
            ## TODO-while done, call the self.env.middle_reset()
            reward_list = torch.cat((reward_list, reward[:, :, None]), dim=2)
            if period < self.env.periods:
                self.env.middle_reset(period)

        return self.get_scores(batch_size, aug_factor, reward_list)

    def _test_one_batch_mip(self, batch_size):
        # Augmentation
        ###############################################
        aug_factor = self.get_aug_factor()

        # Ready
        ###############################################
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()

        # Initialize MIP solver
        ###############################################
        mip_solver = MIPSolver()

        # POMO Rollout
        ###############################################
        reward_list = torch.zeros(size=(batch_size, 0))
        # shape: (batch, 0~self.env.periods)
        state, reward, done = self.env.pre_step()
        for period in range(1,self.env.periods+1):
            # TODO - Get the MIP actions
            # TODO - Update self.env
            reward_list = torch.cat((reward_list, reward[:, :, None]), dim=2)
            if period < self.env.periods:
                self.env.middle_reset(period)

        return self.get_scores(batch_size, aug_factor, reward_list)

    def get_scores(self, batch_size, aug_factor, reward_list):
        # Return
        ###############################################
        reward = self.gen_long_term_rew(batch_size, reward_list)
        # shape: (batch, pomo, periods)

        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()

    def gen_long_term_rew(self, batch_size, rewards, gamma=0.99):
        discounted_rewards = torch.zeros_like(rewards)
    
        # Iterate over each agent/environment in the tensor
        for a in range(batch_size):
            for b in range(self.env.pomo_size):
                R = 0
                for t in reversed(range(self.env.periods)):
                    R = rewards[a, b, t] + gamma * R
                    discounted_rewards[a, b, t] = R
        
        return discounted_rewards
