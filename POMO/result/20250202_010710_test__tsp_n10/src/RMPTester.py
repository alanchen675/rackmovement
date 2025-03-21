import torch
import os
import pandas as pd
from logging import getLogger
from RMPEnv import RMPEnv as Env
from RMPModel import RMPModel as Model
from scip_solver import GRBSolver
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

    def reset_perf_dict(self):
        self.perf_dict = {}
        self.perf_dict['obj'] = []
        self.perf_dict['status'] = []
        self.perf_dict['move_cost'] = []
        self.perf_dict['reward'] = []
        self.perf_dict['reward1'] = []
        self.perf_dict['reward2'] = []
        self.perf_dict['reward3'] = []
        self.perf_dict['reward4'] = []

    def run(self):
        self.time_estimator.reset()
        self.reset_perf_dict()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        if self.tester_params['algorithm']!='mip':
            self.perf_dict['batch'] = []
            self.perf_dict['pomo'] = []

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            if self.tester_params['algorithm']=='pomo':
                score, aug_score = self._test_one_batch(batch_size, episode)
            elif self.tester_params['algorithm']=='comp_heuristic':
                score, aug_score = self._test_one_batch_comp_heuristic(batch_size, episode)
            else:
                score, aug_score = self._test_one_batch_mip(batch_size)

            alg = self.tester_params['algorithm']

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
        perf_df = pd.DataFrame(self.perf_dict)
        perf_df.to_csv(f'{self.result_folder}/perf_{alg}.csv')

    def get_aug_factor(self):
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        return aug_factor

    def _test_one_batch(self, batch_size, episode):
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
                state, reward, done, info_list, move_cost, r_list = self.env.step(selected)
            ## TODO-while done, call the self.env.middle_reset()
            reward_list = torch.cat((reward_list, reward[:, :, None]), dim=2)
            if period < self.env.periods:
                self.env.middle_reset(period)

        reward_cpu = reward.cpu().detach().numpy()
        reward_list = [r.cpu().detach().numpy() for r in r_list]
        for batch in range(selected.shape[0]):
            for pomo in range(selected.shape[1]):
                self.perf_dict['batch'].append(batch+episode)
                self.perf_dict['pomo'].append(pomo)
                self.perf_dict['obj'].append(-reward_cpu[batch][pomo])
                self.perf_dict['reward'].append(-reward_cpu[batch][pomo])
                self.perf_dict['status'].append(0)
                self.perf_dict['move_cost'].append(move_cost[batch][pomo].item())
                self.perf_dict['reward1'].append(-r_list[0][batch][pomo].item())
                self.perf_dict['reward2'].append(-r_list[1][batch][pomo].item())
                self.perf_dict['reward3'].append(-r_list[2][batch][pomo].item())
                self.perf_dict['reward4'].append(-r_list[3][batch][pomo].item())
        return self.get_scores(batch_size, aug_factor, reward)

    def _test_one_batch_comp_heuristic(self, batch_size, episode):
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
                state, reward, done, info_list, move_cost, r_list = self.env.step(selected)
                action += 1
            ## TODO-while done, call the self.env.middle_reset()
            reward_list = torch.cat((reward_list, reward[:, :, None]), dim=2)
            if period < self.env.periods:
                self.env.middle_reset(period)

        reward_cpu = reward.cpu().detach().numpy()
        reward_list = [r.cpu().detach().numpy() for r in r_list]
        for batch in range(selected.shape[0]):
            for pomo in range(selected.shape[1]):
                self.perf_dict['batch'].append(batch+episode)
                self.perf_dict['pomo'].append(pomo)
                self.perf_dict['obj'].append(-reward_cpu[batch][pomo])
                self.perf_dict['reward'].append(-reward_cpu[batch][pomo])
                self.perf_dict['status'].append(0)
                self.perf_dict['move_cost'].append(move_cost[batch][pomo].item())
                self.perf_dict['reward1'].append(-r_list[0][batch][pomo].item())
                self.perf_dict['reward2'].append(-r_list[1][batch][pomo].item())
                self.perf_dict['reward3'].append(-r_list[2][batch][pomo].item())
                self.perf_dict['reward4'].append(-r_list[3][batch][pomo].item())
        #return self.get_long_term_scores(batch_size, aug_factor, reward_list)
        return self.get_scores(batch_size, aug_factor, reward)

    def _test_one_batch_comp_heuristic(self, batch_size, episode):
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
                state, reward, done, info_list, move_cost, r_list = self.env.step(selected)
                action += 1
            ## TODO-while done, call the self.env.middle_reset()
            reward_list = torch.cat((reward_list, reward[:, :, None]), dim=2)
            if period < self.env.periods:
                self.env.middle_reset(period)

        reward_cpu = reward.cpu().detach().numpy()
        reward_list = [r.cpu().detach().numpy() for r in r_list]
        for batch in range(selected.shape[0]):
            for pomo in range(selected.shape[1]):
                self.perf_dict['batch'].append(batch+episode)
                self.perf_dict['pomo'].append(pomo)
                self.perf_dict['obj'].append(-reward_cpu[batch][pomo])
                self.perf_dict['reward'].append(-reward_cpu[batch][pomo])
                self.perf_dict['status'].append(0)
                self.perf_dict['move_cost'].append(move_cost[batch][pomo].item())
                self.perf_dict['reward1'].append(-r_list[0][batch][pomo].item())
                self.perf_dict['reward2'].append(-r_list[1][batch][pomo].item())
                self.perf_dict['reward3'].append(-r_list[2][batch][pomo].item())
                self.perf_dict['reward4'].append(-r_list[3][batch][pomo].item())
        #return self.get_long_term_scores(batch_size, aug_factor, reward_list)
        return self.get_scores(batch_size, aug_factor, reward)

    def _test_one_batch_mip(self, batch_size):
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
        #reward_list = torch.zeros(size=(batch_size, 0))
        # shape: (batch, 0~self.env.periods)
        state, reward, done = self.env.pre_step()
        # Initialize MIP solver
        ###############################################
        self.env_params['test_batch_size'] = batch_size
        mip_solver = GRBSolver(**self.env_params)
        reward = 0
        for period in range(1,self.env.periods+1):
            # The MIP solver solves a snapshot of the system at a time 
            # Update the environment if needed
            # It doesn't have to call the step function
            # It needs only the previous decision and
            # the demand and action limit for the next
            # period.
            # We need to record the metric returned
            # by the solver
            sol_dict = mip_solver.solve_whole()
            for batch in range(len(sol_dict['obj'])):
                self.perf_dict['obj'].append(sol_dict['obj'][batch])
                self.perf_dict['status'].append(sol_dict['status'][batch])
                self.perf_dict['move_cost'].append(sol_dict['move_cost'][batch])
                self.perf_dict['reward1'].append(sol_dict['reward1'][batch])
                self.perf_dict['reward2'].append(sol_dict['reward2'][batch])
                self.perf_dict['reward3'].append(sol_dict['reward3'][batch])
                self.perf_dict['reward4'].append(sol_dict['reward4'][batch])

                rew = (sol_dict['move_cost'][batch]+sol_dict['reward1'][batch]+sol_dict['reward2'][batch]+sol_dict['reward3'][batch]+sol_dict['reward4'][batch])
                reward += rew
                self.perf_dict['reward'].append(rew)
            if period < self.env.periods:
                self.env.middle_reset(period)

        #return self.get_scores(batch_size, aug_factor, reward_list)
        reward /= batch_size
        return reward, reward

    def get_scores(self, batch_size, aug_factor, reward):
        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        print(max_pomo_reward)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()

    def get_long_term_scores(self, batch_size, aug_factor, reward_list):
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
