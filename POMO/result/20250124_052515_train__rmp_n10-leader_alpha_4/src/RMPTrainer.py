import torch
import pandas as pd
from logging import getLogger
from RMPEnv import RMPEnv as Env
from RMPModel import RMPModel as Model
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from utils.utils import *

class RMPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # initialize data logger
        self.reset_problem_dict()
        self.reset_step_dict()

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def reset_problem_dict(self):
        num_rack_types = self.env_params['config'].num_rack_types
        num_scopes = self.env_params['config'].num_scopes
        num_resources = len(self.env_params['config'].resource_weights)
        self.problem_dict = {}
        self.problem_dict['epoch'] = []
        self.problem_dict['batch'] = []
        self.problem_dict['reward_pomo'] = []
        self.problem_dict['reward_1_pomo'] = []
        self.problem_dict['reward_2_pomo'] = []
        self.problem_dict['reward_3_pomo'] = []
        self.problem_dict['reward_4_pomo'] = []
        self.problem_dict['obj_1_changes'] = []
        self.problem_dict['obj_2_changes'] = []
        self.problem_dict['obj_3_changes'] = []
        self.problem_dict['obj_4_changes'] = []
        self.problem_dict['selected_rts'] = []
        self.problem_dict['move_cost'] = []
        self.problem_dict['demand'] = []
        self.problem_dict['action_limit'] = []
        self.problem_dict['scope_rt'] = []
        self.problem_dict['scope_res'] = []

    def reset_step_dict(self):
        num_rack_types = self.env_params['config'].num_rack_types
        num_scopes = self.env_params['config'].num_scopes
        # lvi_sp represents the softplus of resources in the level i scopes
        self.step_dict = {'epoch': [], 'batch': [], 'pomo': [], 'action': [], 'obj_list': [],\
                          'moves': [], 'steps': [], 'scope_rt': []}

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        num_rack_types = self.env_params['config'].num_rack_types
        num_scopes = self.env_params['config'].num_scopes
        num_resources = len(self.env_params['config'].resource_weights)
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # avg_score, avg_loss = self._train_one_batch(batch_size, episode, epoch)
            avg_score, avg_loss, reward, move_cost, reward_list, obj_list, moves_list, steps_list =\
                self._train_one_batch_offline(batch_size, episode, epoch)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            X_init = self.env.prev_pos_rack_map.repeat(1, 1, self.env.pomo_size).permute(0, 2, 1)
            X_init = self.env.one_hot_encode(X_init, num_rack_types, True)
            SXR = torch.einsum('ij,abjk,kc->abic', self.env.scopes_comp, X_init, self.env.resource_table)
            SXR = SXR.cpu().detach().numpy()
            SX = torch.einsum('ij,abjk->abik', self.env.scopes_comp, X_init)
            SX = SX.cpu().detach().numpy()

            demand = self.env.demand.cpu().detach().numpy()
            action_limit = self.env.action_limit.cpu().detach().numpy()
            selected_rts = self.env.selected_node_list.cpu().detach().numpy()

            for batch in range(batch_size):
                self.problem_dict['epoch'].append(epoch)
                self.problem_dict['batch'].append(batch+episode)
                self.problem_dict['reward_pomo'].append(reward[batch].tolist())
                self.problem_dict['reward_1_pomo'].append(reward_list[0][batch].tolist())
                self.problem_dict['reward_2_pomo'].append(reward_list[1][batch].tolist())
                self.problem_dict['reward_3_pomo'].append(reward_list[2][batch].tolist())
                self.problem_dict['reward_4_pomo'].append(reward_list[3][batch].tolist())
                self.problem_dict['selected_rts'].append(selected_rts[batch].tolist())
                self.problem_dict['demand'].append(demand[batch].tolist())
                self.problem_dict['action_limit'].append(action_limit[batch].tolist())
                self.problem_dict['move_cost'].append(move_cost[batch].tolist())
                self.problem_dict['scope_res'].append(SXR[batch, 0].tolist())
                self.problem_dict['scope_rt'].append(SX[batch, 0].tolist())
                self.problem_dict['obj_1_changes'].append(obj_list[batch,:,:,0].tolist())
                self.problem_dict['obj_2_changes'].append(obj_list[batch,:,:,1].tolist())
                self.problem_dict['obj_3_changes'].append(obj_list[batch,:,:,2].tolist())
                self.problem_dict['obj_4_changes'].append(obj_list[batch,:,:,3].tolist())
                #for rt in range(num_rack_types):
                #    # reward for different pomo id
                #    self.problem_dict[f'reward_pomo_{rt}'].append(reward[batch, rt])
                #    self.problem_dict[f'demand_{rt}'].append(demand[batch, rt])
                #    self.problem_dict[f'act_limit_{rt}'].append(action_limit[batch, rt])
                #    self.problem_dict[f'move_cost_{rt}'].append(move_cost[batch, rt])
                #for sc in range(num_scopes):
                #    for res in range(num_resources):
                #        self.problem_dict[f'sc_{sc}_res_{res}'].append(SXR[batch, 0, sc, res])


            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        if epoch%2==0:
            prob_df = pd.DataFrame(self.problem_dict)
            prob_df.to_csv(f'{self.result_folder}/probs_in_epoch_{epoch}.csv')
            self.reset_problem_dict()
        step_df = pd.DataFrame(self.step_dict)
        step_df.to_csv(f'{self.result_folder}/steps_in_epoch_{epoch}.csv')
        self.reset_step_dict()
        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, episode, epoch):
        ## TODO-Revise this to online version
        # Method: run the step function for num_rack_types*num_large_steps
        # For every num_rack_types running of step functions, update the 
        # environment.
        ## TODO-Specify what to update for every num_rack_types running of the step function
        ## TODO-Create the critic network for predicting the long term reward
        # Collect the reward for every num_rack_types running of the step function 
        # and use the rewards to train the critic network

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem*self.env.periods)
        reward_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~self.env.periods)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        for period in range(1,self.env.periods+1):
            self.logger.info(f'POMO Rollout for period {period}')
            #self.logger.info(f'POMO pre_step state is {state}')
            while not done:
                selected, prob = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            ## TODO-while done, call the self.env.middle_reset()
            reward_list = torch.cat((reward_list, reward[:, :, None]), dim=2)
            #self.logger.info(f'POMO state after one period finishes is {state}')
            if period < self.env.periods:
                # The rack movement process is not done
                self.env.middle_reset(period)
                self.model.pre_forward(reset_state)
                state, reward, done = self.env.pre_step()
                done = False
        # Loss
        ###############################################
        ##  TODO-For online implementation, the advantage should be updated:
        #   A. the reward here should be replaced by the long-term rewards
        #   B. the baseline reward should be replaced by:
        #       (1) long term reward estimate of the critic network
        #       (2) Average of the long-term rewards
        ##  TODO-Check if the logprob has to be revised
        reward = self.gen_long_term_rew(batch_size, reward_list)
        # shape: (batch, pomo, periods)
        
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo, periods)
        # log_prob = prob_list.log().sum(dim=2)
        log_prob = self.reshape_and_sum(prob_list.log(), self.env.problem_size)
        # size = (batch, pomo, periods)
        loss = -(advantage * log_prob)  # Minus Sign: To Increase REWARD
        # size = (batch, pomo, periods)
        loss = loss.sum(dim=2)
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward[:,:,0].max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()
    
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

    def reshape_and_sum(self, S, D):
        # Get the shape of the input tensor
        A, B, CD = S.shape
        
        # Reshape S to split the last dimension into D chunks
        S_reshaped = S.view(A, B, -1, D)
        
        # Sum along the last dimension of the reshaped tensor
        T = S_reshaped.sum(dim=-1)
        
        return T

    def _train_one_batch_offline(self, batch_size, episode, epoch):
        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem*self.env.periods)
        obj_list = [[[] for _ in range(self.env.pomo_size)] for _ in range(batch_size)]
        moves_list = [[0 for _ in range(self.env.pomo_size)] for _ in range(batch_size)]
        steps_list = [[0 for _ in range(self.env.pomo_size)] for _ in range(batch_size)]

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        scopes_comp = self.env.scopes_comp.cpu().detach().numpy()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done, info_list, move_cost, reward_list = self.env.step(selected)
            selected_info = selected.cpu().detach().numpy()
            for batch in range(selected.shape[0]):
                for pomo in range(selected.shape[1]):
                    X = info_list[batch][pomo]['mapping']
                    SX = scopes_comp@X
                    self.step_dict['epoch'].append(epoch)
                    self.step_dict['batch'].append(episode+batch)
                    self.step_dict['pomo'].append(pomo)
                    self.step_dict['action'].append(selected_info[batch, pomo])
                    self.step_dict['moves'].append(info_list[batch][pomo]['moves'])
                    self.step_dict['steps'].append(info_list[batch][pomo]['steps'])
                    self.step_dict['scope_rt'].append(SX.tolist())
                    self.step_dict['obj_list'].append(info_list[batch][pomo]['obj_list'])
                    obj_list[batch][pomo].append(info_list[batch][pomo]['obj_list'])
                    moves_list[batch][pomo]+=info_list[batch][pomo]['moves']
                    steps_list[batch][pomo]+=info_list[batch][pomo]['steps']
                    #for sc in range(self.env.num_scopes):
                    #    for rt in range(self.env.num_rack_types):
                    #        self.step_dict[f'sc_{sc}_rt_{rt}'].append(info_list[batch][pomo]['mapping'][sc, rt])
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        # Leader reward scale
        max_pomo_reward, indices = torch.max(reward, dim=1) # get best results from pomo
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        original_advantage = advantage.clone()
        # shape: (batch, pomo)
        # Leader reward scale
        if self.trainer_params['leader_reward']:
            for i in range(reward.shape[0]):
                advantage[i, indices[i]] *= self.trainer_params['leader_reward_alpha']
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -(advantage * log_prob)  # Minus Sign: To Increase REWARD
        # size = (batch, pomo)
        original_loss = -(original_advantage * log_prob)
        loss_mean = loss.mean()
        original_loss_mean = original_loss.mean()

        # Score
        ###############################################
        self.logger.info(f"[One batch] Max reward - {max_pomo_reward}")
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
        reward_info = reward.cpu().detach().numpy()
        reward_list = [r.cpu().detach().numpy() for r in reward_list]

        obj_list = np.array(obj_list)
        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        #return score_mean.item(), loss_mean.item()
        return score_mean.item(), original_loss_mean.item(), reward_info, move_cost, reward_list, obj_list, moves_list, steps_list
