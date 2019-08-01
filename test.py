"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
from gym import utils
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as F
import time
import random

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    return args


def test(opt):
    torch.manual_seed(123)
    if opt.output_path != None:
        env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,
                                                    "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage))
    else:
        env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,None)
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    max_x_pos = 0
    max_x_pos_counter = 0
    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            print('done')
            max_x_pos = 0
            max_x_pos_counter = 0
            env.reset()
            done = False
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        #print(reward)
        env.render()
        state = torch.from_numpy(state)
        if max_x_pos_counter < 50:
            time.sleep(0.06)
        if info['x_pos'] > max_x_pos:
            max_x_pos = info['x_pos']
            max_x_pos_counter = 0
        else:
            max_x_pos_counter += 1
        if max_x_pos_counter > 150:
            if info['x_pos'] < max_x_pos_counter:
                print('must be dancing',info['x_pos'],max_x_pos_counter)
                max_x_pos_counter = 0
            else:
                print('no progress, stopping')
                done = True
        
        if info["flag_get"]:
            print("World {} stage {} completed".format(opt.world, opt.stage))
            done = True
    print('done testing')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
if __name__ == "__main__":
    opt = get_args()
    test(opt)
