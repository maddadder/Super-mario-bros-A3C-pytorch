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
from gym.envs.classic_control import rendering
import pyglet
from shutil import copyfile
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

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
    viewer = rendering.SimpleImageViewer()
    viewer.width = 800 * 2
    viewer.height = 600 * 2
    #1920x1080
    viewer.window = pyglet.window.Window(width=viewer.width, height=viewer.height, resizable=True)
    
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
    is_level_specific_model_loaded = False
    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            print('done')
            max_x_pos = 0
            max_x_pos_counter = 0
            env.reset()
            if torch.cuda.is_available():
                model_file_name = "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, env.world + 1, env.stage + 1)
                if os.path.isfile(model_file_name):
                    model.load_state_dict(torch.load(model_file_name))
                    model.cuda()
                    is_level_specific_model_loaded = True
                else:
                    is_level_specific_model_loaded = False
            else:
                model_file_name = "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, env.world + 1, env.stage + 1)
                if os.path.isfile(model_file_name):
                    model.load_state_dict(torch.load(model_file_name))
                    is_level_specific_model_loaded = True
                else:
                    is_level_specific_model_loaded = False
                model.load_state_dict(torch.load(model_file_name,
                                                map_location=lambda storage, loc: storage))
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
        rgb = env.render('rgb_array')
        state = torch.from_numpy(state)
        
        viewer.imshow(rgb)
        if max_x_pos_counter < 50:
            time.sleep(0.06)
        if reward < 0:
            max_x_pos_counter += 1
        if max_x_pos_counter > 150:
            print('no progress, stopping')
            done = True
        
        if info["flag_get"]:
            print("World {} stage {} completed".format(opt.world, opt.stage))
            done = True
            #copyfile("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage), "{}/a3c_super_mario_bros_{}_{}_{}".format(opt.saved_path, info["world"], info["stage"],random.random()))
        print(is_level_specific_model_loaded,reward,COMPLEX_MOVEMENT[action])
    print('done testing')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
if __name__ == "__main__":
    opt = get_args()
    test(opt)
