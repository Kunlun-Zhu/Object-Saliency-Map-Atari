import argparse
#import go_vncdriver
import tensorflow as tf
from envs import create_env
import subprocess as sp
import util
import model
import numpy as np
from worker import new_env
import cv2
import matplotlib
#matplotlib.use('agg') 
import matplotlib.pyplot as plt
from ODRL_template_matcher.obj_recognizor import TemplateMatcher
import joblib
from scipy.misc import imresize
from PIL import Image
from tqdm import tqdm
from random import randint
import os
import pickle
from main_modified import *
from copy import deepcopy
parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-gpu', '--gpu', default=0, type=int, help='Number of GPUs')
parser.add_argument('-r', '--remotes', default=None,help='The address of pre-existing VNC servers and rewarders to use'
                    '(e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-e', '--env-id', type=str, default="maze",help="Environment id")
parser.add_argument('-a', '--alg', type=str, default="VPN", help="Algorithm: [A3C | Q | VPN]")
parser.add_argument('-mo', '--model', type=str, default="CNN", help="Name of model: [CNN | LSTM]")
parser.add_argument('-ck', '--checkpoint', type=str, default="", help="Path of the checkpoint")
parser.add_argument('-n', '--n-play', type=int, default=1000, help="Num of play")
parser.add_argument('--eps', type=float, default=0.0, help="Epsilon-greedy")
parser.add_argument('--config', type=str, default="", help="config xml file for environment")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
# Hyperparameters
parser.add_argument('-g', '--gamma', type=float, default=0.98, help="Discount factor")
parser.add_argument('--dim', type=int, default=64, help="Number of final hidden units")
parser.add_argument('--f-num', type=str, default='32,32,64', help="num of conv filters")
parser.add_argument('--f-stride', type=str, default='1,1,2', help="stride of conv filters")
parser.add_argument('--f-size', type=str, default='3,3,4', help="size of conv filters")
parser.add_argument('--f-pad', type=str, default='SAME', help="padding of conv filters")
# VPN parameters
parser.add_argument('--branch', type=str, default="4,4,4", help="branching factor")
# 263 abc-1
# 1300

def _process_frame84gray(frame):
    frame = cv2.resize(frame, (84, 84))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [84, 84, 1])
    return frame


max_iterations = 218#650
extratag = 'final_5'
start_val = 217#51
folder = 'ep-final-uncut-full-c'
folderx = 'episode_object_saliency_redo'
os.mkdir('./obj_saliency/'+folder)
tm = TemplateMatcher('./ODRL_template_matcher/MsPacman-v0')
#read_state = cv2.imread('newimg.png')
store_y = []
def evaluate(env, agent, num_play=1, eps=0.0):
    for number_runs in range(start_val,max_iterations):
        read_state = cv2.imread('./obj_saliency/'+folderx+'/'+str(number_runs)+'/image.png')
        actual_state =  cv2.imread('./obj_saliency/'+folderx+'/'+str(number_runs)+'/image.png')
        read_state = cv2.cvtColor(read_state,cv2.COLOR_BGR2RGB)
        last_state = _process_frame84gray(read_state)
        last_features = agent.get_initial_features()
        last_meta = None
        store_y = agent
        [fetched,q] = agent.act(last_state, last_features,meta=last_meta)
        #print(agent.grad_finder(last_state, last_features,meta=last_meta))
        r0,r1,r2,_,_,_ = agent.reward_val(last_state, last_features,meta=last_meta)
        action, features = fetched[0], fetched[1:]      
        
        extracted_objects = tm.match_all_objects(read_state)
        
        object_saliency = np.zeros((210, 160))
        reward_saliency1 = np.zeros((210, 160))
        reward_saliency2 = np.zeros((210, 160))
        reward_saliency3 = np.zeros((210, 160))
        newq_saliency= np.zeros((210, 160))
        newq_saliency1= np.zeros((210, 160))
        
        action_mat = ['noop','up','right','left','down','upright','upleft', 'downright','downleft']
        
        pixelmap = agent.pixel_saliency_map(last_state, last_features,meta=last_meta)
        q_sort = q.argsort()
        print(action_mat[q_sort[-1]] + ' ' + action_mat[pixelmap[2]] + ' ' + action_mat[pixelmap[3]] )
        a = open('./obj_saliency/'+folder+'/scores'+str(number_runs)+'.txt', 'w')
        a.write(action_mat[q_sort[-1]] + ' ' + action_mat[pixelmap[2]] + ' ' + action_mat[pixelmap[3]])
        a.close()
        val1 =  135
        val2 =  29
        val3 =  4
        for obj, locs in tqdm(extracted_objects.items()):
            for loc in tqdm(locs):
                masked_state = np.array(deepcopy(read_state))
                masked_state[loc.up: loc.down, loc.left: loc.right, 0] = val1
                masked_state[loc.up: loc.down, loc.left: loc.right, 1] = val2
                masked_state[loc.up: loc.down, loc.left: loc.right, 2] = val3
                cv2.imwrite('obj_saliency/'+folder+'/'+str(number_runs)+'/masked_state'+str(loc)+'.png',masked_state)
                #masked_state[loc.up: loc.down, loc.left: loc.right, :] = 57
                last_masked_state = _process_frame84gray(masked_state)
                
                [_,qq] = agent.act(last_masked_state, last_features,meta=last_meta)
                pixelmapp = agent.pixel_saliency_map(last_masked_state, last_features,
                            meta=last_meta)
                
                rr0,rr1,rr2,_,_,_ = agent.reward_val(last_masked_state, last_features,meta=last_meta)
                #object_saliency[loc.up: loc.down, loc.left: loc.right] = (qq.max() - q.max())
                #reward_saliency1[loc.up: loc.down, loc.left: loc.right] = (rr0 - r0)
                object_saliency[loc.up: loc.down, loc.left: loc.right] = -(q.max() - qq.max())
                reward_saliency1[loc.up: loc.down, loc.left: loc.right] = (r0 - rr0)
                reward_saliency2[loc.up: loc.down, loc.left: loc.right] = (rr1 - r1)
                reward_saliency3[loc.up: loc.down, loc.left: loc.right] = (rr2 - r2)
                newq_saliency[loc.up: loc.down, loc.left: loc.right] = (pixelmapp[0] - pixelmap[0])
                newq_saliency1[loc.up: loc.down, loc.left: loc.right] = (pixelmapp[1] - pixelmap[1])
        
        object_saliency = (object_saliency - np.min(object_saliency))
        reward_saliency1 = (reward_saliency1 - np.min(reward_saliency1))
        reward_saliency2 = (reward_saliency2 - np.min(reward_saliency2))
        reward_saliency3 = (reward_saliency3 - np.min(reward_saliency3))
        newq_saliency = (newq_saliency - np.min(newq_saliency.mean)) 
        newq_saliency1 = (newq_saliency1 - np.min(newq_saliency1))
	
        sav_dif1_sal = object_saliency-newq_saliency
        sav_dif2_sal = newq_saliency - newq_saliency1

        
        #os.mkdir('./obj_saliency/'+folder+'/'+str(number_runs))
        newS = cv2.cvtColor(read_state,cv2.COLOR_BGR2RGB)
        plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'read_state.png',newS[:171])
        
        filename = '/home/ramitha/final/bayesian_net/extract/f'+str(number_runs)
        with open(filename+'.pkl','wb') as f:
            pickle.dump([extracted_objects,object_saliency[:171],actual_state[:171],actual_state,action,object_saliency],f)
      
        fig, ax = plt.subplots()
        _val, max_val = np.min(sav_dif1_sal), np.max(sav_dif2_sal)
        #intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
        ax.matshow(sav_dif1_sal, cmap=plt.cm.Blues)
        plt.show()
        for i in xrange(160):
            for j in xrange(171):
                c = objsal[j,i]
                ax.text(i, j, str(c), va='center', ha='center')

        plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'objectsaliency[:171]'+extratag+'.png',object_saliency[:171],cmap='gray')
        #plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'reduced-reward-saliency-1'+extratag+'.png',reduced_saliency,cmap='gray')
        
        #plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'rewardsaliency1'+extratag+'.png',sav_rew_sal1,cmap='gray')
        #plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'rewardsaliency2'+extratag+'.png',sav_rew_sal2,cmap='gray')
        #plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'rewardsaliency3'+extratag+'.png',sav_rew_sal3,cmap='gray')
        
        plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'newqsaliency'+extratag+'.png',newq_saliency[:171],cmap='gray')
        plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'newqsaliency1'+extratag+'.png',newq_saliency1[:171],cmap='gray')
        plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'diff1'+extratag+'.png',sav_dif1_sal[:171],cmap='gray')
        plt.imsave('obj_saliency/'+folder+'/'+str(number_runs)+'diff2'+extratag+'.png',sav_dif2_sal[:171],cmap='gray')
        
def run():
    args = parser.parse_args()
    args.task = 0
    args.f_num = util.parse_to_num(args.f_num)
    args.f_stride = util.parse_to_num(args.f_stride)
    args.f_size = util.parse_to_num(args.f_size)
    args.branch = util.parse_to_num(args.branch)

    env = new_env(args)
    args.meta_dim = 0
    device = '/gpu:0' if args.gpu > 0 else '/cpu:0'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    config = tf.ConfigProto(device_filters=device, gpu_options=gpu_options,allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        model_type = 'vpn'
        with tf.device(device):
            with tf.variable_scope("local/learner"):
                agent = eval("model." + args.model)(env.observation_space.shape, 
                    env.action_space.n, type=model_type, 
                    gamma=args.gamma, 
                    dim=args.dim,
                    f_num=args.f_num,
                    f_stride=args.f_stride,
                    f_size=args.f_size,
                    f_pad=args.f_pad,
                    branch=args.branch,
                    meta_dim=args.meta_dim)
                print("Num parameters: %d" % agent.num_param)
        
            saver = tf.train.Saver()
            saver.restore(sess, args.checkpoint)
        np.random.seed(args.seed)
        evaluate(env, agent, args.n_play, eps=args.eps)

if __name__ == "__main__":
    run()
    
    
    
