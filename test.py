import os
import plotly
from plotly.graph_objs import Scatter, Line
import torch
import __future__
from env import Env
from ODRL_template_matcher.obj_recognizor import TemplateMatcher
import cv2
#from modified_explain import *
import matplotlib
import matplotlib.pyplot as plt
import copy
#from newshiftedmap import shiftedColorMap
from copy import deepcopy
import torch.nn as nn
from scipy import ndimage
from filter import *
from segment_graph import *
from mainseg import *
import time
import gym
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy
from torch.autograd import Variable
####################################
import pickle
#from VERBAL.text_gen_tools import *
#from VERBAL.comp_text import *
#from VERBAL.square_attention import *

####################################
'''
def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad

def save(mask, img, blurred,value):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    cv2.imwrite("./saliency_outputs_perturb/img"+str(value)+".png", np.uint8(img))
    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    print(heatmap.shape)
    print(img.shape)
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    #perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    #cv2.imwrite("./saliency_outputs_perturb/perturbated"+str(value)+".png", np.uint8(255 * perturbated))
    #cv2.imwrite("./saliency_outputs_perturb/heatmap"+str(value)+".png", np.uint8(255 * heatmap))
    #cv2.imwrite("./saliency_output_perturb/mask"+str(value)+".png", np.uint8(255 * mask))
    cv2.imwrite("./saliency_outputs_perturb/cam"+str(value)+".png", np.uint8(255 * cam))
    return mask

def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v
'''
tm = TemplateMatcher('./ODRL_template_matcher/MsPacman-v0')
# Globals
Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10
my_dict = {}
# Test DQN
def test(args, T, dqn, val_mem, evaluate=False):
  global Ts, rewards, Qs, best_avg_reward
  env = Env(args)
  env.eval()
  Ts.append(T)
  T_rewards, T_Qs = [], []
  
  # Test performance over several episodes
  done = True
  ramitha_frame_number = 0
  for number in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False
      action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
      state, reward, done= env.step(action)  # Step
      if(number==0):
      ################################################
        if(action == dqn.act(state)):
          ramitha_frame_number += 1
          #Pixel Saliency :
          print(ramitha_frame_number)
          #dqn.pixel_saliency(state)
          #Guided backprop and Guided GradCAM
          #dqn.guided_backprop_and_CAM(state)
          RGBc_state = deepcopy(env.retRGB_state())
          RGBc_state = cv2.cvtColor(RGBc_state,cv2.COLOR_RGB2BGR)
          #plt.imsave('./saliency_outputs_objectsaliency/current_state'+str(ramitha_frame_number)+'.jpg',RGBc_state,cmap='gray')
          masked_state = {} 
          positions_listx = {}
          positions_listy = {}
          compset = []
          comp_set = []

          adv_obs_big = deepcopy(RGBc_state)
          
          sigma = 0.3
          k = 300
          min_val = 20
          '''
          sigma = 0.0
          k = 300
          min_val = 5
          '''
          adv_obs = adv_obs_big[45:185]
          #adv_obs = adv_obs_big[23:]
          plt.imsave('./saliency_outputs_objectsaliency/current_state'+str(ramitha_frame_number)+'.png',adv_obs,cmap='gray')
          #I change the jpg file into png——kunlun due to an error occur
          
          u,width,height = segment(adv_obs, sigma, k, min_val,999)
          compset = []
          object_saliency = np.zeros((210,160))
          for y in range(height):
                  for x in range(width):
                      comp = u.find(y * width + x)
                      compset.append(comp)
          final = np.array(list(set(compset)))
          for i in range(len(final)):
            current_comp = final[i]
            copy_state = deepcopy(adv_obs)
            masked_state = np.zeros((210,160,1))
            
            masked_state = masked_state[45:185]
            #I uncomment this for this game

            full_state = deepcopy(adv_obs_big)
            for y in range(height):
                    for x in range(width):
                        comp = u.find(y * width + x)
                        if(comp == current_comp):
                          copy_state[y, x,:] = 0 #Spoiled image
                          masked_state[y,x,0] = 255 # Maksed Image
            masked_state = masked_state.astype('uint8')

            if( np.count_nonzero(masked_state) <500):

              dst = cv2.inpaint(copy_state,masked_state,3,cv2.INPAINT_TELEA)

              full_state[45:185] = dst
              
              #full_state[23:] = dst
              full_state = np.array(full_state)
              for y in range(height):
                    for x in range(width):
                        comp = u.find(y * width + x)
                        if(comp == current_comp):
                          full_state_m = cv2.resize(cv2.cvtColor( full_state, cv2.COLOR_RGB2GRAY ), (84, 84), interpolation=cv2.INTER_LINEAR)
                          full_state_m = torch.tensor(full_state_m, dtype=torch.float32).div_(255)
                          object_saliency[y,x] = (dqn.evaluate_q(full_state_m.unsqueeze(0).expand(4,-1,-1)) - dqn.evaluate_q(state))
              #plt.imsave('./result/inpaint'+str(i)+'.png',full_state)
          plt.imsave('./saliency_outputs_objectsaliency/object_saliency'+str(ramitha_frame_number)+'.png',object_saliency,cmap='gray')
          '''
          tv_beta = 3
          learning_rate = 0.1
          l1_coeff = 0.01
          tv_coeff = 0.2
          max_iterations =  500
          original_img = deepcopy(RGBc_state)
          reduced_img = cv2.resize(original_img, (84, 84), interpolation=cv2.INTER_LINEAR)
          img = np.float32(original_img) / 255
          blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
          blurred_img2 = np.float32(cv2.medianBlur(original_img, 11)) / 255
          blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
          mask_init = np.ones((28, 28), dtype=np.float32)

          # Convert to torch variables
          img_m = cv2.resize(cv2.cvtColor( img, cv2.COLOR_RGB2GRAY ), (84, 84), interpolation=cv2.INTER_LINEAR)
          img_m = torch.tensor(img_m, dtype=torch.float32).div_(255)
          blurred_m = cv2.resize(cv2.cvtColor( blurred_img2, cv2.COLOR_RGB2GRAY ), (84, 84), interpolation=cv2.INTER_LINEAR)
          blurred_m = torch.tensor(blurred_m, dtype=torch.float32).div_(255)
          mask = numpy_to_torch(mask_init)

          
          upsample = torch.nn.UpsamplingBilinear2d(size=(84, 84))

          optimizer = torch.optim.Adam([mask], lr=learning_rate)

          #img_m = torch.mean(img_m, 1).unsqueeze(0)
          logp = nn.Softmax()(  ( dqn.q_full(img_m.unsqueeze(0).expand(4,-1,-1)) ) )
          category = np.argmax(logp.cpu().data.numpy())
          for i in range(max_iterations):
              upsampled_mask = upsample(mask)
              upsampled_mask = \
                  upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                        upsampled_mask.size(3))

              # Use the mask to perturbated the input image.
              perturbated_input = img_m.mul(upsampled_mask) + \
                                  blurred_m.mul(1 - upsampled_mask)

              noise = np.zeros((84, 84, 3), dtype=np.float32)
              noise = noise + cv2.randn(noise, 0, 0.2)
              
              noise = numpy_to_torch(noise)
              perturbated_input = perturbated_input + noise
              pi = np.squeeze(perturbated_input.detach().numpy())
              pi = np.transpose(pi, (1, 2, 0))
              pi = cv2.resize(cv2.cvtColor( pi, cv2.COLOR_RGB2GRAY ), (84, 84), interpolation=cv2.INTER_LINEAR)
              pi = torch.tensor(pi, dtype=torch.float32).div_(255)

              #perturbated_m = cv2.resize(cv2.cvtColor( perturbated_input, cv2.COLOR_RGB2GRAY ), (84, 84), interpolation=cv2.INTER_LINEAR)
              #perturbated_m = torch.tensor(perturbated_m, dtype=torch.float32).div_(255)
          
              #perturbated_input = torch.mean(perturbated_input, 1).unsqueeze(0)
              optimizer.zero_grad()

              logp = nn.Softmax()((dqn.q_full(pi.unsqueeze(0).expand(4,-1,-1))))
              loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
                     tv_coeff * tv_norm(mask, tv_beta) + logp[0, category]

              loss.backward()
              optimizer.step()

              # Optional: clamping seems to give better results
              mask.data.clamp_(0, 1)

          upsampled_mask = upsample(mask)
          mm = save(upsampled_mask, reduced_img, blurred_img_numpy,ramitha_frame_number)
          '''


      reward_sum += reward
      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()
  # Test Q-values over validation memory
  for state in val_mem:  # Iterate over valid states
    T_Qs.append(dqn.evaluate_q(state))

  avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
  if not evaluate:
    # Append to results
    rewards.append(T_rewards)
    Qs.append(T_Qs)

    # Plot
    _plot_line(Ts, rewards, 'Reward', path='results')
    _plot_line(Ts, Qs, 'Q', path='results')

    # Save model parameters if improved
    if avg_reward > best_avg_reward:
      best_avg_reward = avg_reward
      dqn.save('results')

  # Return average reward and Q-value
  return avg_reward, avg_Q


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color='transparent'), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
####################################
