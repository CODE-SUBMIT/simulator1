from net.iccv1.stock.dqn_algo_scale import *
root_path = "/ai/51/dixinhan/layout3/bedroom1/srl1_scale_r3/"
json_path = "/ai/51/dixinhan/layout3/bedroom1/27705_4.json"

#1.1
dqn = DQNAlgo(filepath=json_path, netname='conv2d', rootpath = root_path, algoname='duel')
#1.2
rewards11, total_rewards11 = dqn.train(max_iteration=1e5, save_pth='/ai/51/dixinhan/demo2/net/iccv1/checkpoints/single4/model.pth')
#1.3
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
plt.plot(list(range(len(rewards11))), rewards11)
plt.xlabel('iteration')
plt.ylabel('reward')
fig.add_subplot(1,2,2)
plt.plot(list(range(len(rewards11))), total_rewards11[1:])
plt.xlabel('iteration')
plt.ylabel('total reward')
plt.show()

#2.1
valid_dqn = DQNAlgo(filepath='./data/16.csv', netname='fc', algoname='duel')
#2.2
rewards12, total_rewards12 = valid_dqn.valid(max_iteration=1e4, load_pth='./modelduelfc.pth')
#2.3
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
plt.plot(list(range(len(rewards12))), rewards12)
plt.xlabel('iteration')
plt.ylabel('reward')
fig.add_subplot(1,2,2)
plt.plot(list(range(len(rewards12))), total_rewards12[1:])
plt.xlabel('iteration')
plt.ylabel('total reward')
plt.show()

#%%

dqn2 = DQNAlgo(filepath='./data/15.csv', netname='fc', algoname='')

#%%

rewards21, total_rewards21 = dqn2.train(max_iteration=1e6, save_pth='./modelfc.pth')

#%%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
plt.plot(list(range(len(rewards21))), rewards21)
plt.xlabel('iteration')
plt.ylabel('reward')
fig.add_subplot(1,2,2)
plt.plot(list(range(len(rewards21))), total_rewards21[1:])
plt.xlabel('iteration')
plt.ylabel('total reward')
plt.show()

#%%

valid_dqn2 = DQNAlgo(filepath='./data/16.csv', netname='fc', algoname='')

#%%

rewards22, total_rewards22 = valid_dqn2.valid(max_iteration=1e6, load_pth='./modelfc.pth')

#%%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
plt.plot(list(range(len(rewards22))), rewards22)
plt.xlabel('iteration')
plt.ylabel('reward')
fig.add_subplot(1,2,2)
plt.plot(list(range(len(rewards22))), total_rewards22[1:])
plt.xlabel('iteration')
plt.ylabel('total reward')
plt.show()

#%%

dqn3 = DQNAlgo(filepath='./data/15.csv', netname='conv1d', algoname='duel')

#%%

rewards31, total_rewards31 = dqn3.train(max_iteration=1e6, save_pth='./modelduelconv.pth')

#%%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
plt.plot(list(range(len(rewards31))), rewards31)
plt.xlabel('iteration')
plt.ylabel('reward')
fig.add_subplot(1,2,2)
plt.plot(list(range(len(rewards31))), total_rewards31[1:])
plt.xlabel('iteration')
plt.ylabel('total reward')
plt.show()

#%%

valid_dqn3 = DQNAlgo(filepath='./data/16.csv', netname='conv1d', algoname='duel')

#%%

rewards32, total_rewards32 = valid_dqn3.valid(max_iteration=1e6, load_pth='./modelduelconv.pth')

#%%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
plt.plot(list(range(len(rewards32))), rewards32)
plt.xlabel('iteration')
plt.ylabel('reward')
fig.add_subplot(1,2,2)
plt.plot(list(range(len(rewards32))), total_rewards32[1:])
plt.xlabel('iteration')
plt.ylabel('total reward')
plt.show()

#%%

dqn4 = DQNAlgo(filepath='./data/15.csv', netname='conv1d', algoname='')

#%%

rewards41, total_rewards41 = dqn4.train(max_iteration=1e6, save_pth='./modelconv.pth')

#%%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
plt.plot(list(range(len(rewards41))), rewards41)
plt.xlabel('iteration')
plt.ylabel('reward')
fig.add_subplot(1,2,2)
plt.plot(list(range(len(rewards41))), total_rewards41[1:])
plt.xlabel('iteration')
plt.ylabel('total reward')
plt.show()

#%%

valid_dqn4 = DQNAlgo(filepath='./data/16.csv', netname='conv1d', algoname='')

#%%

rewards42, total_rewards42 = valid_dqn4.valid(max_iteration=1e6, load_pth='./modelconv.pth')

#%%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
plt.plot(list(range(len(rewards42))), rewards42)
plt.xlabel('iteration')
plt.ylabel('reward')
fig.add_subplot(1,2,2)
plt.plot(list(range(len(rewards42))), total_rewards42[1:])
plt.xlabel('iteration')
plt.ylabel('total reward')
plt.show()

#%%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
plt.plot(list(range(len(rewards12))), rewards12, label='dueling dqn + fc')
plt.plot(list(range(len(rewards12))), rewards22, label='dqn + fc')
plt.plot(list(range(len(rewards12))), rewards32, label='dueling dqn + conv')
plt.plot(list(range(len(rewards12))), rewards42, label='dqn + conv')
plt.xlabel('validation iteration')
plt.ylabel('reward')
fig.add_subplot(1,2,2)
plt.plot(list(range(len(rewards12))), total_rewards12[1:], label='dueling dqn + fc')
plt.plot(list(range(len(rewards12))), total_rewards22[1:], label='dqn + fc')
plt.plot(list(range(len(rewards12))), total_rewards32[1:], label='dueling dqn + conv')
plt.plot(list(range(len(rewards12))), total_rewards42[1:], label='dqn + conv')
plt.xlabel('validation iteration')
plt.ylabel('total reward')
plt.legend()
plt.show()