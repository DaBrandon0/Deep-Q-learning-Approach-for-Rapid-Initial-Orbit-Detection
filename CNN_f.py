import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from scipy.stats import norm
import keras
from keras import layers
plt.rcParams['figure.dpi'] = 500
plt.rcParams.update({'font.size': 3})
from scipy import integrate
from numpy import random
import random
import gym
import numpy as np
import time
import os
import f_datastruct as buf
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import math
from keras import backend as K
import pickle

 
meanx = 0
varx = 0
meany = 0
vary = 0
savestart = np.array
rewards = np.array([])
average_rewards = np.array([])

# 0: Move left
# 1: Move down
# 2: Move right
# 3: Move up

def pdf(x,y):
     return ((1/(np.pi*2*varx*vary))*np.exp(-(1/2)*(((x-meanx)/varx)**2+((y-meany)/vary)**2)))
def prob(xlow,xhigh,ylow,yhigh):
      return integrate.nquad(pdf,[[xlow, xhigh],[ylow, yhigh]])

def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

def main():
  os.system('cls')
  plt.ion()
  plt.rcParams["figure.figsize"] = [7.00, 3.50]
  plt.rcParams["figure.autolayout"] = True
  remember = buf.RingBuf(1000000)
  model = frozolake_model(4)
  frames = 0
  while (frames <= 1000):
    env, state = initenv(mean_x=0, mean_y=1, variance_x=3, variance_y=3)
    frames = train(env=env,state=state,model=model,remember=remember, frames=frames)
    env, state = initenv(mean_x=1, mean_y=0, variance_x=3, variance_y=3)
    frames = train(env=env,state=state,model=model,remember=remember, frames=frames)
    env, state = initenv(mean_x=1, mean_y=1, variance_x=3, variance_y=3)
    frames = train(env=env,state=state,model=model,remember=remember, frames=frames)
  #afile = open(r'C:\Users\brand\Documents\GitHub\Rapid_IOD\GYM\past.pkl', 'wb')
  #pickle.dump(remember, afile)
  #afile.close()
  plt.plot(average_rewards)
  plt.ylabel('average reward')
  plt.ylabel('episode')
  plt.show()
  model.save("GYM\model.keras")

def train(state, env, model,remember, frames):
  global rewards
  global average_rewards
  steps = 0
  prevloc = 0
  global savestart 
  savestart = np.copy(state)
  done = 0
  graph = plt.imshow(state, cmap='gray',norm=LogNorm(vmin=0.0000000000000001, vmax=.01, clip= True))
  while (1 != done and steps != 200):
    prevloc, done, state = q_iteration(env,model,state,frames,remember,prevloc)
    graph.remove()
    graph = plt.imshow(state, cmap='gray',norm=LogNorm(vmin=0.0000000000000001, vmax=.01, clip= True))
    steps = steps + 1
    frames = frames + 1
  average_rewards = np.append(average_rewards,(np.sum(rewards)/np.size(rewards)))
  os.system('cls')
  print(average_rewards)
  print(f"\nframes thus far:", frames)
  rewards = np.array([])
  state = np.copy(savestart)
  env.reset()
  plt.clf()
  return frames

def initenv(mean_x,mean_y,variance_x,variance_y):
  global meanx 
  global meany
  global varx
  global vary
  meanx = mean_x
  varx = variance_x
  meany = mean_y
  vary = variance_y
  cell = ""
  for i in range (0,84):
    cell = cell + "F"
  map = []
  for i in range (0,84):
    map.append(cell)
  list1 = list(map[0])
  list1[0] = 'S'
  map[0] = ''.join(list1)
  list1 = list(map[meany])
  list1[meanx] = 'G'
  map[meany] = ''.join(list1)
  state = []
  row = []
  for j in range (84):
    for i in range(84):
      if j == 0 and i == 0:
        row.append(0)
      else:
        row.append(pdf(i,j))
    state.append(row)
    row = []
  state = np.array(state)
  state = (state-np.min(state))/(np.max(state)-np.min(state)) 
  env = gym.make("FrozenLake-v1", desc = map,  render_mode = "ansi", is_slippery = False,)
  env.reset()
  return env,state

def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal, size):
    """Do one deep Q learning iteration.
    
    Params:
    - model: The DQN
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal
    
    """
    
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q_values = model.predict([next_states, np.ones(actions.shape)],batch_size = size, verbose = 0)
    # The Q values of the terminal states is 0 by definition, so override them
    #next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    model.fit(
        x = [start_states, actions], y = actions * Q_values[:, None],
        epochs=1, batch_size=len(start_states), verbose = 0
    )


def frozolake_model(n_actions):
      init = keras.initializers.HeNormal(seed=1)
    # We assume a theano backend here, so the "channels" are first.
      forozolake_shape = (84, 84, 1)

    # With the functional API we need to define the inputs.
      frames_input = keras.Input(forozolake_shape, name='frames')
      actions_input = keras.Input((n_actions,), name='mask')

    # Previous implementation normalized input to [0,1] range, but our pdf matrix already has that range cuz proability laws.
      normalized = frames_input
    
    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
      conv_1 = layers.Conv2D(filters= 16, kernel_size= (8,8), strides= (4,4), activation='relu', kernel_initializer=init)(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
      conv_2 = layers.Conv2D(filters= 32, kernel_size= (4,4), strides=(2,2), activation='relu', kernel_initializer=init)(conv_1)
    # Flattening the second convolutional layer.
      conv_flattened = keras.layers.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
      hidden = keras.layers.Dense(256, activation='relu', kernel_initializer=init)(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
      output = keras.layers.Dense(n_actions, kernel_initializer=init)(hidden)
    # Finally, we multiply the output by the mask!
      filtered_output = keras.layers.Multiply()([output, actions_input])
      model = keras.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
      optimizer = optimizer=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
      model.compile(optimizer, loss= huber_loss)
      return model

def q_iteration(env, model, state, iteration, memory, prevloc):
    global rewards
    # Choose epsilon based on the iteration
    epsilon = epsilondecaying(iteration)
    # Choose the action 
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, state)

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    next_step = env.step(action)
    done = next_step[2]
    reward = next_step[1]
    newloc = next_step[0]
    if (reward == 0):
      reward = state[next_step[0]//84][next_step[0]%84]
      #reward = prob(next_step[0]//84,(next_step[0]//84)+1,next_step[0]%84,(next_step[0]%84)+1)[0]-1
      #math.log(prob(next_step[0]//84,(next_step[0]//84)+1,next_step[0]%84,(next_step[0]%84)+1)[0],10)
      new_state = np.copy(state)
      new_state[newloc//84][newloc%84] = -1
    else:
      new_state = np.copy(state)
      new_state[newloc//84][newloc%84] = 2
    rewards = np.append(rewards,[reward], axis=0)
    memory.append([state, action, reward, new_state, done])
    if (iteration%4 == 0):
      # Sample and fit every 4 steps
      number = math.log(memory.__len__(),1.2)
      if (number == 0):
          number = 1
      number = round(number)
      batch = memory.random(number)
      encodethis = np.array([item[1] for item in batch])
      encoded = np.empty((0,4), int)
      for item in encodethis:
        dummy = item
        item = []
        for i in range(4):
            if (dummy == i):
              item.append(1)
            else:
              item.append(0)
        encoded = np.append(encoded, np.array([item]), axis=0)
      fit_batch(model, .99, np.array([item[0] for item in batch]), encoded, np.array([item[2] for item in batch]), np.array([item[3] for item in batch]), np.array([item[4] for item in batch]), size=number)
    prevloc = next_step[0]
    return prevloc, next_step[2], new_state

def epsilondecaying (iteration):
     compare = [1-iteration*.001,0]
     return np.max(compare)

def choose_best_action(model, state):
     predall= np.array([[1,1,1,1]])
     state = np.array([state])
     results = model.predict([state, predall], batch_size = 1, verbose = 0)
     itemindex = np.where(results == np.max(results))
     return itemindex[1][random.randint(0,itemindex[1].size-1)]


def image_maker(value):
    state = []
    row = []
    for j in range (84):
        for i in range(84):
            row.append(value)
        state.append(row)
        row = []
    return np.array([state])

main()


