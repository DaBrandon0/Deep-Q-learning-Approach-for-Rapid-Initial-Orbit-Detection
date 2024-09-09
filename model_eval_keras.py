from gym.envs.toy_text.frozen_lake import generate_random_map
import gym
import numpy as np
import keras
import os
import time
from scipy import integrate

meanx = 1
varx = 3
meany = 0
vary = 3

def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = keras.ops.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

def pdf(x,y):
     return ((1/(np.pi*2*varx*vary))*np.exp(-(1/2)*(((x-meanx)/varx)**2+((y-meany)/vary)**2)))

def prob(xlow,xhigh,ylow,yhigh):
      return integrate.nquad(pdf,[[xlow, xhigh],[ylow, yhigh]])

def image_maker(value):
    state = []
    row = []
    for j in range (84):
        for i in range(84):
            row.append(value)
        state.append(row)
        row = []
    return np.array([state])

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
env = gym.make("FrozenLake-v1", desc = map, render_mode = "ansi", is_slippery = False)
before = env.reset()
model = keras.models.load_model('GYM\model.keras', custom_objects={'huber_loss': huber_loss})
for layer in model.layers: print(layer.get_config(), layer.get_weights())

state = []
row = []
for j in range (84):
    for i in range(84):
        row.append(pdf(i,j))
    state.append(row)
    row = []
prevloc = 0
done = 0
state = np.array(state)
predall= np.array([[1,1,1,1]])
state = np.array([state])
state = (state-np.min(state))/(np.max(state)-np.min(state)) 
state[0][0][0] = 0
while (1 != done):
    results = model.predict([state, predall], batch_size = 1)
    itemindex = np.where(results == np.max(results))
    after = env.step(itemindex[1][0])
    done = after[2]
    state[0][after[0]//84][after[0]%84]=0
    before = after
    print(env.render())
    time.sleep(.1)
    os.system('cls||clear')
    