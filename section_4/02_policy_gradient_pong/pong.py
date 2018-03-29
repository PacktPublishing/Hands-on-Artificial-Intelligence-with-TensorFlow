# from __future__ import print_function
import numpy as np
import cPickle as pickle
import gym

# Hyperparameters
H = 200                 # Number of hidden layer neurons
batch_size = 10         # Every how many episodes to do a param update?
gamma = 0.99            # Discount factor for reward
decay_rate = 0.99       # Decay factor for RMSProp leaky sum of grad^2
learning_rate = 1e-4

render = True
resume = False
resume_checkpoint = 100

# Model initialization
D = 80 * 80 # Input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save-'+str(resume_checkpoint)+'.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k : np.zeros_like(v) for k,v in model.iteritems()}   # update buffers that add up gradients over a batch
rmsprop_cache = {k : np.zeros_like(v) for k,v in model.iteritems()} # rmsprop memory

def sigmoid(x):
    """ Sigmoid "squashing" function to interval [0,1] """
    return 1.0 / (1.0 + np.exp(-x))

def preprocess(I):
    """ Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]       # Crop the actual playarea
    I = I[::2, ::2, 0]  # Downsample by factor of 2
    I[I == 144] = 0     # erase background (background type 1)
    I[I == 109] = 0     # erase background (background type 2)
    I[I != 0] = 1       # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """ Take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0 # Reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    """ Forward pass """
    h = np.dot(model['W1'], x)      # Compute hidden layer neuron activations
    h[h < 0] = 0                    # ReLU nonlinearity
    logp = np.dot(model['W2'], h)   # Compute log probability of going up
    p = sigmoid(logp)               # Sigmoid gives probability of going up
    return p, h                     # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    """ Backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None   # Used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
reward_sum = 0
running_reward = None
episode_number = 0 if resume == False else resume_checkpoint

while True:
    if render: env.render()

    # Preprocess the observation and set input to network to be difference image
    cur_x = preprocess(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # Forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)

    # Roll the dice!
    # NOTE: See output for env.unwrapped.get_action_meanings() for the all the actions
    action = 2 if np.random.uniform() < aprob else 3

    # Record various intermediates (needed later for backprop)
    xs.append(x)                # Observation
    hs.append(h)                # Hidden state
    y = 1 if action == 2 else 0 # A "fake" label
    dlogps.append(y - aprob)    # Grad that encourages the action that was taken to be taken

    # Step the game environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # Record reward (has to be done after we call step() to get reward for previous action)

    if done: # An episode finished
        episode_number += 1

        # Stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], [] # Reset array memory

        # Compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # Standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # Modulate the gradient with advantage (PG magic happens right here)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # Accumulate grad over batch

        # Perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.iteritems():
                g = grad_buffer[k] # Gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # Reset batch gradient buffer

        # Book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'Resetting env. Episode reward total was %f. Running mean: %f' % (reward_sum, running_reward)
        if episode_number % 100 == 0: pickle.dump(model, open('save-'+str(episode_number)+'.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # Reset env
        prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print ('Ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
