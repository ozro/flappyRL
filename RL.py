import numpy as np
import random

random.seed(8000)
np.random.seed(8000)

from tensorflow import set_random_seed, Session, ConfigProto
set_random_seed(8000)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras import backend
from keras.initializers import lecun_uniform


sess = Session(config=ConfigProto(inter_op_parallelism_threads=1))
backend.set_session(sess)

"""
Here are two values you can use to tune your Qnet
You may choose not to use them, but the training time
would be significantly longer.
Other than the inputs of each function, this is the only information
about the nature of the game itself that you can use.
"""
PIPEGAPSIZE  = 100
BIRDHEIGHT = 24

class QNet(object):

    def __init__(self):

        """
        Initialize neural net here.
        You may change the values.

        Args:
            num_inputs: Number of nodes in input layer
            num_hidden1: Number of nodes in the first hidden layer
            num_hidden2: Number of nodes in the second hidden layer
            num_output: Number of nodes in the output layer
            lr: learning rate
        """
        self.num_inputs = 2
        self.num_hidden1 = 50
        self.num_hidden2 = 10
        self.num_output = 2

        self.lr = 0.1;
        self.build()

        self.gamma = 0.01
        self.gamma_growth = 1.0001
        self.gamma_max = 0.995

        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.epsilon_min = 0

        self.memory = [] 
        self.memory_size = 1000
        self.memory_index = 0
        self.exploring = True

        self.batch_min = 200
        self.batch_size = 200

        self.last_reward = 0

    def build(self):
        """
        Builds the neural network using keras, and stores the model in self.model.
        Uses shape parameters from init and the learning rate self.lr.
        You may change this, though what is given should be a good start.
        """
        model = Sequential()
        model.add(Dense(self.num_hidden1, init=lecun_uniform(seed=8000), input_shape=(self.num_inputs,)))
        model.add(Activation('relu'))

        #model.add(Dense(self.num_hidden2, init=lecun_uniform(seed=8000)))
        #model.add(Activation('relu'))

        model.add(Dense(self.num_output, init=lecun_uniform(seed=8000)))
        model.add(Activation('linear'))

        rms = RMSprop(lr=self.lr)
        model.compile(loss='mse', optimizer=rms)
        self.model = model

    def flap(self, input_data):
        """
        Use the neural net as a Q function to act.
        Use self.model.predict to do the prediction.

        Args:
            input_data (Input object): contains information you may use about the 
            current state.

        Returns:
            (choice, prediction, debug_str): 
                choice (int) is 1 if bird flaps, 0 otherwise. Will be passed
                    into the update function below.
                prediction (array-like) is the raw output of your neural network,
                    returned by self.model.predict. Will be passed into the update function below.
                debug_str (str) will be printed on the bottom of the game
        """

        # state = your state in numpy array
        # prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size=1)[0]
        # choice = make choice based on prediction
        # debug_str = ""
        # return (choice, prediction, debug_str)

        state = np.array((input_data.distX, input_data.distY)).reshape(1, self.num_inputs)
        prediction = self.model.predict(state, batch_size=1)
        flap = np.argmax(prediction) 

        if(random.random() <= self.epsilon):
            print("EXPLORATION!")
            flap = 1 if random.random() < 0.5 else 0

        return flap, prediction, "(" + str(round(state[0][0])) + "," + str(round(state[0][1])) + ") [" + str(round(prediction[0][0])) + ", " +  str(round(prediction[0][1])) + "], R:" + str(self.last_reward)

    def remember(self, last_state, target):
        if(self.exploring):
            self.memory.append(np.array((last_state, target)))
            if(len(self.memory) == self.memory_size):
                self.exploring = False
        else:
            self.memory[self.memory_index] = np.array((last_state, target))
            self.memory_index = (self.memory_index + 1) % self.memory_size

        print(len(self.memory))

    def batch_sample(self):
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        last_states = np.squeeze(minibatch[:, 0])
        targets = np.squeeze(minibatch[:, 1])
        
        self.model.fit(last_states, targets, batch_size = self.batch_size, epochs=1)

    def update(self, last_input, last_choice, last_prediction, crash, scored, playerY, pipVelX):
        """
        Use Q-learning to update the neural net here
        Use self.model.fit to back propagate

                Args:
            last_input (Input object): contains information you may use about the
                input used by the most recent flap() 
            last_choice: the choice made by the most recent flap()
            last_prediction: the prediction made by the most recent flap()
            playerY: y position of the bird, used for calculating new state
            pipVelX: velocity of pipe, used for calculating new state

        Returns:
            None
        """
        # This is how you calculate the new (x,y) distances
        # new_distX = last_input.distX + pipVelX
        # new_distY = last_input.pipeY - playerY

        # state = compute new state in numpy array
        # reward = compute your reward
        # prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size = 1)

        # update old prediction from flap() with reward + gamma * np.max(prediction)
        # record updated prediction and old state in your mini-batch
        # if batch size is large enough, back propagate
        # self.model.fit(old states, updated predictions, batch_size=size, epochs=1)

        last_state = np.array((last_input.distX, last_input.distY)).reshape(1, self.num_inputs)

        new_distX = last_input.distX + pipVelX
        new_distY = last_input.pipeY - playerY
        next_state = np.array((new_distX, new_distY)).reshape(1, self.num_inputs)

        next_prediction = self.model.predict(next_state, batch_size = 1) 

        if(crash):
            reward = -1000
        elif(scored):
            reward = 100
        elif(PIPEGAPSIZE - new_distY > 0 and new_distY - BIRDHEIGHT> 0): #inside target
            reward = 10
        elif(new_distY - BIRDHEIGHT < 0): #below target
            if(last_choice == 0):
                reward = -100
            else:
                reward = -1
        else: #above target
            if(last_choice == 0):
                reward = -1
            else:
                reward = -100

        self.last_reward = reward

        if(crash):
            #target = reward
            target = reward + self.gamma * np.max(next_prediction)
        else:
            target = reward + self.gamma * np.max(next_prediction)

        target_prediction = last_prediction.copy()
        target_prediction[0][last_choice] = target

        self.remember(last_state, target_prediction)

        print("Flap:", last_choice, "State:", last_state, "Prediction", last_prediction, "Target", target_prediction, "Reward", reward)
        
        if(len(self.memory) >= self.batch_min):

            if(self.epsilon > self.epsilon_min):
                self.epsilon *= self.epsilon_decay
            elif(self.epsilon < self.epsilon_min):
                self.epsilon = self.epsilon_min
            print("Epsilon:", self.epsilon)

            if(self.gamma < self.gamma_max):
                self.gamma *= self.gamma_growth
            elif(self.gamma > self.gamma_max):
                self.gamma = self.gamma_max
            print("Gamma:", self.gamma)

            self.batch_sample()
        else:
            print("Waiting for more samples")


class Input:
    def __init__(self, playerX, playerY, pipeX, pipeY,
                distX, distY):
        """
        playerX: x position of the bird
        playerY: y position of the bird
        pipeX: x position of the next pipe
        pipeY: y position of the next pipe
        distX: x distance between the bird and the next pipe
        distY: y distance between the bird and the next pipe
        """
        self.playerX = playerX
        self.playerY = playerY
        self.pipeX = pipeX
        self.pipeY = pipeY
        self.distX = distX
        self.distY = distY

