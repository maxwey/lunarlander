import gym
import random
import numpy
import tflearn
import tflearn.layers.core as tflearn_core
import tflearn.layers.estimator as tflearn_estimator

import sys

env = gym.make('LunarLander-v2')

#######################################################
#                Variable Declarations                #
#######################################################

keep_probability = 0.8
learn_rate = 0.0001

#######################################################
#                  Neural Net Def                     #
#######################################################


def create_new_neural_model(size):
    net = tflearn_core.input_data(shape = [None, size, 1], name = 'input_layer')

    net = tflearn_core.fully_connected(net, 128, activation = 'relu')
    net = tflearn_core.dropout(net, keep_probability)

    net = tflearn_core.fully_connected(net, 128, activation = 'relu')
    net = tflearn_core.dropout(net, keep_probability)

    net = tflearn_core.fully_connected(net, 4, activation = 'softmax')
    net = tflearn_estimator.regression(net, optimizer = 'adam', learning_rate=learn_rate, loss = 'categorical_crossentropy', name = 'output_layer')
    model = tflearn.DNN(net, tensorboard_dir = 'model_log')

    return model


#######################################################
#                     Load Model                      #
#######################################################

if len(sys.argv) != 1 :
   tmodel = create_new_neural_model(8)
   tmodel.load(sys.argv[1])
else:
   tmodel = None

# Run 100 demos
for episode in range(100):
    observation = env.reset()
    total_score = 0

    prev_observation = None

    for frame_number in range(1000):
        env.render()

        # pick a random action from the ones available
        if prev_observation is None or tmodel is None:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(tmodel.predict(prev_observation.reshape(-1, len(prev_observation), 1))[0])

        observation, reward, done, info = env.step(action)

        total_score += reward
        prev_observation = observation

        if done:
            print("%d episode was done! Took %d frames for a score of %f" % (episode, frame_number, total_score))
            break
