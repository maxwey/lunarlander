import gym
import random
import numpy
import tflearn
import tflearn.layers.core as tflearn_core
import tflearn.layers.estimator as tflearn_estimator

import multiprocessing
import os

#######################################################
#                Variable Declarations                #
#######################################################

# set the total number of random attempts to generate the training set
training_set_episode_count = 90000
training_set_max_frames_per_episode = 2000
score_threshold = 50

cores_to_use = 22

keep_probability = 0.8
learning_rate = 0.0001
epochs = 3

#######################################################
#                     Environment                     #
#######################################################


def create_training_population():
    env = gym.make('LunarLander-v2')

    # list of observations and the associated move
    sample_population = []
    # list of the outcome scores
    scores_list = []

    for episode in range(training_set_episode_count):
        observation = env.reset()
        total_score = 0

        current_moves = []
        prev_observation = None

        for frame_number in range(training_set_max_frames_per_episode):
            #env.render()

            # pick a random action from the ones available
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if prev_observation is not None:
                # store the state and the action that was taken
                current_moves.append([action, prev_observation])

            total_score += reward
            prev_observation = observation

            if done:
                if episode % 1000 == 0:
                   print("%d episode was done!" % (episode))
                break


        if total_score > score_threshold:
            for move in current_moves:

                # indicate what the expected neural network output is given an input
                if move[0] == 0:
                    output = [1, 0, 0, 0]
                elif move[0] == 1:
                    output = [0, 1, 0, 0]
                elif move[0] == 2:
                    output = [0, 0, 1, 0]
                elif move[0] == 3:
                    output = [0, 0, 0, 1]

                # save to the population the Environment, and the expected output
                sample_population.append([move[1], output])
            scores_list.append(total_score)


    # save the data to avoid having to re-run this
    numpy.save('data/ll-trainingdata-%d.npy' % ( os.getpid() ), numpy.array(sample_population))

    return sample_population


process_list = []
for _ in range(cores_to_use):
    p = multiprocessing.Process(target = create_training_population, args = ())
    p.start()

for _ in range(cores_to_use):
    p.join()


files = os.listdir('data')
combined_list = []
for name in files:
    newl = numpy.load('data/' + name).tolist()
    combined_list += newl

numpy.save('ll-training-data-combined.npy', numpy.array(combined_list))

