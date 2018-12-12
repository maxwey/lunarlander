import gym
import random
import numpy
import statistics
import tensorflow
import tflearn
import tflearn.layers.core as tflearn_core
import tflearn.layers.estimator as tflearn_estimator

env = gym.make('LunarLander-v2')

#######################################################
#                Variable Declarations                #
#######################################################

# set the total number of random attempts to generate the training set
training_set_episode_count = 50000
training_set_max_frames_per_episode = 2000
score_threshold = 50

keep_probability = 0.8
learning_rate = 0.0001
epochs = 1


#######################################################
#                     Environment                     #
#######################################################

# Creates Tensorflow neural net
def create_new_neural_model(size):
    net = tflearn_core.input_data(shape = [None, size, 1], name = 'input_layer')

    net = tflearn_core.fully_connected(net, 128, activation = 'relu')
    net = tflearn_core.dropout(net, keep_probability)

    net = tflearn_core.fully_connected(net, 128, activation = 'relu')
    net = tflearn_core.dropout(net, keep_probability)

    net = tflearn_core.fully_connected(net, 4, activation = 'softmax')
    net = tflearn_estimator.regression(net, optimizer = 'adam', learning_rate=learning_rate, loss = 'categorical_crossentropy', name = 'output_layer')
    model = tflearn.DNN(net, tensorboard_dir = 'model_log')

    return model


# Trains neural net from given data set
def train_model(data, model = None):
    observations = numpy.array([x[0] for x in data]).reshape(-1, len(data[0][0]), 1)
    target = [x[1] for x in data]

    if model is None:
        tensorflow.reset_default_graph()
        model = create_new_neural_model(len(observations[0]))

    model.fit({'input_layer' : observations}, {'output_layer' : target}, n_epoch = epochs, show_metric = True)

    return model


# Creates a neural net and trains model from previous iterations, updating
# population set with best selected moves from last generation's episodes.
def iterative():
    # Load starting population from data file (created by selecting best moves from random population)
    sample_population = numpy.load('tdata.npy').tolist()
    # Keep original starting population size
    len_pop = len(sample_population)
    # "Concentration" in current population of newly selected
    new_conc = 0.5
    # "Concentration" in current population of old population
    old_conc = 0.5

    print("GENERATION 0")
    # Train first model on starting population
    prev_model = train_model(sample_population)
    new_pop = score(prev_model)
    # If positive scoring plays found, add to sample_population
    if len(new_pop) > 0:
        # replace portion of old population with new population
        new_pop = int((len_pop * new_conc) / len(new_pop)) * new_pop
        sample_population = new_pop + random.sample(sample_population, int(len_pop * old_conc))

    i = 0
    # iteratively train model and update population with best selected moves from last generation
    while (1):
        i += 1
        print("\nGENERATION", i)
        # train model on previous model and updated population
        tmodel = train_model(sample_population, prev_model)
        tmodel.save("model{}.tflearn".format(i))

        new_pop = score(tmodel)
        # If positive scoring plays found, add to sample_population
        if len(new_pop) > 0:
            # replace portion of old population with new population
            new_pop = int((len_pop * new_conc) / len(new_pop)) * new_pop
            sample_population = new_pop + random.sample(sample_population, int(len_pop * old_conc))

        prev_model = tmodel


# Given trained model, run episodes of games and select best scoring moves
def score(tmodel):
    scores_list = []
    new_pop = []
    # run 200 episodes
    for episode in range(200):
        env.reset()
        total_score = 0

        current_moves = []
        prev_observation = None

        # cut off game at 600 frames
        for frame_number in range(600):
            if episode % 10 == 0:
                env.render()

            # pick a random action from the ones available
            if prev_observation is None:
                action = env.action_space.sample()
            else:
                action = numpy.argmax(tmodel.predict(prev_observation.reshape(-1, len(prev_observation), 1))[0])

            observation, reward, done, info = env.step(action)

            if prev_observation is not None:
                # store the state and the action that was taken
                current_moves.append([action, prev_observation])

            total_score += reward
            prev_observation = observation

            if done:
                print("%d episode was done! Took %d frames for a score of %f" % (episode, frame_number, total_score))
                break

        # keep best scores to add to population
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
                new_pop.append([move[1], output])
        scores_list.append(total_score)

    print('Average accepted score:', statistics.mean(scores_list))
    print('Median score for accepted scores:', statistics.median(scores_list))

    return new_pop


if __name__ == '__main__':
    iterative()
