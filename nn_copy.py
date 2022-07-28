from imp import new_module
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import random
import gym

from parser_file import layer_constant
from parser_file import no_iter, genetic_value
from parser_file import cross_mutate_value

class pyGad:
    def __init__(self, pop_size, iter):
        self.env = None
        self.curr_generation = []
        self.curr_score = []
        self.max_population_size = pop_size
        self.iteration = iter
        self.generation_no = 0

    def get_action_size(self):
        if self.env is None:
            return None
        return self.env.action_space.n

    def get_observation_size(self):
        if self.env is None:
            return None
        return self.env.reset().shape

    def set_environment(self, env_string):
        self.env = gym.make(env_string)

    def create_individual(self, inp_shape, out_size):
        model = Sequential()
        layer, layer_detail, activation_detail = layer_constant()
        #model.add(Dense(30, activation='relu', input_shape=inp_shape))
        #model.add(Dense(10, activation='relu'))
        
        model.add(Dense(layer_detail[0], activation=activation_detail[0], input_shape=inp_shape))

        for i in range(layer-1):
            model.add(Dense(layer_detail[i+1], activation=activation_detail[i+1]))

        model.add(Dense(out_size, activation='softmax'))
        return model

    def evaluvate(self):
        if self.env is None:
            return None

        print("Evaluation for Generation {} : ".format(self.generation_no))

        for i,model in enumerate(self.curr_generation):
            total_reward = 0
            ob = self.env.reset()
            prev_action = 0
            for iter in range(self.iteration):
                action = np.argmax(model.predict(ob.reshape(1, 128))[0])
                ob, rew, done, info = self.env.step(action)
                rew += self.fitness_function(prev_action, action)
                prev_action = action
                if done == True:
                    break
                total_reward += rew
            self.env.close()

            print("Model : {} || Score : {}".format(i, total_reward))

            self.curr_score.append(total_reward)

    def fitness_function(self, curr_action, prev_action):
        if(abs(curr_action - prev_action) == 1):
            return 1.0 
        return -1.0

    def play_current_best(self, index_value):
        
        curr_best_index = np.argmax(np.array(self.curr_score))

        if index_value != -1:
            curr_best_index = index_value

        curr_best_model = self.curr_generation[curr_best_index]

        print("Best Model {} ".format(curr_best_index))

        ob = self.env.reset()
        for iter in range(self.iteration):
            self.env.render()
            action = np.argmax(curr_best_model.predict(ob.reshape(1, 128))[0])
            ob, rew, done, info = self.env.step(action)
        self.env.close()

    def random_generation(self):
        if self.env is None:
            return None

        self.generation_no += 1

        for _ in range(self.max_population_size):
            self.curr_generation.append(self.create_individual(self.get_observation_size(), 
                                                self.get_action_size()))
        
    def get_sorted(self):
        index = list(range(len(self.curr_score)))
        index.sort(key = self.curr_score.__getitem__)
        self.curr_score[:] = [self.curr_score[i] for i in index]
        self.curr_generation[:] = [self.curr_generation[i] for i in index]

    def show_generation(self):
        print("Display Generation {} Population : ".format(self.generation_no))
        for i, score in enumerate(self.curr_score):
            print("Model {} || Score {}".format(i, score))

    def next_generation(self):
        
        self.generation_no += 1
        new_generation = []

        cross, mutate = cross_mutate_value()

        #adding top_k models to the next generation
        genome_no, generation_no = genetic_value()
        top_k_models = self.top_k_population(genome_no)
        new_generation.append(top_k_models[0])

        k = 0

        for i in range(len(mutate)):
            if mutate[i] == -1:
                new_generation.append(top_k_models[k])
                k = k+1
            else:
                new_generation.append(self.mutate_operator(0.2, top_k_models[mutate[i]-1]))

        for i in range(len(cross)):
            new_generation.append(self.crossover_operator(0.5, top_k_models[cross[i][0]], top_k_models[cross[i][1]]))

        #adding top mutated models
        #new_generation.append(self.mutate_operator(0.2, top_k_models[0]))
        #new_generation.append(self.mutate_operator(0.2, top_k_models[1]))

        #creating offsprint of the two models
        #new_generation.append(self.crossover_operator(0.5, top_k_models[0], top_k_models[1]))

        #creating a random environment
        #model = self.create_individual(self.get_observation_size(), 
        #                                        self.get_action_size())
        #new_generation.append(model)

        #and other GA algorithms
        
        self.curr_generation = new_generation
        self.curr_score = []

    def top_k_population(self, k):
        self.get_sorted()
        return self.curr_generation[len(self.curr_generation)-k:][::-1]

    def crossover_operator(self, rate, parent1, parent2):
        weights_1 = parent1.get_weights()
        weights_2 = parent2.get_weights()

        new_weights = []

        for layer1, layer2 in zip(weights_1, weights_2):
            noise = random.uniform(0, 1)
            if noise < rate:
                new_weights.append(layer1)
            else:
                new_weights.append(layer2)

        new_model = self.create_individual(self.get_observation_size(), 
                                                self.get_action_size())

        new_model.set_weights(new_weights)
        return new_model


    def mutate_operator(self, rate, parent):
        weights = parent.get_weights()
        for layers in weights:
            noise = random.uniform(0, 1)
            if noise < rate:
                layers += noise

        parent.set_weights(weights)
        return parent

