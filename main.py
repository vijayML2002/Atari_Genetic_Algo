
import numpy as np
from nn_copy import pyGad

from parser_file import layer_constant
from parser_file import no_iter, genetic_value
from parser_file import cross_mutate_value

iteration = no_iter()
genome_no, genetic_no = genetic_value()
pop_size = genome_no

print("Population size : ", pop_size)

gmodel = pyGad(pop_size, iteration)
gmodel.set_environment("SpaceInvaders-ram-v0")

gmodel.random_generation()
gmodel.evaluvate()

gmodel.play_current_best(-1)
#gmodel.play_current_best(1)


for i in range(100):
    gmodel.next_generation()
    gmodel.evaluvate()
    gmodel.play_current_best(-1)
    #gmodel.play_current_best(1)


