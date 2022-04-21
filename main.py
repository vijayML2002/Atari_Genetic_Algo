import numpy as np
from nn import pyGad

iteration = 100
pop_size = 5

gmodel = pyGad(pop_size, iteration)
gmodel.set_environment("SpaceInvaders-ram-v0")

gmodel.random_generation()
gmodel.evaluvate()

gmodel.show_generation()

gmodel.get_sorted()
gmodel.show_generation()


