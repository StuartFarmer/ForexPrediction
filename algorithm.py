from __future__ import print_function

import numpy as np
import itertools
import os

from neat import nn, population, statistics

import random
import csv
import pickle

DAYS_OF_DATA = 180 # how many days the organism is given to work with
DAYS_OUT = 15 # how many days out the evaluation attempts to predict
EVALUATIONS_PER_ORGANISMS = 100
GENERATIONS_TO_EVALUATE = 1000

# import the forex data from the csv into a list
data = []

with open('data.csv', 'rU') as f:  #opens PW file
    reader = csv.reader(f)
    data = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
    f.close() #close the csv

def eval_fitness(genomes):
	fitness = 0
	something = 0
	best_fitness = -99999
	for g in genomes:
		fitness = 0
		net = nn.create_feed_forward_phenotype(g)

		#generate some testData
		testData = []

		while len(testData) < EVALUATIONS_PER_ORGANISMS:
			index = random.randrange(0, len(data) - DAYS_OF_DATA - DAYS_OUT - 1)

			selection = [float(val) for sublist in data[0 + index:DAYS_OF_DATA + index] for val in sublist]

			x = data[DAYS_OF_DATA + index + 1:DAYS_OF_DATA + index + DAYS_OUT + 1]
			
			nextSelection = []
			for d in x:
				nextSelection.append(float(d[3]))
			
			testData.append((selection, nextSelection))

		# activate the neural net
		for d in testData:
			predictions = net.serial_activate(d[0])
			actual = d[1]
			for x, prediction in enumerate(predictions):
				fitness += -(abs(prediction-actual[x]))/(DAYS_OUT * EVALUATIONS_PER_ORGANISMS) # describes average error on EACH evaluation
		
		# evaluate the fitness
		g.fitness = fitness
		# save this sucker
		if fitness > best_fitness:
			best_fitness = fitness
			with open('best_brain', 'wb') as output:
				pickle.dump(g, output, pickle.HIGHEST_PROTOCOL)
    		
			with open('best_fitness', 'a') as output:
				output.write(str(best_fitness))

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')

pop = population.Population(config_path)
pop.run(eval_fitness, GENERATIONS_TO_EVALUATE)

winner = pop.statistics.best_genome()

print('winning brain exported via pickle')

with open('best_brain', 'wb') as output:
    pickle.dump(winner, output, pickle.HIGHEST_PROTOCOL)