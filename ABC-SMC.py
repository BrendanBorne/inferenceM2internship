#!/usr/bin/env python3
# coding: utf-8

# Author: Brendan Borne

print("""
           ____   _____             _____ __  __  _____ 
     /\   |  _ \ / ____|           / ____|  \/  |/ ____|
    /  \  | |_) | |       ______  | (___ | \  / | |     
   / /\ \ |  _ <| |      |______|  \___ \| |\/| | |     
  / ____ \| |_) | |____            ____) | |  | | |____ 
 /_/    \_\____/ \_____|          |_____/|_|  |_|\_____|
                                                        
 """)

# --------- INITIALIZATIONS --------- #

"""
We first import the needed libraries and toymodel.
"""

# Libraries
import numpy as np
import pandas as pd
import psutil
from datetime import datetime
import argparse
import json
import toymodel as model
from execute_loop import loop
pd.options.mode.chained_assignment = None  # default='warn'

"""
The parameters to estimate, the number of generations to run the ABC-SMC algorithm on, the number of particles we want to accept 
and the maximum number of simulations per generation will be the script's command line arguments.
"""

# Create parser
parser = argparse.ArgumentParser('python3 abc-smc algorithm')
# Arguments to estimate
parser.add_argument(
    "--arguments","-a",
    help='Arguments JSON file path',
    required=True,
    type=str,
    default='arguments.json'
)
# Number of generations
parser.add_argument(
    "--generations","-g",
    help="Number of generations to iterate on",
    required=True,
    type=int,
    default=5
)
# Number of accepted particles
parser.add_argument(
    "--accparc","-ap",
    help='Number of particles to accept per generation', 
    required=True,
    type=int,
    default=100
)
# Number of maximum runs to perform
parser.add_argument(
    "--maxruns",'-mr',
    help='Maximum number of runs to perform per generation',
    required=True,
    type=int,
    default=200
)
# Parse the arguments
arguments = parser.parse_args()

"""
We need to define a function to run the model, and a function to compute distance between simulation and observation data.
"""

# Running the model 
def run_model():
    # If we want to set a maximum infection spread distance, we need to set the matrix each time:
    model.aMatrAdjacence = model.setMatrix()
    sim = model.runSimulation()
    # Calculate summary statistics
    res = sim['detectedIPI'] # In this example, the summary statistics will be the number of detected PI animals.
    return res

# Computing distance between simulation and observation
def compute_distance(obs):
    sim = run_model()
    dist = 0
    for i in range(len(obs)):
        dist += (obs[i]-sim[i])**2
    return dist

"""
We then need to initialize the model. 
Since it uses a contact network, initializing it for each loop would be too long.
We also import the observation data.
"""

# Toy model
print('Initializing model :')

# Used to compute execution time
startTime = datetime.now()

# Setting a seed for reproductibility
np.random.seed(123456789) 

# 'Observation' data
observed_data = pd.read_csv('data/observed_data.csv')
observed_data.drop(columns={'Unnamed: 0'},inplace=True)
#observed_data = observed_data.groupby('t').mean().reset_index()
sum_stats_obs = observed_data['detectedIPI']

print('Model and observation data set. \n')
# ----------- ABC-SMC ----------- #

"""
Initialization of SMC :
We define all the variables we'll have to work with.
"""

print('Initializing SMC :')
T = arguments.generations # Number of generations to perform the SMC on
N = arguments.accparc # Number of particles to accept
maxattempts = arguments.maxruns # Maximum number of attempts to perform

# perturbation_kernel = 0.04 # Perturbation kernel to disturb the particles
threshold_quantile = 0.6 # Quantile to reduce epsilon as the SMC goes

t = 0 # Generation management
epsilon = 0 # Particle acceptation threshold

# Load arguments
print('Loading arguments to estimate...')
# Opening JSON file
with open(arguments.arguments) as json_file:
    prior = json.load(json_file)

cols = [key for key in prior.keys()]
cols.extend(['gen','w','dist'])

# All accepted particles
all_acc_particles = pd.DataFrame(columns=cols)
# Previously accepted particles
previous_acc_particles = pd.DataFrame(columns=cols)
# Current accepted particles
current_acc_particles = pd.DataFrame(columns=cols)

print('Everything is loaded. \n')

"""
Computation : 
We can now do the actual computations of the ABC-SMC.
"""

print("Running ABC-SMC...")

aTrack = []

# We are going to run the SMC for T generations
for t in range(T): 

    ii = 1 # Particles count for this generation
    totattempts = 0 # Attempts count for this generation

    # Repeat until we have accepted N particles
    while(ii <= N): 
        
        # Total attempts count
        totattempts += 1

        """
        The main loop is externalized both for clarity and to make it easier to work on specific aspects of the ABC-SMC.
        Upgrading the inference method can be done mostly in this part of the sources, without having to put one's hands in the externalized loop.
        """
        # Loop externalization
        out = loop(prior, ii, previous_acc_particles, model, compute_distance, sum_stats_obs, epsilon, t)

        # We only add the particle if it was accepted
        if out['accepted']:
            pars = out['pars']
            current_acc_particles = pd.concat([current_acc_particles, pars])
            
        ii=out['ii']

    
    # Normalize the weights
    nbAccParc=current_acc_particles.shape[0]
    current_acc_particles['w'] = current_acc_particles['w'] / sum(current_acc_particles['w'])

    # Keep track of the accepted particles number, total attempts number, epsilon values
    aTrack.append([t+1, nbAccParc, totattempts, epsilon])

    # Define next threshold
    epsilon = np.quantile(current_acc_particles['dist'], threshold_quantile)
    # Set dataframes
    previous_acc_particles = current_acc_particles
    all_acc_particles = pd.concat([all_acc_particles, current_acc_particles])
    current_acc_particles = pd.DataFrame(columns=cols)

    # Interruption condition
    print("Generation:", t+1, "\t Particles found:", nbAccParc, "\t Total attempts:", totattempts)
    if totattempts>=maxattempts: break

    # Increment generation number
    #t+=1

"""
The computations have been done.
Now we only need to export our results.
"""

# Exporting the generated particles
path_p = 'results/particles.csv'
all_acc_particles.to_csv(path_p)

# Exporting some tracking information
path_t = 'results/tracking.csv'
dfTrack = pd.DataFrame(aTrack, columns=['gen','acc_parc','tot_att','eps'])
dfTrack.to_csv(path_t)

print('')
print('Particles exported to ', path_p)
print('Tracking information exported to', path_t)
print('')
# Execution time
print('Execution time:', datetime.now() - startTime)
# Getting % usage of virtual_memory ( 3rd field)
print('RAM memory % used:', psutil.virtual_memory()[2])
# Getting usage of virtual_memory in GB ( 4th field)
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)