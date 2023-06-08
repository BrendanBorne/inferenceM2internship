#!/usr/bin/env python3
# coding: utf-8

# Author: Brendan Borne

# --------------------------------------------------------------------- #
# Bovine Viral Diarrhea (BVD) spread toy model.
# --------------------------------------------------------------------- #

# ---------------------------- LIBRARIES ----------------------------- #
import numpy as np
import pandas as pd
import math
from datetime import datetime
from scipy.stats import norm
from numpy.random import default_rng
rng = default_rng()
# --------------------------------------------------------------------- #

# Used to compute execution time
startTime = datetime.now()

# --------------------------- NODE CLASS ------------------------------ #
#                                                                       # 
# Variables                                                             #
#   dState : number of cows in each state | dictionary                  #
#   nodeSize : herd size                  | int                         #
#   id : herd id                          | int                         #
#   dep : herd department                 | str                         #
#                                                                       #
# Functions                                                             #
#   getState : returns dState                                           #
#   epidemyDynamics : simulates the spread of the epidemy               #
#                                                                       #
# --------------------------------------------------------------------- # 

class Node:

    def __init__(self, id, N, dep):
        self.id = id
        self.nodeSize = N
        self.dep = dep

        # If the node comes from a bordering department, the PI number is set to argument borderRisk (R)
        if self.dep in outDepartments:
            self.dState = {'S':self.nodeSize-params['borderRisk'], 'I':0, 'R':0, 'P':params['borderRisk']}
        # Otherwise, the starting number of PI is 0
        else:
            self.dState = {'S':self.nodeSize, 'I':0, 'R':0, 'P':0}

    def getState(self):
        return self.dState

    def epidemyDynamics(self):

        dStateTmp = self.dState

        # S -> I or S-> P
        # Computing infection rate given the amount of infected neighbours and model parameters
        infectionRate = params['betaOut'] * vVoisinsInfectes[self.id] + (params['betaInI']*self.dState['I'])/self.nodeSize + (params['betaInP']*self.dState['P'])/self.nodeSize
        # Transforming the rate into probability
        infectionProbability = 1 - math.exp(-infectionRate*params['dt'])
        if infectionProbability>1 :
            print(infectionProbability)
        # Drawing from a binomial law with computed probability
        infectionNb = np.random.binomial(self.dState['S'], infectionProbability)
        # Modifying state dictionary according to the result
        dStateTmp['S'] -= infectionNb
        dStateTmp['I'] += (1-params['mu'])*infectionNb
        dStateTmp['P'] += params['mu']*infectionNb
        
        # I -> R
        # Computing healing rate given model parameters
        healingRate = params['gamma']
        # Transforming the rate into probability
        healingProbability = 1 - math.exp(-healingRate*params['dt'])
        # Drawing from a binomial law with computed probability
        healingNb = np.random.binomial(self.dState['I'], healingProbability)
        # Modifying state dictionary according to the result
        dStateTmp['I'] -= healingNb
        dStateTmp['R'] += healingNb

        # R -> S
        # Computing immunity loss rate given model parameters
        immunityLossRate = params['omega']
        # Transforming the rate into probability
        immunityLossProbability = 1 - math.exp(-immunityLossRate*params['dt'])
        # Drawing from a binomial law with computed probability
        immunityLossNb = np.random.binomial(self.dState['R'], immunityLossProbability)
        # Modifying state dictionary according to the result
        dStateTmp['R'] -= immunityLossNb
        dStateTmp['S'] += immunityLossNb

        self.dState = dStateTmp

# ---------------------------- FUNCTIONS ------------------------------ #
# Initialization of adjacency matrix function
def initializeMatrix(row):
    # As we take herd distance into account, we use the transmission factor
    aMatrAdjacence[int(row['originID']),int(row['destID'])] = row['FT']
    aMatrAdjInit[int(row['originID']),int(row['destID'])] = row['FT']

    return row

# Initialization of metapopulation function
def initializeMetapopulation(row):
    # Generating a node with an id, a size and a department
    node = Node(row['nodeID'],row['size'],row['dep'])
    # Filling the nodes list
    lNodesMetapopulation.append(node) 
    return row

# Initialization of simulation function
def initializeSimulation(metapopSize):
    # Reinitializing metapopulation
    for n in lNodesMetapopulation:
        # If the node comes from a bordering department, the PI number is set to argument borderRisk (R)
        if n.dep in outDepartments:
            n.dState = {'S':n.nodeSize-params['borderRisk'], 'I':0, 'R':0, 'P':params['borderRisk']}
        # Otherwise, the starting number of PI is 0
        else:
            n.dState = {'S':n.nodeSize, 'I':0, 'R':0, 'P':0}

    # Infecting first herd
    # The first infected herds are chosen randomly
    # firstInfectedNb = np.random.randint(0,metapopSize)
    nbToInfect = int((metapopSize * params['initInfected']) // 1)
    firstInfectedNbs = rng.choice(metapopSize, size=nbToInfect, replace=False)

    # Infecting the first PI in this herd and constructing the infected herds list
    vInfectes = np.zeros(metapopSize)
    for ii in firstInfectedNbs:
        nP = np.random.randint(1,3)
        lNodesMetapopulation[ii].dState = {'S':lNodesMetapopulation[ii].dState['S']-1, 'I':0, 'R':0, 'P':nP}
        vInfectes[ii] = 1

    # Constructing the list containing the number of infected neighbours
    # This list is the mathematical result of the multiplication of the adjacency matrix by the previously defined infected list
    vVoisinsInfectes = np.matmul(aMatrAdjacence, vInfectes)

    return(vInfectes, vVoisinsInfectes)

# Update of infected list function
def updateInfected(node):
    # This list only counts PI individuals since they're the ones contaminating their neighbours
    vInfectes[node.id] = node.dState['P']

def setParameters(p):
    global params
    for key,value in p.items():
        params[key]=value

    return params

# --------------------------------------------------------------------- #

# ---------------------------- PARAMETERS ----------------------------- # 
#                                                                       #
# omega : immunity loss rate (1/omega = immunity duration)              #
# gamma : healing rate (1/gamma = infection duration)                   #
# betaInI : infection rate induced by I individuals inside herd         #
# betaInP : infection rate induced by P individuals inside herd         #
# betaOut : between herd infection rate                                 #
# mu : probability of becoming PI instead of I                          #
# Tmax : duration of simulation                                         #
# dt : time step                                                        #
# network : network on which the simulation will be computed            #
# output : folder in which the results will be exported                 #
# runs : number of runs                                                 #
# maxDist : maximum pathogen transmission distance                      #
# initInfected : initial proportion of infected herds                   #
# --------------------------------------------------------------------- #

params = {
        'betaInI':0.2,
        'betaInP':0.2,
        'betaOut':10,
        'borderRisk':1,
        'gamma':0.05,
        'omega':0.001,
        'mu':0.1,
        'Tmax':365,
        'dt':14,
        'runs':1,
        'maxDist':5000, 
        'initInfected':0.2 
    }
# Load the network
network = 'data/network.csv'

"""
  _             _
 | |           | |      
 | |____   ____| |_   __
 | '_ \ \ / / _` \ \ / /
 | |_) \ V / (_| |\ V / 
 |_.__/ \_/ \__,_| \_/               

"""
                      
# ------------------------ IMPORTATION OF DATA ------------------------ #

#print('Setting up network...')

# Importing network data and transforming distance into transmission parameter
dfLinks = pd.read_csv(network)
# We get distance data
distances = dfLinks['distance']
# Fit the normal distribution to the data
mu, sigma = norm.fit(distances)
mu = 0
sigma = 10000
# Transform the data using the fitted parameters
transformed_distances= np.abs(norm.pdf(distances, mu, sigma))
dfLinks['FT'] = transformed_distances

# Bordering departments list
outDepartments = ["44","49","50","53"] 

dfNodes = pd.DataFrame(dfLinks[['destID','size_destination','destination']])
dfNodes['destination'] = dfNodes['destination'].astype(str).str[:2]
dfNodes.rename(columns={'destID':'nodeID','size_destination':'size','destination':'dep'},inplace=True)
dfNodes.drop_duplicates(inplace=True)

# Constructing metapopulation
# List that will contain nodes
lNodesMetapopulation = []

# Filling the list
dfNodes.apply(initializeMetapopulation, axis=1)

# --------------------------------------------------------------------- #

# ---------------------- MATRIXES AND VECTORS ------------------------- #
#                                                                       #
# aMatrAdjacence : adjacency matrix                                     #
# vInfectes : infected herds list                                       #
# vVoisinsInfectes : number of infected neighbours list                 #
#                                                                       #
# --------------------------------------------------------------------- #      

# Constructing adjacency matrix
# We need the matrix's size
metapopSize = np.max(dfLinks['originID']) + 1
# Filling the matrix with zeros
aMatrAdjInit = np.zeros((metapopSize,metapopSize)) # We keep the initial matrix in memory
aMatrAdjacence = np.zeros((metapopSize,metapopSize)) # And use another one for the actual simulations

# If we want to use maxDist:
def setMatrix():
    # Used to transform matrix when using a maxDist parameter
    # dfLinks.apply(initializeMatrix,axis=1)
    aMatrAdjacence = aMatrAdjInit
    minFT = dfLinks.query(f'distance > {params["maxDist"]}')
    minFT = minFT['FT']
    minFT = max(minFT)
    # minFT = max(dfLinks.query('distance > @params["maxDist"]')['FT'])
    aMatrAdjacence[aMatrAdjacence <= minFT] = 0

    return aMatrAdjacence

# Initialize matrix with values
dfLinks.apply(initializeMatrix,axis=1)

def runSimulation():
    # Epidemy dynamics output 
    lOutput = []

    for runNumber in range(params['runs']):

        global vInfectes, vVoisinsInfectes
    
        vInfectes, vVoisinsInfectes = initializeSimulation(metapopSize)

        total_Infected = set(())

        for t in range(0, params['Tmax'], params['dt']):
            # Counting the number of S, I, R and P for each time step
            nbS = 0
            nbI = 0
            nbR = 0
            nbP = 0
            nbDetected = 0

            for n in lNodesMetapopulation:

                # We do not want to simulate epidemy dynamics for herds that are in the bordering departments
                if n.dep not in outDepartments:
                    n.epidemyDynamics()

                    # Observation scheme

                    if t % 7 == 0:

                        # Death of PI animals
                        dead = np.random.binomial(n.dState['P'],0.01)
                        n.dState['P'] -= dead
                        n.dState['S'] += dead

                        # Detection of PI animals
                        detected = np.random.binomial(n.dState['P'], 1) # Chance to detect IPIs as parameter ?
                        n.dState['P'] -= detected
                        n.dState['S'] += detected
                        
                
                # Updating infected vector
                updateInfected(n)

                nbS += n.dState['S']
                nbI += n.dState['I']
                nbR += n.dState['R']
                nbP += n.dState['P']
                nbDetected += detected                                 

                if n.dState['I'] >= 1 or n.dState['P'] >= 1:
                    total_Infected.add(n.id)

            vVoisinsInfectes = np.matmul(aMatrAdjacence,vInfectes)

            # Output
            lOutput.append([runNumber, t, nbS, nbI, nbR, nbP, nbDetected]) 


    # Shaping the output
    dfOutput = pd.DataFrame(lOutput, columns=['run','t','S','I','R','P', 'detectedIPI'])

    return(dfOutput)