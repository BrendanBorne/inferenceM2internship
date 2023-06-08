#!/usr/bin/env python3
# coding: utf-8

# Author: Brendan Borne

"""
This script is used to externalize the ABC-SMC's main loop.
Upgrading the inference method can be done mostly in the ABC-SMC.py script of the sources, without having to put one's hands in the externalized loop.
It is not designed to be executed by itself.
"""

from scipy.stats import uniform
from scipy.stats import norm
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

def loop(prior, ii, previous_acc_particles, model, compute_distance, sum_stats_obs, epsilon, t):
    # SAMPLE PARTICLE
    # Create the particle
    proposed_particle = {}
    accepted=False      
    # If we are on generation 0, we sample from the prior
    if t == 0:     
        for name, value in prior.items():
            if value['type'] == "unif":
                val = np.random.uniform(low=value['min'],high=value['max'],size=1)
                proposed_particle[name] = val[0]
            else:
                raise Exception('Distribution type not supported in current version')   
    # If we are not on generation 0, we sample from the previous particle population
    else: 
        # Sample from previous population
        sample_index = np.random.choice(previous_acc_particles.shape[0],1,p=previous_acc_particles['w'])[0]
        sampled_particle = previous_acc_particles.iloc[sample_index]
        proposed_particle = sampled_particle
        # Disturb particle
        for name, value in prior.items():
            if value['type'] == "unif":
                while True:
                    disturbed_val = np.random.normal(proposed_particle[name], 2*np.std(previous_acc_particles[name]), 1)
                    # Check if new value is contained in initial prior
                    if uniform.pdf(x=disturbed_val,loc=value['min'],scale=value['max']) > 0:
                        proposed_particle[name] = disturbed_val[0]
                        break           
            else:
                raise Exception('Distribution type not supported in current version')

    # SIMULATE WITH PROPOSED PARTICLE
    # Compute distance
    pars = proposed_particle
    model.params = model.setParameters(pars)
    distance = compute_distance(sum_stats_obs)
    # If we are on the first generation
    if t==0:
        # Compute weight
        w = 1
        # Add particle to current accepted particles
        pars.update({'gen':t, 'w':w,'dist':distance})
        accepted=True
        pars = pd.DataFrame([pars])
        #current_acc_particles = pd.concat([current_acc_particles, pd.DataFrame([pars])])
        # Increment the number of accepted particles
        ii+=1
    # If we are not on the first generation
    else:
        # We only accept this particule if distance <= epsilon
        if distance<=epsilon:
            # Compute weight
            w = 0
            numerator = 1 
            denominator = previous_acc_particles['w']
            for name, value in prior.items():
                if value['type'] == "unif":
                    numerator = numerator * uniform.pdf(x=proposed_particle[name],loc=value['min'],scale=value['max'])
                    denominator = denominator * norm.pdf(x=proposed_particle[name],loc=previous_acc_particles[name],scale=2*np.std(previous_acc_particles[name]))
                    #denominator = denominator * norm.pdf(x=proposed_particle[name],loc=previous_acc_particles[name],scale=perturbation_kernel) 
                else:
                    raise Exception('Distribution type not supported in current version')
            w = numerator / np.sum(denominator)    
            # Add particle to current accepted particles
            pars.update({'gen':t, 'w':w,'dist':distance})
            accepted=True
            pars = pd.DataFrame([pars])
            #current_acc_particles = pd.concat([current_acc_particles, pd.DataFrame([pars])])
            # Increment number of accepted particles
            ii+=1

    """
    We return the results to the main script.
    """
    # Return result
    out = {
        'ii':ii,
        'accepted':accepted,
        'pars':pars
        }

    return out