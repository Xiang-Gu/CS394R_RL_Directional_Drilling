# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 05:53:32 2019
TC2
"""

import numpy as np
from algo import ValueFunctionWithApproximation
from tile_coding import tiles
class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 num_tilings:int,
                 num_tiles:np.array,
                 initw:np.array):
        """
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.num_tilings = num_tilings
        
        #If init w is not a scalar (this means we are initializing with a previously saved weight vector)
        if np.shape(initw) is not (): 
            self.w = initw
        else:
            self.w = initw*np.ones((int(num_tilings*np.product(num_tiles))),dtype=float)

    def __call__(self,s):
        """
        return the value of given state; V_hat(s)

        input:
            state
        output:
            value of the given state
        """
        #Determine tile index 
        feature_vector = tiles(self.w.shape[0],self.num_tilings,s)
        estimated_value = np.sum(self.w[feature_vector])
        return estimated_value

    def s2f(self,s):
        #Determine tile index 
        feature_vector = tiles(self.w.shape[0],self.num_tilings,s)
        return feature_vector

    def getWeight(self):
        return self.w

    def update(self,alpha,G,s_tau):   
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
#        print(s_tau)
        if s_tau.shape[0] != 5:
            print('STATES SENT TO V WITHOUT ACTION')
        #s_tau must be converted from continuous to feature form
        feature_vector = self.s2f(s_tau)
        #The gradient w/r to the weights is the feature vector for linear approximation
        gradient_V_w = feature_vector 
        V_hat = self.__call__(s_tau)
#        print('Value of state tau:',V_hat) ############# remove before submission
#        print('old W:',self.w) ############# remove before submission
        self.w[gradient_V_w]+=alpha*(G-V_hat)
#        print('New estimate for state '+str(s_tau)+' = '+str(self.__call__(s_tau)))
        return None