#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:11:28 2019

@author: root
"""
from gym.envs.registration import register
register(
    id='DeterministicEnv-v0',
    entry_point='env.deterministicEnvironment:DeterministicEnv',
)

register(
    id='NoisyNextStateEnv-v0',
    entry_point='env.noisyNextStateEnvironment:NoisyNextStateEnv',
)

register(
    id='StochasticEnv-v0',
    entry_point='env.stochasticEnvironment:StochasticEnv',
)
        
register(
    id='RandomStartingStateEnv-v0',
    entry_point='env.randomStartingStateEnvironment:RandomStartingStateEnv',
)

register(
    id='SmallVaringGoalRep1Env-v0',
    entry_point='env.smallVaringGoalRep1Environment:SmallVaringGoalRep1Env',
)

register(
    id='LargeVaringGoalRep1Env-v0',
    entry_point='env.largeVaringGoalRep1Environment:LargeVaringGoalRep1Env',
)

register(
    id='SmallVaringGoalRep2Env-v0',
    entry_point='env.smallVaringGoalRep2Environment:SmallVaringGoalRep2Env',
)

register(
    id='LargeVaringGoalRep2Env-v0',
    entry_point='env.largeVaringGoalRep2Environment:LargeVaringGoalRep2Env',
)

register(
    id='SmallVaringGoalRep0Env-v0',
    entry_point='env.smallVaringGoalRep0Environment:SmallVaringGoalRep0Env',
)

register(
    id='LargeVaringGoalRep0Env-v0',
    entry_point='env.largeVaringGoalRep0Environment:LargeVaringGoalRep0Env',
)