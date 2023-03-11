from dataclasses import dataclass
from operator import itemgetter
from typing import Callable, Iterator, List, Mapping, Sequence, Set, Tuple

import numpy as np

from src.rl_lib.approximate_dynamic_programming import QValueFunctionApprox
from src.rl_lib.distribution import Categorical, Choose, Distribution, SampledDistribution
from src.rl_lib.function_approx import learning_rate_schedule, Tabular
from src.rl_lib.markov_decision_process import MarkovDecisionProcess, NonTerminal, State, Terminal


def generateReturnSequence ( a: float, b: float, p: float ) -> Sequence [ Distribution [ float ] ]:
    while True:
        yield Categorical ( {
            a: p,
            b: (1 - p)
            } )


def generateUtilityFunction ( utilityAlpha: float ) -> Callable [ [ float ], float ]:
    return lambda x: (1 - (np.exp ( -utilityAlpha * x ))) / utilityAlpha


def epsilon_greedy_action ( q: QValueFunctionApprox [ float, float ], nt_state: NonTerminal [ float ], actions: Set [ float ], epsilon: float
                            ) -> float:
    '''
    given :
    a non-terminal state,
    a Q-Value Function (in the form of a FunctionApprox: (state, action) -> Value
    epislon

    return an action sampled from the probability distribution implied by an
    epsilon-greedy policy that is derived from the Q-Value Function.
    '''
    greedy_action: float = max ( ((a, q ( (nt_state, a) )) for a in actions), key = itemgetter ( 1 ) ) [ 0 ]
    return Categorical ( { a: epsilon / len ( actions ) + (1 - epsilon if a == greedy_action else 0.) for a in actions } ).sample ( )


@dataclass ( init = False )
class AssetAllocationMDP ( MarkovDecisionProcess [ float, float ] ):
    """
    State : Wealth Wt,
    Action : investment in risky asset (= x_t)
    Investment in riskless asset : W_t - x_t

    risky asset return : { a ~ p; b ~ ( 1 - p ) }
    utility function : ( 1 - exp ( -alpha * Wealth ) ) / alpha
    """

    w0: float
    a: float
    b: float
    p: float
    utilityAlpha: float
    steps: int
    totalSample: int
    actionSpaceLimit: int

    riskyDist: Sequence [ Distribution [ float ] ]
    riskfreeRate: Sequence [ float ]
    utilityFunc: Callable [ [ float ], float ]

    def __init__ ( self, w0: float, a: float, b: float, p: float, utilityAlpha: float, steps: int, ttlSample: int, actionLimit: int = 10 ):
        super ( ).__init__ ( )
        self.w0 = w0
        self.a = a
        self.b = b
        self.p = p
        self.utilityAlpha = utilityAlpha
        self.steps = steps
        self.totalSample = ttlSample
        self.riskyDist = generateReturnSequence ( a, b, p )
        self.riskfreeRate = np.zeros ( steps )
        self.utilityFunc = generateUtilityFunction ( utilityAlpha )
        self.actionSpaceLimit = actionLimit

    def step ( self, state: NonTerminal [ float ], action: float, ts: int ) -> SampledDistribution [ Tuple [ State [ float ], float ] ]:
        def sample_func ( wealth = state, allocation = action, t = ts ) -> Tuple [ State [ float ], float ]:
            next_wealth: float = allocation * (1 + self.riskyDist [ t ].sample ( )) + (wealth.state - allocation) * (1 + self.riskfreeRate [ t ])
            reward: float = self.utilityFunc ( next_wealth ) if t == self.steps - 1 else 0
            next_state: State [ float ] = Terminal ( next_wealth ) if t == self.steps - 1 else NonTerminal ( next_wealth )
            return (next_state, reward)

        return SampledDistribution ( sampler = sample_func, expectation_samples = self.totalSample )

    def actions ( self, state: NonTerminal [ float ] ) -> Sequence [ float ]:
        return np.linspace ( 0, state.state, self.actionSpaceLimit )

    def getNonTerminalStates ( self, noOfTraces: int = 10000 ) -> Sequence [ NonTerminal [ float ] ]:
        ind: int = 0
        ret: List [ NonTerminal [ float ] ] = [ ]
        while ind < noOfTraces:
            s: State [ float ] = NonTerminal ( self.w0 )
            ts: int = 1
            ret.append ( s )
            ind = ind + 1
            while isinstance ( s, NonTerminal ):
                s = self.steps ( s, self.actions ( s ) [ 0 ], ts ).sample ( ) [ 0 ]
                ts = ts + 1
                ret.append ( s )
        return ret


@dataclass ( init = False )
class TDMDPAssetAllocationSol:
    """
    ∆w = α · (Rt+1 + γ · Q(St+1, At+1; w) − Q(St, At; w)) · ∇wQ(St, At; w)
    """
    mdp: AssetAllocationMDP
    initial_qvf_dict: Mapping [ Tuple [ NonTerminal [ float ], int ], float ]
    learning_rate_func: Callable [ [ int ], float ]
    qvfs: Iterator [ QValueFunctionApprox [ float, int ] ]
    max_episode_length: int
    gamma: float
    epsilon_as_func_of_episodes: Callable [ [ int ], float ]  # lambda k: k ** -0.5

    def __init__ ( self, w0: float, a: float, b: float, p: float, utilityAlpha: float, steps: int, totalSample: int, max_episode_length: int, epsilon_as_func_of_episodes: Callable [ [ int ], float ], gamma: float = 0.9, exponent: float = 0.5, initial_learning_rate: float = 0.03, half_life: float = 1000 ):
        self.mdp = AssetAllocationMDP ( w0, a, b, p, utilityAlpha, steps, totalSample )
        self.initial_qvf_dict = { (s, a): 0. for s in self.mdp.getNonTerminalStates ( ) for a in self.mdp.actions ( s ) }
        self.learning_rate_func = learning_rate_schedule ( initial_learning_rate = initial_learning_rate, half_life = half_life, exponent = exponent )
        self.max_episode_length = max_episode_length
        self.gamma = gamma
        self.epsilon_as_func_of_episodes = epsilon_as_func_of_episodes

    def glieSARSASolve ( self ) -> None:
        """
            Text Book P357
            GLI SARSA produces a generator (Iterator) of Q-Value Function estimates at the end
            of each atomic experience.
            The while True loops over trace experiences.
            The inner while loops over time steps—each of these steps involves the following
            """
        q: QValueFunctionApprox [ float, float ] = Tabular ( values_map = self.initial_qvf_dict, count_to_weight_func = self.learning_rate_func )
        yield q
        num_episodes: int = 0
        ntStateDist: Distribution [ NonTerminal [ float ] ] = Choose ( self.mdp.getNonTerminalStates ( ) )
        while True:
            num_episodes += 1
            epsilon: float = self.epsilon_as_func_of_episodes ( num_episodes )
            state: NonTerminal [ float ] = ntStateDist.sample ( )
            action: float = epsilon_greedy_action ( q = q, nt_state = state, actions = set ( self.mdp.actions ( state ) ), epsilon = epsilon )
            steps: int = 0
            while isinstance ( state, NonTerminal ) and steps < self.max_episode_length:
                next_state, reward = self.mdp.step ( state, action ).sample ( )
                if isinstance ( next_state, NonTerminal ):
                    next_action: float = epsilon_greedy_action ( q = q, nt_state = next_state, actions = set ( self.mdp.actions ( next_state ) ), epsilon = epsilon )
                    q = q.update ( [ ((state, action), reward + self.gamma * q ( (next_state, next_action) )) ] )
                    action = next_action
                else:
                    q = q.update ( [ ((state, action), reward) ] )
                yield q
                steps += 1
                state = next_state


if __name__ == '__main__':

    print ( "Transition Map" )
    print ( "--------------" )

    print ( "Stationary Distribution" )
    print ( "-----------------------" )
