from dataclasses import dataclass
from typing import Callable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

from src.rl_lib.approximate_dynamic_programming import NTStateDistribution, QValueFunctionApprox
from src.rl_lib.control_utils import get_vf_and_policy_from_qvf
from src.rl_lib.distribution import Categorical, Choose, Distribution, SampledDistribution, Uniform
from src.rl_lib.function_approx import learning_rate_schedule, Tabular
from src.rl_lib.markov_decision_process import MarkovDecisionProcess, NonTerminal, State, Terminal
from src.rl_lib.td import glie_sarsa


def generateReturnSequence ( a: float, b: float, p: float ) -> Sequence [ Distribution [ float ] ]:
    while True:
        yield Categorical ( {
            a: p,
            b: (1 - p)
            } )


def generateUtilityFunction ( utilityAlpha: float ) -> Callable [ [ float ], float ]:
    return lambda x: (1 - (np.exp ( -utilityAlpha * x ))) / utilityAlpha


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

    riskyDist: Sequence [ Distribution [ float ] ]
    riskfreeRate: Sequence [ float ]
    utilityFunc: Callable [ [ float ], float ]

    def __init__ ( self, w0: float, a: float, b: float, p: float, utilityAlpha: float, steps: int, ttlSample: int ):
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

    def step ( self, state: NonTerminal [ float ], action: float, ts: int ) -> SampledDistribution [ Tuple [ State [ float ], float ] ]:
        def sample_func ( wealth = state, allocation = action, t = ts ) -> Tuple [ State [ float ], float ]:
            next_wealth: float = allocation * (1 + self.riskyDist [ t ].sample ( )) + (wealth.state - allocation) * (1 + self.riskfreeRate [ t ])
            reward: float = self.utilityFunc ( next_wealth ) if t == self.steps - 1 else 0
            next_state: State [ float ] = Terminal ( next_wealth ) if t == self.steps - 1 else NonTerminal ( next_wealth )
            return (next_state, reward)

        return SampledDistribution ( sampler = sample_func, expectation_samples = self.totalSample )

    def actions ( self, state: NonTerminal [ float ] ) -> Sequence [ float ]:
        # TODO fix infinity
        while True:
            yield Uniform ( right = state.state ).sample ( )

    def getNonTerminalStateDistribution ( self, noOfTraces: int = 10000 ) -> NTStateDistribution [ float ]:
        ind: int = 0
        ret: List [ float ] = [ ]
        while ind < noOfTraces:
            s: State [ float ] = NonTerminal ( self.w0 )
            ts: int = 1
            ret.append ( s )
            ind = ind + 1
            while isinstance ( s, NonTerminal ):
                s = self.steps ( s, self.actions ( s ) [ 0 ], ts ).sample ( ) [ 0 ]
                ts = ts + 1
                ret.append ( s )
        return Choose ( ret )


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
    epsilon_as_func_of_episodes: float

    def __init__ ( self, w0: float, a: float, b: float, p: float, utilityAlpha: float, steps: int, totalSample: int, max_episode_length: int, epsilon_as_func_of_episodes: float, gamma: float = 0.9, exponent: float = 0.5, initial_learning_rate: float = 0.03, half_life: float = 1000 ):
        self.mdp = AssetAllocationMDP ( w0, a, b, p, utilityAlpha, steps, totalSample )
        self.initial_qvf_dict = { (s, a): 0. for s in self.mdp.getNonTerminalStateDistribution ( ) for a in self.mdp.actions ( s ) }
        self.learning_rate_func = learning_rate_schedule ( initial_learning_rate = initial_learning_rate, half_life = half_life, exponent = exponent )
        self.max_episode_length = max_episode_length
        self.gamma = gamma
        self.epsilon_as_func_of_episodes = epsilon_as_func_of_episodes

    def glieSARSASolve ( self ) -> None:
        self.qvfs = glie_sarsa ( mdp = self.mdp, states = Uniform ( self.mdp.getNonTerminalStateDistribution ( ) ), approx_0 = Tabular ( values_map = self.initial_qvf_dict, count_to_weight_func = self.learning_rate_func ), gamma = self.gamma, epsilon_as_func_of_episodes = self.epsilon_as_func_of_episodes, max_episode_length = self.max_episode_length )

    def printSolution ( self ) -> None:
        import itertools
        import src.rl_lib.iterate as iterate
        num_updates = num_episodes * self.max_episode_length
        final_qvf: QValueFunctionApprox [ float, int ] = iterate.last ( itertools.islice ( self.qvfs, num_updates ) )
        opt_vf, opt_policy = get_vf_and_policy_from_qvf ( mdp = self.mdp, qvf = final_qvf )
        print ( f"GLIE SARSA Optimal Value Function with {num_updates: d } updates" )
        print ( opt_vf )
        print ( f"GLIE SARSA Optimal Policy with {num_updates: d } updates" )
        print ( opt_policy )


if __name__ == '__main__':

    print ( "Transition Map" )
    print ( "--------------" )

    print ( "Stationary Distribution" )
    print ( "-----------------------" )
