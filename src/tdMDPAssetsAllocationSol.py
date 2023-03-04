from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from src.rl_lib.distribution import Distribution, SampledDistribution
from src.rl_lib.markov_decision_process import MarkovDecisionProcess, NonTerminal, State, Terminal


@dataclass
class AssetAllocationMDP ( MarkovDecisionProcess [ float, float ] ):

    riskyDist: Sequence [ Distribution [ float ] ]
    riskfreeRate: Sequence [ float ]
    allocation_choices: Sequence [ float ]
    utility_func: Callable [ [ float ], float ]
    steps: int

    def step ( self, state: NonTerminal [ float ], action: float, ts: int, ttlSamples: int ) -> SampledDistribution [ Tuple [ State [ float ], float ] ]:
        def sample_func ( wealth = state, allocation = action, t = ts ) -> Tuple [ State [ float ], float ]:
            next_wealth: float = allocation * (1 + self.riskyDist [ t ].sample ( )) + (wealth.state - allocation) * (1 + self.riskfreeRate [ t ])
            reward: float = self.utility_func ( next_wealth ) if t == self.steps - 1 else 0
            next_state: State [ float ] = Terminal ( next_wealth ) if t == self.steps - 1 else NonTerminal ( next_wealth )
            return (next_state, reward)

        return SampledDistribution ( sampler = sample_func, expectation_samples = ttlSamples )

    def actions ( self, state: NonTerminal [ float ] ) -> Sequence [ float ]:
        return self.allocation_choices


if __name__ == '__main__':

    print ( "Transition Map" )
    print ( "--------------" )

    print ( "Stationary Distribution" )
    print ( "-----------------------" )
