from dataclasses import dataclass
from operator import itemgetter
from typing import Callable, Iterator, List, Mapping, Sequence, Set, Tuple

import numpy as np

from src.rl_lib.approximate_dynamic_programming import back_opt_qvf, back_opt_vf_and_policy, QValueFunctionApprox
from src.rl_lib.distribution import Categorical, Choose, Distribution, SampledDistribution
from src.rl_lib.function_approx import AdamGradient, DNNApprox, DNNSpec, learning_rate_schedule, Tabular
from src.rl_lib.markov_decision_process import MarkovDecisionProcess, NonTerminal, State, Terminal
from pprint import pprint

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

    a: float
    b: float
    p: float
    utilityAlpha: float
    steps: int
    totalSample: int
    actionSpaceLimit: int

    riskyDist: Distribution [ float ]
    riskfreeRate: float
    alloc_choices: Sequence [ float ]
    utilityFunc: Callable [ [ float ], float ]

    def __init__ ( self, a: float, b: float, p: float, utilityAlpha: float, steps: int, ttlSample: int = 1000, actionLimit: int = 20 ):
        super ( ).__init__ ( )
        self.a = a
        self.b = b
        self.p = p
        self.utilityAlpha = utilityAlpha
        self.steps = steps
        self.totalSample = ttlSample
        self.riskyDist = Categorical ( {
            a: p,
            b: (1 - p)
            } )
        self.riskfreeRate = 0.0
        self.utilityFunc = generateUtilityFunction ( utilityAlpha )
        self.actionSpaceLimit = actionLimit

    def step ( self, state: NonTerminal [ float ], action: float, ts: int ) -> SampledDistribution [ Tuple [ State [ float ], float ] ]:
        def sample_func ( wealth = state, allocation = action, t = ts ) -> Tuple [ State [ float ], float ]:
            next_wealth: float = wealth.state * (allocation * (1 + self.riskyDist.sample ( )) + (1.0 - allocation) * (1 + self.riskfreeRate))
            reward: float = self.utilityFunc ( next_wealth ) if t == self.steps - 1 else 0
            next_state: State [ float ] = Terminal ( next_wealth ) if t == self.steps - 1 else NonTerminal ( next_wealth )
            return (next_state, reward)

        return SampledDistribution ( sampler = sample_func, expectation_samples = self.totalSample )

    def actions ( self, state: NonTerminal [ float ] ) -> Sequence [ float ]:
        return np.linspace ( 0, 1.0, self.actionSpaceLimit )

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
        
    risky_alloc_choices: Sequence[ float ]
    feature_funcs: Sequence [ Callable [ [ Tuple [ float, float ] ], float ] ]  # = [ lambda _: 1., lambda w_x: w_x [ 0 ], lambda w_x: w_x [ 1 ], lambda w_x: w_x [ 1 ] * w_x [ 1 ] ]
    dnn_spec: DNNSpec # = DNNSpec ( neurons = [ ], bias = False, hidden_activation = lambda x: x, hidden_activation_deriv = lambda y: np.ones_like ( y ), output_activation = lambda x: - np.sign ( a ) * np.exp ( -x ), output_activation_deriv = lambda y: -y )
    initial_wealth: float
    
    def __init__ ( self, w0: float, a: float, b: float, p: float, utilityAlpha: float, steps: int, totalSample: int, max_episode_length: int, epsilon_as_func_of_episodes: Callable [ [ int ], float ], feature_funcs: Sequence [ Callable [ [ Tuple [ float, float ] ], float ] ], dnn: DNNSpec, gamma: float = 0.9, exponent: float = 0.5, initial_learning_rate: float = 0.03, half_life: float = 1000 ):
        self.mdp = AssetAllocationMDP ( a, b, p, utilityAlpha, steps, totalSample )
        self.initial_qvf_dict = { (s, a): 0. for s in self.mdp.getNonTerminalStates ( ) for a in self.mdp.actions ( s ) }
        self.learning_rate_func = learning_rate_schedule ( initial_learning_rate = initial_learning_rate, half_life = half_life, exponent = exponent )
        self.max_episode_length = max_episode_length
        self.gamma = gamma
        self.epsilon_as_func_of_episodes = epsilon_as_func_of_episodes
        
        self.feature_funcs = feature_funcs
        self.dnn_spec = dnn
        self.initial_wealth = w0

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

    def uniform_actions( self ) -> Choose[ float ]:
        return Choose( self.mdp.actions( ) )
    
    def get_qvf_func_approx ( self ) -> DNNApprox [ Tuple [ NonTerminal [ float ], float ] ]:

        adam_gradient: AdamGradient = AdamGradient ( learning_rate = 0.1, decay1 = 0.9, decay2 = 0.999 )
        ffs: List [ Callable [ [ Tuple [ NonTerminal [ float ], float ] ], float ] ] = [ ]
        for f in self.feature_functions:
            def this_f ( pair: Tuple [ NonTerminal [ float ], float ], f = f ) -> float:
                return f ( (pair [ 0 ].state, pair [ 1 ]) )

            ffs.append ( this_f )

        return DNNApprox.create ( feature_functions = ffs, dnn_spec = self.dnn_spec, adam_gradient = adam_gradient )

    def get_states_distribution ( self, t: int ) -> SampledDistribution [ NonTerminal [ float ] ]:

        actions_distr: Choose [ float ] = self.uniform_actions ( )

        def states_sampler_func ( ) -> NonTerminal [ float ]:
            wealth: float = self.initial_wealth
            for i in range ( t ):
                distr: Distribution [ float ] = self.mdp.riskyDist
                rate: float = self.mdp.riskfreeRate
                alloc: float = actions_distr.sample ( )
                wealth = wealth * ( alloc * ( 1 + distr.sample ( ) ) + ( 1.0 - alloc ) * ( 1 + rate ) )
            return NonTerminal ( wealth )

        return SampledDistribution ( states_sampler_func )

    def backward_induction_qvf ( self ) -> Iterator [ QValueFunctionApprox [ float, float ] ]:

        init_fa: DNNApprox [ Tuple [ NonTerminal [ float ], float ] ] = self.get_qvf_func_approx ( )

        mdp_f0_mu_triples: Sequence [ Tuple [ MarkovDecisionProcess [ float, float ], DNNApprox [ Tuple [ NonTerminal [ float ], float ] ], SampledDistribution [ NonTerminal [ float ] ] ] ] = [ (self.mdp, init_fa, self.get_states_distribution ( i )) for i in range ( self.mdp.steps ) ]

        num_state_samples: int = 300
        error_tolerance: float = 1e-6

        return back_opt_qvf ( mdp_f0_mu_triples = mdp_f0_mu_triples, γ = 1.0, num_state_samples = num_state_samples, error_tolerance = error_tolerance )

    def get_vf_func_approx ( self, ff: Sequence [ Callable [ [ NonTerminal [ float ] ], float ] ] ) -> DNNApprox [ NonTerminal [ float ] ]:

        adam_gradient: AdamGradient = AdamGradient ( learning_rate = 0.1, decay1 = 0.9, decay2 = 0.999 )
        return DNNApprox.create ( feature_functions = ff, dnn_spec = self.dnn_spec, adam_gradient = adam_gradient )

    def backward_induction_vf_and_pi ( self, ff: Sequence [ Callable [ [ NonTerminal [ float ] ], float ] ] ) -> Iterator [ Tuple [ ValueFunctionApprox [ float ], DeterministicPolicy [ float, float ] ] ]:

        init_fa: DNNApprox [ NonTerminal [ float ] ] = self.get_vf_func_approx ( ff )

        mdp_f0_mu_triples: Sequence [ Tuple [ MarkovDecisionProcess [ float, float ], DNNApprox [ NonTerminal [ float ] ], SampledDistribution [ NonTerminal [ float ] ] ] ] = [ (self.mdp, init_fa, self.get_states_distribution ( i )) for i in range ( self.mdp.steps ) ]

        num_state_samples: int = 300
        error_tolerance: float = 1e-8

        return back_opt_vf_and_policy ( mdp_f0_mu_triples = mdp_f0_mu_triples, γ = 1.0, num_state_samples = num_state_samples, error_tolerance = error_tolerance )


if __name__ == '__main__':
    
    w0 : float = 100
    a : float = 1.2
    b : float = 0.9
    p : float = 0.45
    utilityAlpha : float = 1.0
    steps : float = 10
    totalSample : int = 1000
    max_episode_length : int = 200    
    epsilon_as_func_of_episodes: Callable [ [ int ], float ] = lambda k: k ** -0.5
            
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
        
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
        
    solver : TDMDPAssetAllocationSol = TDMDPAssetAllocationSol( w0, a, b, p, utilityAlpha, steps, totalSample, max_episode_length, epsilon_as_func_of_episodes, feature_funcs, dnn )
    it_qvf : Iterator[QValueFunctionApprox[float, float]] = solver.backward_induction_qvf()

    print("Backward Induction on Q-Value Function")
    print("--------------------------------------")
    print()
    for t, q in enumerate(it_qvf):
        print(f"Time {t:d}")
        print()
        opt_alloc: float = max(
            ((q((NonTerminal(init_wealth), ac)), ac) for ac in solver.mdp.actions( )), key=itemgetter(0)
        )[1]
        val: float = max(q((NonTerminal(init_wealth), ac)) for ac in solver.mdp.actions( ))
        print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            pprint(wts.weights)
