import random
from dataclasses import dataclass

from src.rl_lib.distribution import FiniteDistribution, Mapping


@dataclass ( frozen = True )
class StockReturn ( FiniteDistribution [ float ] ) :
    '''
    A distribution with two outcomes.
    Returns a with probability p and b with probability 1 - p.
    '''
    p: float
    a: float
    b: float

    def sample ( self ) -> float :
        return self.a if random.uniform ( 0, 1 ) <= self.p else self.b

    def table ( self ) -> Mapping [ bool, float ] :
        return { True : self.p, False : 1 - self.p }

    def probability ( self, outcome: bool ) -> float :
        return self.p if outcome else 1 - self.p
