import itertools
import unittest

from src.rl_lib.distribution import Categorical
from src.tdMDPAssetsAllocationSol import AssetAllocationMDP


class TestMDPFormation ( unittest.TestCase ):
    def setUp ( self ) -> None:
        self.mdp = AssetAllocationMDP ( w0 = 1000, a = 1.1, b = 0.93, p = 0.5, utilityAlpha = 1.0, steps = 10, ttlSample = 10000, )

    def testRiskyAssetReturn ( self ):
        trace = list ( itertools.islice ( self.mdp.riskyDist, self.mdp.steps ) )

        self.assertTrue ( all ( isinstance ( outcome.sample ( ), float ) for outcome in trace ) )
        self.assertTrue ( all ( isinstance ( outcome, Categorical ) for outcome in trace ) )
        self.assertTrue ( all ( len ( outcome.probabilities ) is 2 for outcome in trace ) )
        self.assertTrue ( all ( 1.1 in outcome.probabilities for outcome in trace ) )

        self.assertTrue ( )
