import unittest

from src.rl_lib.distribution import Categorical
from src.rl_lib.markov_process import NonTerminal
from src.tdMDPAssetsAllocationSol import AssetAllocationMDP


class TestMDPFormation ( unittest.TestCase ):
    def setUp ( self ) -> None:
        self.mdp = AssetAllocationMDP ( a = 1.1, b = 0.93, p = 0.5, utilityAlpha = 1.0, steps = 10, t = 0, ttlSample = 10000 )

    def testRiskyAssetReturn ( self ):
        self.assertTrue ( isinstance ( self.mdp.riskyDist.sample ( ), float ) )
        self.assertTrue ( isinstance ( self.mdp.riskyDist, Categorical ) )
        self.assertTrue ( len ( self.mdp.riskyDist.probabilities ) is 2 )
        self.assertTrue ( 1.1 in self.mdp.riskyDist.probabilities )
        self.assertTrue ( len ( self.mdp.actions ( NonTerminal ( 1.0 ) ) ) is 20 )
