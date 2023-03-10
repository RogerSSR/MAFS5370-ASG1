# MAFS5370-ASG1

Discrete-Time Asset-Allocation Case Description

Consider the discrete-time asset allocation example in section 8.4 of Rao and Jelvis. Suppose the single-time-step
return of the risky asset from time t to t+1 as $ Y_t = a, prob = p, and b, prob = ( 1 - p ) $. Suppose that T = 10, use
the TD
method to find the Q function, and hence the optimal strategy.

we formulate the problem as a continuous states and continuous actions discrete-time finite-horizon MDP

At each of discrete time steps labeled t = 0, 1, . . . , T − 1, we are allowed to allocate the wealth Wt at time t to a
portfolio of a risky asset and a riskless asset in an unconstrained manner with no transaction costs. The risky asset
yields a random return ∼ N (µ, σ2 ) over each single time step (for a given µ ∈ R and a given σ ∈ R +). The riskless
asset yields a constant return denoted by r over each single time step (for a given r ∈ R). We assume that there is no
consumption of wealth at any time t < T, and that we liquidate and consume the wealth WT at time T. So our goal is
simply to maximize the Expected Utility of Wealth at the final time step t = T by dynamically allocating xt ∈ R in the
risky asset and the remaining Wt − xt in the riskless asset for each t = 0, 1, . . . , T − 1. Assume the
single-time-step discount factor is γ and that the Utility of Wealth at the final time step t = T is given by the
following CARA function:





