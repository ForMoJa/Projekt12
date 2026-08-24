import copy
import itertools
import numpy as np
import scipy
from functools import reduce
from fractions import Fraction
from operator import getitem

def get_deg_models(n):
    A = [[0]*n]
    while len(A) != 0:
        c = A.pop()
        ma = 0
        o = []
        for i in range(0, n):
            ma = max(ma, c[i])
            if c[i] == 0:
                o.append(i)
        m = len(o)
        for i in range(1,(1 << m)-1):
            P = []
            for j in range(m):
                if(i & (1<<j)):
                    P.append(o[j])
            d = copy.deepcopy(c)
            for j in range(0, len(P)):
                d[P[j]] = ma+1
            A.append(d)
        d = copy.deepcopy(c)
        for k in o:
            d[k] = ma+1
        yield d

def complement_p_supp(S, p):
    '''
    S is a set of strategies 
    p is a probability distribution

    returns: all pure strategies in S which non-zero proability
    '''
    L = []
    for s in S:
        if reduce(getitem, s, p) == 0:
            L.append(s)
    return L

def p_si(S, p, i, si):
    '''
    S is the set of strategies
    p is a probability distribution
    i is a player 
    si is a given strategy of player i

    return the marginal probability of player i choosing strategy si
    '''
    value = 0
    for s in S:
        if si != s[i]:
            continue
        value += reduce(getitem, s, p) # type: ignore
    return value 

def calculate_expectency(Xi, S, p, i, si=None):
    ''' 
    X is pay-off matrices of player i
    S is set of strategies
    p is distribution
    i is player
    si is given strategy with p(si) > 0 (optional)
    '''
    expected = 0
    if si is not None:
        if p_si(S,p,i,si) == 0:
            raise ValueError('p(s_i) > 0 for the expected value E_(s_i) to be defined.')
        for s in S: 
            if si == s[i]:
                expected += reduce(getitem, s, Xi)*(reduce(getitem, s, p) / p_si(S, p, i, si)) # type: ignore
    else:
        for s in S:
            expected += reduce(getitem, s, Xi) * reduce(getitem, s, p) # type: ignore
    return expected     

def solve_inequalites(X, m, maxD, E, vanishing_marginals, verbose=False):
    '''
	X pay-off matrices
    m is given model
	maxD maximal degree of the model
	E is a list where E[i] is the expected pay-off of player i
    '''
    n = len(vanishing_marginals)

    if verbose:
        print(f"Testing model m={m}")

    min_deg = [maxD]*len(vanishing_marginals)
    # for i in range(vanishing_marginals):
        # min_deg[i]=maxD

    for j in range(len(vanishing_marginals)):
        si = vanishing_marginals[j]
        for s in m.keys():
            if s[si[0]] == si[1]:
                min_deg[j] = min(min_deg[j], m[s])
    
    for d in range(1, maxD + 1): # solving a LP for each of the degrees
        J = []
        for s in m.keys():
            if m[s] == d:
                J.append(s)
        if verbose:
            print(f"d={d}, J ={J}")

        # weight vector is simply zeros; 
        # optimzing constant function is equivlanet to checking whether feasible set is empty
        c = np.zeros(len(J)) 
        b = np.zeros(n)

        A = np.zeros((n, len(J)))
        for k, si in enumerate(vanishing_marginals):
            for l, s in enumerate(J):
                if si[1] == s[si[0]] and min_deg[k] == d:
                    A[k][l] = reduce(getitem, s, X[si[0]]) - E[si[0]] # type: ignore

        LP_res = scipy.optimize.linprog(c, A_ub = A, b_ub = b, bounds = (1, None))
        status = LP_res.status 
        x = LP_res.x
        if status != 0:
            return False
        if verbose:
            print(f"Found solution at d={d}, namely x={x}")

    return True

def get_S(D):
    '''
    D is a tuple of integers

    returns the set of strategies given a finite number of strategies D[i] player i can choose
    '''
    if len(D) == 1 :
        return [k for k in range(D[0])]
    d = D[-1]
    D.pop()
    return list(itertools.product([k for k in range(0, d)], get_S(D)))

def test_DE(X, p, D):
    '''
    X is a game
    p is a probability distribution
    D is a tuple of pure strategies of the player

    return whether p is DE in game X
    '''

    E = copy.deepcopy(D)
    S = get_S(E)
    print("strategies: ", S)
    print("number of pure strategies for each player: ", D)

    expected_val = []
    for i in range(0, len(D)):
        expected_val.append(calculate_expectency(X[i], S, p, i))
        
    print("Expected value for each player: ", expected_val)
    marginals = []
    vanishing_marginals = []
    for i in range(0, len(D)):
        for j in range(0, D[i]):
            if(p_si(S, p, i, j) > 0):
                marginals.append([i, j])
            else:
                vanishing_marginals.append([i, j])
    print("Marginals: ")
    print(marginals)
    print("\nVanishing Marginals: ")
    print(vanishing_marginals)

    # check whether p lies on the Spohn variety; if not, p cannot possible be a DE    
    for si in marginals:
        if expected_val[si[0]] != calculate_expectency(X[si[0]], S, p, si[0], si[1]):
            print("p is not on the Spohn variety!")
            print("p is not DE by s_i=", si)
            print(expected_val[si[0]],"!=",calculate_expectency(X[si[0]], S, p, si[0], si[1]))
            return False
        
    T = complement_p_supp(S, p)
    # if all marginals of p are defined, lying on the Spohn variety is already a sufficient condition -> return True
    if len(vanishing_marginals) == 0:
        print("Lies on the Spohn ideal and all marginals defined.")
        return True

    dic = {}
    for m in get_deg_models(len(T)):
        # print(m)
        maxd = 0
        for i in range(0, len(T)):
            dic[T[i]] = m[i]
            maxd = max(maxd, m[i])
        # print(dic)
        if solve_inequalites(X, dic, maxd, expected_val, vanishing_marginals):
            print("Model found: ", m)
            return True
    return False
	
def bach_strawinsky():
    '''
    returns the classic game of Bach--Strawinsky
    '''

    X = [
    [[2,0],
    [0,3]],

    [[3,0],
    [0,2]]
    ]
    return X 

def prisoners_dilema(): 
    '''
    returns the classic game of `Prisoner's dilemma' 
    '''
    X = [
    [[-1,-3],
    [0,-2]],

    [[-1,0],
    [-3,-2]]
    ]
    return X
	

if __name__ == "__main__":

    S = (list(itertools.product([0, 1], [0, 1])))

    X = prisoners_dilema()

    p = [
    [Fraction(0,1),Fraction(1,1)],
    [Fraction(0,1),Fraction(0,1)]
    ]

    if test_DE(X, p, [2,2]):
        print(f"=============")
        print(f"Returns True")
        print(f"The probability distribution")
        print(f"   p={p}")
        print(f"is a DE for game")
        print(f"   X={X}")
    else:
        print(f"=============")
        print(f"Returns False")
        print(f"The probability distribution")
        print(f"   p={p}")
        print(f"is NOT a DE for game")
        print(f"   X={X}")

    #print(calculate_expectency(X[0], S, p, 0))