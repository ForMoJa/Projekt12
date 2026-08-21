import scipy
import numpy
import itertools
from functools import reduce
from operator import getitem
from fractions import Fraction
import copy

def _powerset(s):
	'''
	S is some arbitrary set
	
	returns: The powerset
	'''
	x = len(s)
	P = []
	for i in range(1 << x):
		P.append([s[j] for j in range(x) if (i & (1 << j))])
	return P

def get_models(B):
	'''
	B is an array
	
	returns all ordered partitions of B
	'''
	A = []
	while(len(B) != 0):
		C = B[-1]
		D = C[-1]
		B.pop()
		C.pop()
		for i in _powerset(D):
			if(len(i) != 0):
				if(len(i) == len(D)):
					C2 = copy.deepcopy(C)
					C2.append(D)
					A.append(C2)
				else:
					E = copy.deepcopy(D)
					for j in i:
						E.remove(j)
					C2 = copy.deepcopy(C)
					C2.append(i)
					C2.append(E)
					B.append(C2)
	return A

def turn_model_into_deg_array(m, n):
    '''
	m is a model
    n is the number of players
	
	returns: Tranlated model m into a model which only carries degree
	
	attention: the base set of the model is expected to be of the form [1,..,n] 
	'''
    
    a = []
    done = False
    for i in range(1,n+1):
        done = False
        for j in range(0,len(m)):
            if(done):
                break
            for k in range(0,len(m[j])):
                if(m[j][k]==i):
                    a.append(j+1)
                    done = True
                    break
    return a

def get_deg_models(n):
    '''
    n is the number of players

    combines get_model and turn_model_into_deg_array
    '''
    s = [k for k in range(1, n+1)]
    M = get_models([[s]])
    MD = []
    for m in M:
        MD.append(turn_model_into_deg_array(m,n))
    return MD

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

def solve_inequalites(X, m, maxD, E, vanishing_marginals):
    '''
	X pay-off matrices
    m is given model
	maxD maximal degree of the model
	E is a list where E[i] is the expected pay-off of player i
    '''
    n = len(vanishing_marginals)
    l = 1 # lower bound
    
    for d in range(1, maxD+1): # solving a LP for each of the degrees
        # base = complement_p_supp(S, p)
        J = []
        # for s in S:
        for s in m.keys():
            if m[s] == d:
                J.append(s)

        print(f"d = {d}, J = {J}")

        # weight vector is simply zeros; 
        # optimzing constant functino is equivlanet to checking whether feasible set is empty
        c = numpy.zeros(len(J)) 
        b = numpy.zeros(n)

        A = numpy.zeros((n,len(J)))
        for k, si in enumerate(vanishing_marginals):
            for l, s in enumerate(J):
                if si[1] == s[si[0]]:
                    A[k][l] = reduce(getitem, s, X[si[0]]) - E[si[0]] # type: ignore

        LP_res = scipy.optimize.linprog(c, A_ub = A, b_ub = b, bounds = (l, None))
        status = LP_res.status 
        x = LP_res.x
        print(x)
        if status != 0:
            return False

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

    M = get_deg_models(len(T))
    dic = {}
    for m in M:
        maxd = 0
        for i in range(0, len(T)):
            dic[T[i]] = m[i]
            maxd = max(maxd, m[i])
        if solve_inequalites(X, dic, maxd, expected_val, vanishing_marginals):
            print("Model found: ", m)
            return True
    return False
	
def bach_strawinsky():
	X=[
    [[2,0],
    [0,3]],

    [[3,0],
    [0,2]]
    ]
	return X 

def prisoners_dilema(): 
	X=[
    [[-1,-3],
    [0,-2]],

    [[-1,0],
    [-3,-2]]
    ]
	return X
	

if __name__ == "__main__":

    S = (list(itertools.product([0, 1], [0, 1])))

    X = bach_strawinsky()

    p = [
    [Fraction(1,1),Fraction(0,1)],
    [Fraction(0,1),Fraction(0,1)]
    ]

    if test_DE(X, p, [2,2]):
        print("The probability distribution ", p," is a DE!")
        print("for game ", X)
    else:
        print("The probability distribution ", p," is NOT a DE!")
        print("for game ", X)

    #print(calculate_expectency(X[0], S, p, 0))