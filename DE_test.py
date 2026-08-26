import copy
import numpy as np
import scipy
from functools import reduce
from fractions import Fraction
from operator import getitem

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

def get_S(D):
    '''
    D is a tuple of integers

    returns the set of strategies given a finite number of strategies D[i] player i can choose
    '''
    if len(D) == 1 :
        return [[k] for k in range(D[0])]
    d = D[-1]
    D.pop()
    G=get_S(D)
    H=[]
    for i in G:
        for k in range(d):
            H.append([k]+i)
    return H
	
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

def prisoners_dilemma(): 
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

def track_game():
    X = [
    [
        [0,1],
        [0,-1]
    ],
    [
        [-2,0],
        [1,-1]
    ]
    ]

    return X
	
def bigger_prisoners_dilemma():
	X = [
        [
            [[-1,-2],
            [0,0]],

            [[-2,-3],
            [0,-2]]
        ],
	
	    [
            [[-1,0],
	        [-2,0]],
              
	        [[-2,0],
	        [-3,-2]]
        ],
	
	    [
            [[-1,-2],
	        [-2,-3]],

            [[0,0],
	        [0,-2]]
        ]
	]
	return X

def test_DE(X, p, D, verbose=False):
	'''
    X is a game
    p is a probability distribution
    D is a tuple of pure strategies of the player

	return whether p is DE in game X
	'''

	E = copy.deepcopy(D)
	S = get_S(E)
	if verbose:
		print("strategies: ", S)
		print("number of pure strategies for each player: ", D)

	expected_val = []
	for i in range(0, len(D)):
		expected_val.append(calculate_expectency(X[i], S, p, i))
        
	if verbose: 
		print("Expected value for each player: ", expected_val)

	# marginals are p(s_i) for some pure strategy s_i \in S_i
	marginals = []
	vanishing_marginals = []
	for i in range(0, len(D)):
		for j in range(0, D[i]):
			if(p_si(S, p, i, j) > 0):
				marginals.append([i, j])
			else:
				vanishing_marginals.append([i, j])
	if verbose:
		print("Marginals: ")
		print(marginals)
		print("\nVanishing Marginals: ")
		print(vanishing_marginals)

	# check whether p lies on the Spohn variety; if not, p cannot possible be a DE    
	for si in marginals:
		if expected_val[si[0]] != calculate_expectency(X[si[0]], S, p, si[0], si[1]):
			print("p is not on the Spohn variety!")
			print(f"p is not DE by marginal s_i={si}")
			print(expected_val[si[0]],"!=",calculate_expectency(X[si[0]], S, p, si[0], si[1]))
			return False
        
	T = complement_p_supp(S, p)
	# if all marginals of p are defined, lying on the Spohn variety is already a sufficient condition 
	if len(vanishing_marginals) == 0:
		print("p lies on the Spohn ideal and all marginals defined.")
		return True
    
	max_dim = len(vanishing_marginals) + 1
	M = [[[max_dim]*len(T), vanishing_marginals]] # Starting with empty model
	while len(M) != 0:
		m_2 = M.pop()
		model = m_2[0]
		vm = m_2[1] # currently vanishing marginals
        
		unsupp = []
		deg = 1
		for i in range(len(model)):
			if model[i] == max_dim:
				unsupp.append(i)
			else:
				deg = max(deg, model[i] + 1)
		x = len(unsupp)

		# Consider all possible next layers
		y = 1 << x
		for index in range(1, y):
			index2 = y - index
			newdeg = [unsupp[j] for j in range(0, len(unsupp)) if (index2 & (1 << j))]
			newm = copy.deepcopy(model)
			for k in newdeg:
				newm[k] = deg
			marginals_resolved = []
			vm_left = []
			marginals = []
			for d in D:
				marginals.append([0]*d)
			for j in newdeg:
				for k in range(len(D)):
					marginals[k][T[j][k]] += 1
			for v in vm:
				if marginals[v[0]][v[1]] != 0:
					marginals_resolved.append(v)
				else:
					vm_left.append(v)
			if len(marginals_resolved) == 0:
				continue
			global count_polytopes
			count_polytopes += 1
			# Check polytope
			n = len(marginals_resolved) # Number of inequalities
			m = len(newdeg) # Number of variables

			# weight vector is simply zeros; 
			# optimzing constant function is equivlanet to checking whether feasible set is empty
			c = np.zeros(m) 
			b = np.zeros(n)

			A = np.zeros((n, m))
			J = []
			for k in newdeg:
				J.append(T[k])
                        
			for k, margres in enumerate(marginals_resolved):
				for l, s in enumerate(J):
					if margres[1] == s[margres[0]]:
						A[k][l] = reduce(getitem, s, X[margres[0]]) - expected_val[margres[0]] # type: ignore

			LP_res = scipy.optimize.linprog(c, A_ub = A, b_ub = b, bounds = (1, None))
			status = LP_res.status 
			x = LP_res.x
			if status == 0:
				if len(vm_left) != 0:
					M.append([newm,vm_left])
				else:
					print("\nModel found: ", m)
					print(T)
					return True

if __name__ == "__main__":
    global count_polytopes
    count_polytopes=0
    # X = bigger_prisoners_dilemma()

    D = [3,3]

    X = [
    [[1,1,0],
    [1,0,1],
    [0,1,1]],
    
    [[1,1,0],
    [1,0,1],
    [0,1,1]],
    ]

    p = [
    [Fraction(0,1),Fraction(0,1),Fraction(1,1)],
    [Fraction(0,1),Fraction(0,1),Fraction(0,1)],
    [Fraction(0,1),Fraction(0,1),Fraction(0,1)]
    ]

    if len(X) != len(D) and len(X[0]) != len(p) and len(X[0][0]) != len(p[0]):
        print("Dimensions do not match; You're stupid")
        exit(1)

    S = get_S([3,3])
    print(f"S = {S}")

    if test_DE(X, p, [3,3]):
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
    print("Checked ", count_polytopes," polytopes for being empty.")