import numpy as np
def question1():
    def standardize(A):
        # will give mean of zero and standard deviation of one.
        mean = np.mean(A, axis = 0)
        std = np.std(A, axis = 0)
        return (A - mean) / std[None,:]
    
    def varianceCovariance(A):
        M = np.mean(A, axis = 0)
        return (A - M).T.dot((A - M)) / (A.shape[0]-1)
    
    def sortedEigenValuesAndVectors(C):
        eVals, eVecs = np.linalg.eig(C)
        for eVec in eVecs: # Asserting that the eigen vectors each have norm of (very close to) 1.
            np.testing.assert_array_almost_equal(np.linalg.norm(eVec), 1.0)
        ePairs = [(np.abs(eVals[i]), eVecs[:,i]) for i in range(len(eVals))]
        ePairs.sort()
        ePairs.reverse()
        return ePairs
        
    A = np.random.multivariate_normal(np.array([2,3,0]), np.array([[10,7,5],[7,6,4],[5,4,3]]), 1000)
    A = standardize(A)
    C = varianceCovariance(A)
    ePairs = sortedEigenValuesAndVectors(C) # get the sorted eigen pairs (values & vectors)
    firstComponentVector = ePairs[0][1]
    secondComponentVector = ePairs[1][1]
    dotProduct = np.dot(firstComponentVector, secondComponentVector) # first.T * second
    np.testing.assert_array_almost_equal(dotProduct, 0) # test of orthogonality, dot == 0
    
    print('\First Component \n%s' %firstComponentVector)
    print('\Second Component \n%s' %secondComponentVector) 

question1()