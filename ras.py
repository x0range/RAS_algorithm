import numpy as np
#import pdb

def ras(m, in_marginals, out_marginals, diagonal_zero=True, max_iterations=1000, convergence_threshold=0.0000001):
    """Compute RAS algorithm for maximum entropy generation of I-O-matrices under
        restrictions of column and row sums.
    Arguments:
        m: numpy matrix or None - Matrix initialization. Will be drawn from uniform if None.
        in_marginals: numpy 1 x n matrix
        out_marginals: numpy n x 1 matrix 
        diagonal_zero: bool - Impose zero main diagonal condition (x_ij = 0 for i=j)
        max_iterations: int
        
    Returns:
        numpy matrix"""
    
    """Assert consistent dimensionality of marginals"""
    assert in_marginals.shape[0] == out_marginals.shape[1] == 1
    assert in_marginals.shape[1] == out_marginals.shape[0]
    n = out_marginals.shape[0]
    
    """Assert consistency of input and output sums"""
    assert np.sum(in_marginals) == np.sum(out_marginals)
    
    if m is None:
        """Initialize matrix"""
        m = np.matrix(np.random.uniform(0, 1, n**2).reshape(n, n))
    else:
        """Assert consistent dimensionality of initial matrix"""
        assert m.shape[0] == m.shape[1] == n
    
    """Assert non-negativity"""
    assert (m>=0).all()
    assert (in_marginals>=0).all()
    assert (out_marginals>=0).all()
    
    """Set additional diagonal zero condition"""
    if diagonal_zero:
        np.fill_diagonal(m, 0)
    
    """Iterative computation"""
    for i in range(max_iterations):
        m_old = m.copy()
        
        """Firts part, w.r.t. row marginals"""
        row_sums = np.sum(m, axis=1)
        col_factor = np.divide(out_marginals, row_sums, out=np.zeros_like(out_marginals.astype('float')), where=out_marginals!=0)
        m = np.multiply(m, col_factor)
        
        m_first = m.copy()
        
        """Second part, w.r.t. column marginals"""
        col_sums = np.sum(m, axis=0)
        row_factor = np.divide(in_marginals, col_sums, out=np.zeros_like(in_marginals.astype('float')), where=in_marginals!=0)
        m = np.multiply(row_factor, m)
        
        """Test convergence"""
        if (np.max(np.abs(m - m_old)) < convergence_threshold) and \
               (np.max(np.abs(m - m_first)) < convergence_threshold):
            return m
        #print(m_first)
        #print(m)
        #pdb.set_trace()
        
        if np.isinf(m).any():
            break
        
    """If this point is reached, the algorithm has failed to converge in the given number of iteration steps"""
    print("Algorithm has failed to converge")
    return None

def create_core_periphery_matrix(core_size, matrix_size, density=0.2, maximum_entropy_optimized=True):
    """Function to create a random core periphery matrix.
        Arguments:
            core_size: int - Number of nodes in the core
            matrix_size: int - Number of nodes in the matrix
            density: float \in (0,1) - density of core periphery links
            maximum_entropy_optimized: bool - return maximum entropy optimized exposure matrix (or a random one)
        Returns:
            Tuple of
                numpy 1 x matrix_size int matrix - input marginals
                numpy matrix_size x 1 int matrix - output marginals
                numpy matrix_size x matrix_size float matrix - exposure matrix (weighted adjacency matrix), Either
                                                               maximum entropy optimized with RAS algorithm or random
                                                               depending on argument.
                numpy matrix_size x matrix_size int \in (0, 1) matrix - unweighted adjacency matrix
        """
    
    """Define core periphery parameters"""
    periphery_size = matrix_size - core_size
    
    """Random marginal generation"""
    out_marginals = np.matrix(np.random.randint(0, 100, matrix_size)).transpose()
    in_marginals = np.matrix(np.random.randint(0, 100, matrix_size))
    """Increase magnitude of core"""
    out_marginals[:core_size, :] += np.matrix(np.random.randint(0, 900, core_size)).transpose()
    in_marginals[:, :core_size] += np.matrix(np.random.randint(0, 900, core_size))
    
    """Create initial matrix to reflect core-periphery structure"""
    """Create blocks. This gives 4 matrices."""
    core_block = np.ones(shape=(core_size, core_size))
    core_periphery_block_right = np.random.choice([0, 1], p=[1.-density, density], size=(core_size, periphery_size))
    core_periphery_block_left = np.random.choice([0, 1], p=[1.-density, density], size=(periphery_size, core_size))
    periphery_block = np.zeros(shape=(periphery_size, periphery_size))
    """Combine blocks. This gives one matrix with 0 and 1 entries."""
    adj_unweighted = np.matrix(np.block([[core_block, core_periphery_block_right], \
                                         [core_periphery_block_left, periphery_block]]))
    """Add randomness"""
    random_matrix = np.matrix(np.random.uniform(0, 1, matrix_size**2).reshape(matrix_size, matrix_size))
    adj_weighted = np.multiply(adj_unweighted, random_matrix)
    
    """Set periphery in and out marginals to zero where this is implied by the structure of the initial matrix, 
        i.e. where row or column sums in the initial matrix are zero. Alternatively, elements could be added to 
        the matrix to avoid zero row and column sums for these periphery elements"""
    out_marginals = np.multiply(out_marginals, np.sign(np.sum(adj_unweighted, axis=1)))
    in_marginals = np.multiply(in_marginals, np.sign(np.sum(adj_unweighted, axis=0)))
    
    """Balance in and out marginals"""
    if np.sum(in_marginals) > np.sum(out_marginals):
        out_marginals[0, 0] += np.sum(in_marginals) - np.sum(out_marginals)
    else:
        in_marginals[0, 0] += np.sum(out_marginals) - np.sum(in_marginals)
    
    convergence = None
    
    """Optimize if flag is set"""
    if maximum_entropy_optimized:
        maximum_entropy_result = ras(adj_weighted, in_marginals, out_marginals)
        if maximum_entropy_result is not None:
            adj_weighted = maximum_entropy_result
            convergence = True
        else:
            print("Maximum entropy matrix generation failed. Returning initial random matrix.")
            convergence = False
    return in_marginals, out_marginals, adj_weighted, adj_unweighted, convergence
    

if __name__ == "__main__":
    """Examples"""
    np.set_printoptions(suppress=True)
    
    print("\nFixed 2x2 matrix without constraint on main diagonal")
    print(ras(None, np.matrix([9, 11]), np.matrix([17, 3]).transpose(), False))
    print("\nFixed 2x2 matrix with zero constraint on main diagonal")
    print(ras(None, np.matrix([9, 11]), np.matrix([17, 3]).transpose(), True))

    print("\nRandom 4x4 matrix with and without diagonal constraint")
    """Random marginal generation"""
    out_marginals = np.matrix(np.random.randint(0, 100, 4)).transpose()
    in_marginals = np.matrix(np.random.randint(0, 100, 4))
    """Balance in and out marginals"""
    if np.sum(in_marginals) > np.sum(out_marginals):
        out_marginals[0, 0] += np.sum(in_marginals) - np.sum(out_marginals)
    else:
        in_marginals[0, 0] += np.sum(out_marginals) - np.sum(in_marginals)
    print(ras(None, in_marginals, out_marginals, False))
    print(ras(None, in_marginals, out_marginals, True))

    if False:
        print("\nRandom 3x3 matrix with pure input and pure output nodes, with and without diagonal constraint")
        """Random marginal generation"""
        out_marginals = np.matrix(np.random.randint(0, 100, 3)).transpose()
        in_marginals = np.matrix(np.random.randint(0, 100, 3))
        """Set out marginals for pure input nodes to zero. Same for in marginals for pure output nodes."""
        in_marginals[0, 0] = 0
        out_marginals[2, 0] = 0
        """Balance in and out marginals"""
        if np.sum(in_marginals) > np.sum(out_marginals):
            out_marginals[0, 0] += np.sum(in_marginals) - np.sum(out_marginals)
        else:
            in_marginals[0, 2] += np.sum(out_marginals) - np.sum(in_marginals)
        print(ras(None, in_marginals, out_marginals, False))
        print(ras(None, in_marginals, out_marginals, True))

    print("\nRandom 7x7 matrix with pure input and pure output nodes, with and without diagonal constraint")
    """Random marginal generation"""
    out_marginals = np.matrix(np.random.randint(0, 100, 7)).transpose()
    in_marginals = np.matrix(np.random.randint(0, 100, 7))
    """Set out marginals for pure input nodes to zero. Same for in marginals for pure output nodes."""
    in_marginals[[0, 0], [0, 1]] = 0
    out_marginals[[5, 6], [0, 0]] = 0
    """Balance in and out marginals"""
    if np.sum(in_marginals) > np.sum(out_marginals):
        out_marginals[0, 0] += np.sum(in_marginals) - np.sum(out_marginals)
    else:
        in_marginals[0, 6] += np.sum(out_marginals) - np.sum(in_marginals)
    print(ras(None, in_marginals, out_marginals, False))
    print(ras(None, in_marginals, out_marginals, True))

    print("\nRandom 30x30 matrix with core-periphery structure, with and without diagonal constraint")
    core_size = 10
    matrix_size = 30
    in_marginals, out_marginals, m_init, _, __ = create_core_periphery_matrix(core_size, \
                                                                          matrix_size, \
                                                                          maximum_entropy_optimized=False)
    print(ras(m_init, in_marginals, out_marginals, False))
    print(ras(m_init, in_marginals, out_marginals, True))

    print("\nCore-periphery structure function can also handle maximum entropy optimization of the matrix.")
    in_marginals, out_marginals, m_init, _, convergence = create_core_periphery_matrix(core_size, \
                                                                                       matrix_size, \
                                                                                       maximum_entropy_optimized=True)
    if not convergence:
        print("But in this case, the algorithm has failed to converge. Will probably fail again.")
    else:
        print("Matrix is:")
        print(m_init)
    print("In this case, rerunning the RAS algorithm will return always the same matrix since it is already",
                                                                                                 "converged:")
    ras_result = ras(m_init, in_marginals, out_marginals, True)
    if ras_result is None:
        print("Algorithm has failed to converge.")
    else:
        if convergence:
            print("Matrices are identical: ", np.isclose(m_init, ras_result).all())
        else:
            print("Algorithm converged. Matrix is: ")
            print(ras_result)
    
