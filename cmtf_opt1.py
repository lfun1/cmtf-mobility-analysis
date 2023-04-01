import warnings
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.cp_tensor import CPTensor, validate_cp_rank, cp_to_tensor, cp_normalize
from tensorly.decomposition._cp import initialize_cp

import scipy
from scipy import optimize

import numpy as np

def cmtf_opt(tensor, matrix, rank, init="svd", n_iter_max=100, tol=1e-6, normalize_factors=False):
    
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)

    # initialize values
    tensor_cp = initialize_cp(tensor, rank, init=init)
    
    # the coupled factor should be initialized with the concatenated dataset
    coupled_unfold = tl.concatenate((tl.unfold(tensor, 0), matrix), axis=1)
    coupled_init = initialize_cp(coupled_unfold, rank, init=init)
    tensor_cp.factors[0] = coupled_init.factors[0]
    
    V = tl.transpose(tl.lstsq(tensor_cp.factors[0], matrix)[0])
    
    # Create x0 = x_vec:
    t_vec = tensor_cp.factors
    m_vec = tl.tensor_to_vec(V)
    
    x_vec = vectorize_tensor(t_vec, m_vec, tensor, matrix, rank)
    
    # NCG optimization functions
        # f = cmtf_loss_func (may have to edit parameters)
        # x0 = initialized values for tensor_cp and V
        # fprime = cmtf_grad_func (may have to edit parameters)
        # args = tensor, matrix (additional fixed variables)
        # gtol = tol
        # maxiter=n_iter_max
    # output_opt = optimize.fmin_cg(cmtf_loss_func, x_vec, fprime=cmtf_grad_func, args=(tensor, matrix), gtol=tol, maxiter=n_iter_max)
    
    # NCG optimization
    output_opt = optimize.fmin_cg(cmtf_loss_func, x_vec, fprime=cmtf_grad_func, args=(tensor, matrix, rank), gtol=tol, maxiter=n_iter_max)
    
    # Newton-CG optimization
    # output_opt = optimize.fmin_ncg(cmtf_loss_func, x_vec, fprime=cmtf_grad_func, args=(tensor, matrix, rank), avextol=tol, maxiter=n_iter_max)
    
    # Reconstruct tensor_cp and V from vectorized form in output_opt
    tensor_cp, V = recon_vec_to_tensor(output_opt, tensor, matrix, rank)
    
    matrix_pred = CPTensor((None, [tensor_cp.factors[0], V]))

    if normalize_factors:
        tensor_cp = cp_normalize(tensor_cp)
        matrix_pred = cp_normalize(matrix_pred)

    return tensor_cp, matrix_pred


def vectorize_tensor(t_vec, m_vec, tensor, matrix, rank):
    """Vectorize factor matrices.
    
    t_vec : vectorized form of factor matrices for tensor
    m_vec : vectorized form of factor matrices for matrix
    
    NOTE: for inputted data, it seems that t_vec is an np.array with 3 elements (each element = np.array / 1 factor matrix A)
          for random data, t_vec is an np.array with all elements already vectorized (but only on the first iteration??)
    SO: code is now adjusted to accomodate random
    """
    
    N = tl.ndim(tensor) # dimension of tensor, number of factor matrices for tensor
    
    x_vec = tl.tensor_to_vec(t_vec[0])

    for a in range(1, N):
        x_vec = tl.concatenate((x_vec, tl.tensor_to_vec(t_vec[a])))

    x_vec = tl.concatenate((x_vec, m_vec))
    
    return x_vec


def recon_vec_to_tensor(x_vec, tensor, matrix, rank):
    """Takes in vectorized form of factor matrices (3 factor matrices A^i for tensor, 1 factor matrix V for matrix)
    and returns the tensor (in CP form) and factor matrix V.
    
    x_vec : vectorized form of factor matrices
    tensor : original tensor
    matrix : original matrix
    rank : rank
    """
    
    N = tl.ndim(tensor) # dimension of tensor, number of factor matrices for tensor
    
    # initialize tensor_cp
    tensor_cp = initialize_cp(tensor, rank, init='svd')
    idx_start = 0
    
    for i in range(N):        
        tensor_cp.factors[i] = tl.vec_to_tensor(x_vec[idx_start : idx_start + tensor.shape[i] * rank], 
                                                (tensor.shape[i], rank))
        idx_start = idx_start + tensor.shape[i] * rank
    
    V = tl.vec_to_tensor(x_vec[idx_start : ], (matrix.shape[1], rank))
    
    return tensor_cp, V


def cmtf_loss_func(x_vec, tensor, matrix, rank):
    """Computes value of loss function f.
    
    x_vec = vectorized np.array of tensor_cp, V
    """
    # Reconstruct tensor_cp and V from vectorized form
    tensor_cp, V = recon_vec_to_tensor(x_vec, tensor, matrix, rank)
    
    return (0.5 * tl.norm(tensor - cp_to_tensor(tensor_cp)) ** 2
            + 0.5 * tl.norm(matrix - cp_to_tensor((None, [tensor_cp.factors[0], V]))) ** 2)


def cmtf_grad_func(x_vec, tensor, matrix, rank, coupled_n=0):
    """Computes entire gradient of loss function f.
    
    x_vec = vectorized np.array of tensor_cp, V
    
    Note: coupled_n is in (n-1) where n is coupled mode in paper (to correspond with Python, TensorLy indexing)."""
    
    # Reconstruct tensor_cp and V from vectorized form
    tensor_cp, V = recon_vec_to_tensor(x_vec, tensor, matrix, rank)
    
    tensor_pred = cp_to_tensor(tensor_cp) # corresponds to Z
    N = tl.ndim(tensor) # N, dimension of tensor, number of factor matrices for tensor
    
    # compute all partial derivatives with respect to factor matrices of tensor (A^i)
    f1_deriv = tl.zeros(N+1, dtype='object')
    f2_deriv = tl.zeros(N+1, dtype='object')
    
    for ii in reversed(range(N)):
        
        # khatri_rao product (A^-i) skipping A for each factor matrix A
        kr_A = khatri_rao(tensor_cp.factors, skip_matrix=ii)
        
        # partial derivative f1
        f1_deriv[ii] = tl.matmul((tl.unfold(tensor_pred, ii) - tl.unfold(tensor, ii)), kr_A)
        
        # partial derivative f2
            # note: may need to make coupled_n an array if multiple matrices coupled in different modes
        if ii == coupled_n:
            f2_deriv[ii] = ( - tl.matmul(matrix, V) + tl.matmul(tl.matmul(tensor_cp.factors[coupled_n], tl.transpose(V)), V) )
            # NOTE: using A instead of kr_A so that dimensions match
        else:
            f2_deriv[ii] = 0
         
    # computing partial derivatives for factor matrices corresponding solely to matrix
    
    # khatri_rao product for all factor matrices corresponding to tensor (A)
    kr_all = khatri_rao(tensor_cp.factors)
    # coupled factor matrix for tensor and matrix
    A_n = tensor_cp.factors[coupled_n]
    
    # NOTE: not sure if A^(i) in paper refers to kr_all or just A^(n), ASSUMING it is A^(n)
    f2_deriv[N] = ( - tl.matmul(tl.transpose(matrix), A_n) + tl.matmul(tl.matmul(V, tl.transpose(A_n)), A_n) )
    f1_deriv[N] = 0
    
    # combining gradients
    f_deriv = f1_deriv + f2_deriv
    grad_f = tl.tensor_to_vec(f_deriv)
    
    grad_f_vec = vectorize_tensor(grad_f[0:-1], tl.tensor_to_vec(grad_f[-1]), tensor, matrix, rank)
    
    return grad_f_vec