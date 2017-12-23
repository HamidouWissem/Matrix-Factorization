import numpy as np

def objective(P, Q, R, mask, rho):
    """
    The objective function of the complete problem 
    Input :
    P : Matrix of size C x I
    Q : Matrix of size U x C
    R : Matrix of size U x I
    mask : Matrix 0-1 of size U x I
    rho : positif real 

    Output :
    val : value of the function 
    grad_P : gradient P
    grad_Q : gradient  Q
    """
    r = (R - Q.dot(P)) * mask

    val = np.sum(r ** 2)/2. + rho /2. * (np.sum(Q ** 2) + np.sum(P ** 2))

    grad_P = -np.dot(Q.T,r) + rho * P

    grad_Q = -np.dot(r, P.T) + rho * Q

    return val, grad_P, grad_Q


def objective_Q(P0, Q, R, mask, rho):
    """
    This function returns two values : 
        -The value of the objective function at P0 fixed 
        -The value of the gradient of Q 
    """
    val,_ ,grad_Q = objective(P0, Q, R, mask, rho)

    return (val, grad_Q)


def objective_P(P, Q0, R, mask, rho):
    """
    This function returns two values : 
        -The value of the objective function at Q0 fixed 
        -The value of the gradient of P
    """
    val, grad_P,_ = objective(P, Q0, R, mask, rho)

    return (val, grad_P)