import numpy as np
import exactRP

def prim_from_q(q,gamma=1.4):
    """
    Compute and return the primitive variables from state vector array q.
    The first index of q[:,:] is the spatial (1D) index, while the second
    index is 0, 1 or 2, meaning rho, rho*u, rho*etot. 
    """
    rho       = q[:,0]
    assert rho.min()>=0, 'Error: Negative density detected'
    u         = q[:,1]/rho
    etot      = q[:,2]/rho
    ekin      = 0.5*u**2
    eth       = etot-ekin
    assert eth.min()>=0, 'Error: Negative thermal energy detected'
    p         = (gamma-1) * rho * eth
    csadi     = np.sqrt( gamma * p / rho )
    return rho,u,eth,etot,p,csadi
    
def flux_from_prim(rho,u,etot,p):
    """
    Compute and return flux from a set of primitive variables, each being
    a 1D array. The first index of the resulting flux flux[:,:] is again
    the spatial dimension, while the second index is 0, 1 or 2, meaning
    rho*u, rho*u*u+p, rho*htot*u. 
    """
    flux      = np.zeros((rho.shape[0],3))
    flux[:,0] = rho*u
    flux[:,1] = rho*u*u + p
    flux[:,2] = ( rho*etot + p ) * u
    return flux

def flux_hll(q,gamma=1.4,dx=None,dt=None,slope=None):
    """
    The interface flux for the Harten, Lax, van Leer (HLL) Riemann solver.
    The q[:,:] is the state vector (see prim_from_q() above), and the function
    returns flux[:,:] which has spatial coordinate of length nx+1, i.e. the
    flux[1,:] is the flux between q[0,:] and q[1,:]. So, for nx=100, q has
    [100,3] elements and flux has [101,3] elements.
    """
    
    # Left and right cells
    q_L       = q[:-1].copy()
    q_R       = q[1:].copy()

    if slope is not None:
        print('Slope limiter not yet implemented. Exercise: Implement it here using the MUSCL-Hancock algorithm.')

    # Update everything before calling the Riemann solver
    rho_L,u_L,eth_L,etot_L,p_L,csadi_L = prim_from_q(q_L,gamma=gamma)
    rho_R,u_R,eth_R,etot_R,p_R,csadi_R = prim_from_q(q_R,gamma=gamma)
    f_L       = flux_from_prim(rho_L,u_L,etot_L,p_L)
    f_R       = flux_from_prim(rho_R,u_R,etot_R,p_R)
    
    # HLL Algorithm
    s_L       = np.minimum( u_L - csadi_L , u_R - csadi_R )  # lambda_minus
    s_R       = np.maximum( u_L + csadi_L , u_R + csadi_R )  # lambda_plus
    flux      = np.zeros((q.shape[0]+1,q.shape[1]))
    flvw      = flux[1:-1,:]  # Ignore the left and rightmost fluxes for now
    ii        = (s_L>=0)                           # Left state mask
    flvw[ii]  = f_L[ii]
    ii        = np.logical_and( s_R>0 , s_L<0 )    # Middle state mask
    flvw[ii]  = ( s_R[ii,None]*f_L[ii] - s_L[ii,None]*f_R[ii] + s_R[ii,None] * s_L[ii,None] *
                  ( q_R[ii] - q_L[ii] ) ) / ( s_R[ii,None] - s_L[ii,None] )
    ii        = (s_R<=0)                           # Right state mask
    flvw[ii]  = f_R[ii]
    return flux

def analytic_riemann_problem_solution(x,t,gamma,rho_l,u_l,p_l,rho_r,u_r,p_r):
    """
    For testing your hydrodynamics code, you can set up a Riemann problem,
    and compare the numerical results against the analytic solution computed
    by this function. x is the spatial grid, t is the time at which you want
    to compute the analytic solution, rho_l,u_l,p_l are the initial density, 
    velocity and pressure left of x=0, rho_r,u_r,p_r are the initial density, 
    velocity and pressure right of x=0.
    """
    stateL  = [rho_l,u_l,p_l]
    stateR  = [rho_r,u_r,p_r]
    s       = x/t
    rp      = exactRP.exactRP(gamma, stateL, stateR)
    success = rp.solve()
    exact_rho, exact_p, exact_v, exact_etherm, exact_cspd = rp.sample(s)
    return exact_rho, exact_p, exact_v, exact_etherm, exact_cspd
