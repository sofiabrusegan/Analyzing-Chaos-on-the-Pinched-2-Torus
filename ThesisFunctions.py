import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def phi_range(phi):

    ''' 
    For a given phi, it returns phi mod 2pi such that phi lies within (-pi, pi]
    '''

    # Using boundary (-pi, pi]
    if abs(phi) == math.pi:
        return math.pi
    
    # Ensures -pi < phi < pi
    while abs(phi) > math.pi:
        if phi < -math.pi:
            phi += 2*math.pi
        if phi > math.pi:
            phi -= 2*math.pi
    return phi

def cart_to_polar(z):

    ''' 
    Converts z = (x, y, px, py) into polar coordinates
    
    Arguments: z = (x,y,px, py)
    
    Returns: z0 = (r, phi, pr, pphi)
    '''

    x, y, px, py = z
    r = math.sqrt(x**2 + y**2)
    # Safeguard to avoid division by 0
    if r < 1e-6:
        r = 1e-6
    phi = phi_range(math.atan2(y,x))
    pr = (x*px + y*py)/r
    pphi = x*py - y*px

    return np.array([r, phi, pr, pphi])

def polar_to_cart(z):

    ''' 
    Converts z = (r, phi, pr, pphi) into Cartesian coordinates
    
    Arguments: z = (r, phi, pr, pphi)
    
    Returns: z0 = (x, y, px, py)
    '''

    r, phi, pr, pphi = z

    # Safeguard to avoid 0 division
    if abs(r) < 1e-6:
        # Note: r should never be exactly 0, so np.sign() should never return 0
        r = np.sign(r) * 1e-6
    x = r*math.cos(phi)
    y = r*math.sin(phi)
    px = pr*math.cos(phi) - ((pphi * math.sin(phi)) / r)
    py = pr*math.sin(phi) + ((pphi * math.cos(phi))/r)

    return np.array([x,y, px, py])

#Hamiltonian function in polar coordinates
def H_polar(z, c = 0.0, eps = 0.0):

    '''
    The Hamiltonian function in polar coordinates with perturbation (rcos(phi))^4
    '''

    r, phi, pr, pphi = z
    return 0.5*(pr**2 + (pphi / r)**2) - 0.5*(r**2) + 0.25*(r**4) + c*pphi + eps*(r * math.cos(phi))**4

#Hamiltonian function in Cartesian coordinates
def H_cart(z, c = 0.0, eps = 0.0):

    '''
    The Hamiltonian function in polar coordinates with perturbation (rcos(phi))^4
    '''

    x, y, px, py = z
    r_squared = x**2 + y**2
    return 0.5 * (px**2 + py**2) + 0.25*r_squared**2 - 0.5*r_squared + c*(x*py - y*px) + eps*(x**4)

def pr_squared(r, phi, pphi, c = 0, eps = 0):
    '''
    This function finds the value of pr^2 on the energy level H=0
    It is used to find the initial condition for most of the integration procedures
    '''

    perturbation = eps * (r*math.cos(phi))**4
    rsquared = r**2
    rtothefour = r**4
    return -(pphi**2)/(rsquared) + rsquared - 0.5*rtothefour - 2*c*pphi - 2*perturbation

def cart_dynamics(t, z, c = 0.0, eps = 0.0):

    '''
    Hamiltonian dynamics in Cartesian coordinates
    '''

    x, y, px, py = z
    r_squared = x**2 + y**2
    sol = np.array([px - c*y, py + c*x, x*(1-r_squared) - c*py - 4*eps*(x**3), y*(1-r_squared) + c*px])
    return sol

def polar_dynamics(t, z, c = 0.0, eps = 0.0):
    
    '''
    Hamiltonian dynamics in polar coordinates
    '''

    r, phi, pr, pphi = z
    if r < 1e-6:
        r = 1e-6
        print("WARNING: r was too small, so values may not be accurate")
    sol = np.array([pr, phi_range((pphi / (r**2)) + c), ((pphi**2)/(r**3)) - (r**3) + r - (4*eps*(r**3)* (math.cos(phi)**4)), 4*eps*(r**4)*(math.cos(phi)**3)*math.sin(phi)])
    return sol

def RK45(f, t0, tf, z0, tol, max_step_user = math.inf, c = 0.0, eps = 0.0):
    
    '''
    Uses RK45 method found in SciPy module in polar coordinates

    Arguments: 
    f: function representing dynamics of system
    t0: lower bound for integration
    tf: upper bound for integration
    z0: initial condition of iteration. It determines the solution set we are looking for.
    tol: determines the relative accuracy of solution (number of correct digits)
    max_step_user: maximum step size allowed in integration
    c: magnetic field strength
    eps: perturbation strength

    Returns:
    temp.y: (x_1, ..., x_n) at each integration step
    '''

    rs, phis, prs, pphis = integrate.solve_ivp(f, [t0, tf], z0, args = (c, eps,), max_step = max_step_user, rtol = tol).y
    return rs, [phi_range(phi) for phi in phis], prs, pphis

# This is modified to the function RK45 because the second element is not restricted to (-pi, pi]
def RK45cart(f, t0, tf, z0, tol, max_step_user = math.inf, c = 0.0, eps = 0.0):
    
    '''
    Uses RK45 method found in SciPy module in Cartesian coordinates

    Arguments: 
    f: function representing dynamics of system
    t0: lower bound for integration
    tf: upper bound for integration
    z0: initial condition of iteration. It determines the solution set we are looking for.
    tol: determines the relative accuracy of solution (number of correct digits)
    max_step_user: maximum step size allowed in integration
    c: magnetic field strength
    eps: perturbation strength

    Returns:
    temp.y: (x_1, ..., x_n) at each integration step
    '''

    xs, ys, pxs, pys = integrate.solve_ivp(f, [t0, tf], z0, args = (c, eps,), max_step = max_step_user, rtol = tol).y
    return xs, ys, pxs, pys


def henon_f(t, z, c = 0.0, eps = 0.0):

    ''' 
    Modified polar dynamics to prepare for Poincare map with section phi = 0mod2pi.
    This is to be used together with the Henon algorithm.
    '''

    r, phi, pr, pphi = z
    dphidt = pphi / (r**2) + c
    # Potential error: second return: 
    return np.array([pr/dphidt, 1, (((pphi**2)/(r**3))- r**3 + r - (eps*4*(r**3)*(math.cos(phi)**4)))/dphidt, (eps*4*(r**3)*(math.cos(phi)**3)*math.sin(phi))/dphidt])

def henon_alg_polar(f, henonf, t0, tf, z0, tol, N, max_it, max_step_user = math.inf, c = 0.0, eps = 0.0):
    '''
    This algorithm uses Henon's trick to plot Poincare maps.
    It integrates f forwards, and, when the surface of section phi = 0 mod 2pi is crossed, Henon's trick is performed. 

    Arguments:
    f: the original polar dynamics of the Hamiltonian system
    henonf: the modified polar dynamics, where all functions are divided by dphi/dt
    t0: lower bound for integral
    tf: upper bound for integral
    z0: initial condition of iteration. It determines the solution set we are looking for.
    tol: determines the relative accuracy of solution (number of correct digits)
    N: determines the ideal number of points plotted
    max_step_user: maximum step size allowed in integration
    c: magnetic field strength
    eps: perturbation strength

    Returns:
    rvals: r-values to plot on Poincare map
    prvals: pr-values to plot on Poincare map
    '''

    # Initiate lists to store the points which will be plotted
    poincare_rs = []
    poincare_prs = []
    # For eps = 0, this is conserved, but not for eps >0
    # Also stored to see how well H is conserved along Poincare points
    poincare_pphis = []

    counter = 0

    while len(poincare_rs) < N and counter<max_it:
        counter +=1
        
        rs, phis, prs, pphis = RK45(f, t0, tf, z0, tol, max_step_user, c, eps)
        
        for i in range(1, len(phis)):
            # Crossing over in one direction only (derivative of phi is increasing)
            if phis[i-1]<0 and phis[i] > 0:
                rs1, _, prs1, pphis1 = RK45(henonf, phis[i], 0, np.array([rs[i], phis[i], prs[i], pphis[i]]), tol, max_step_user, c, eps)
                poincare_rs.append(rs1[-1])
                poincare_prs.append(prs1[-1])
                poincare_pphis.append(pphis1[-1])
            
            if len(poincare_rs) >= N:
                return poincare_rs, poincare_prs, poincare_pphis
        
        t0, tf = tf, 2*tf
            
    return poincare_rs, poincare_prs, poincare_pphis

def henon_f_cart(t, z, c = 0.0, eps = 0.0):

    ''' 
    Modified cartesian dynamics to prepare for Poincare map with section phi = 0mod2pi, which is equivalent to y = 0, x >= 0.
    This is to be used together with the Henon algorithm.
    '''

    x, y, px, py = z
    r_squared = x**2 + y**2
    dydt = py + c*x
    sol = np.array([(px - c*y)/dydt, 1, (x*(1-r_squared) - c*py - 4*eps*(x**3))/dydt, (y*(1-r_squared) + c*px)/dydt])
    return sol

def henon_alg_cart(f, henonf, t0, tf, z0, tol, N, max_it, max_step_user = math.inf, c = 0.0, eps = 0.0):
    '''
    This algorithm uses Henon's trick to plot Poincare maps.
    It integrates f forwards, and, when the surface of section phi = 0 mod 2pi is crossed, which is equivalent to x >= 0 and y = 0, Henon's trick is performed. 

    Arguments:
    f: the original cartesian dynamics of the Hamiltonian system
    henonf: the modified cartesian dynamics, where all functions are divided by dy/dt
    t0: lower bound for integral
    tf: upper bound for integral
    z0: initial condition of iteration. It determines the solution set we are looking for.
    tol: determines the relative accuracy of solution (number of correct digits)
    N: determines the ideal number of points plotted
    max_step_user: maximum step size allowed in integration
    c: magnetic field strength
    eps: perturbation strength

    Returns:
    rvals: r-values to plot on Poincare map
    prvals: pr-values to plot on Poincare map


    Plan of action: integrate, and stop if y changes sign. Then, before performing Henon's trick, ensure that x >= 0. 
    '''

    # Initiate lists to store the Poincare points
    poincare_xs = []
    poincare_ys = []
    poincare_pxs = []
    poincare_pys = []

    counter = 0

    while len(poincare_xs) < N and counter < max_it:
        counter +=1
        xs, ys, pxs, pys = RK45cart(f, t0, tf, z0, tol, max_step_user, c, eps)

        for i in range(1, len(ys)):
            # Ensures crossing of section phi = 0 mod 2pi with increasing phi only
            if (ys[i-1] < 0 and ys[i]>0) and (xs[i] >= 0):
                z0 = np.array([xs[i], ys[i], pxs[i], pys[i]])
                xs1, ys1, pxs1, pys1 = RK45cart(henonf, ys[i], 0, z0, tol, max_step_user, c, eps)
                poincare_xs.append(xs1[-1])
                poincare_ys.append(ys1[-1])
                poincare_pxs.append(pxs1[-1])
                poincare_pys.append(pys1[-1])
            
            # Enough points have been collected
            if len(poincare_xs) >= N:
                return poincare_xs, poincare_ys, poincare_pxs, poincare_pys
        
        # Repeats process with new time interval to continue the integration procedure
        t0, tf = tf, 2*tf
       

    return poincare_xs, poincare_ys, poincare_pxs, poincare_pys

def henon_alg_cart_bwd(f, henonf, t0, tf, z0, tol, N, max_it, max_step_user = math.inf, c = 0.0, eps = 0.0):
    '''
    This algorithm uses Henon's trick to plot Poincare maps.
    It integrates f backwards, and, when the surface of section phi = 0 mod 2pi is crossed, which is equivalent to x >= 0 and y = 0, Henon's trick is performed. 

    Arguments:
    f: the original cartesian dynamics of the Hamiltonian system
    henonf: the modified cartesian dynamics, where all functions are divided by dy/dt
    t0: lower bound for integral
    tf: upper bound for integral
    z0: initial condition of iteration. It determines the solution set we are looking for.
    tol: determines the relative accuracy of solution (number of correct digits)
    N: determines the ideal number of points plotted
    max_step_user: maximum step size allowed in integration
    c: magnetic field strength
    eps: perturbation strength

    Returns:
    rvals: r-values to plot on Poincare map
    prvals: pr-values to plot on Poincare map


    Plan of action: integrate, and stop if y changes sign. Then, before performing Henon's trick, ensure that x >= 0. 
    '''

    # Initiate lists to store the Poincare points
    poincare_xs = []
    poincare_ys = []
    poincare_pxs = []
    poincare_pys = []

    counter = 0

    while len(poincare_xs) < N and counter < max_it:
        counter +=1
        xs, ys, pxs, pys = RK45cart(f, t0, tf, z0, tol, max_step_user, c, eps)

        for i in range(1, len(ys)):
            # Ensures crossing of section phi = 0 mod 2pi with decreasing phi only, as the integration is done backwards in time
            if (ys[i-1] > 0 and ys[i]<0) and (xs[i] >= 0):
                z0 = np.array([xs[i], ys[i], pxs[i], pys[i]])
                xs1, ys1, pxs1, pys1 = RK45cart(henonf, ys[i], 0, z0, tol, max_step_user, c, eps)
                poincare_xs.append(xs1[-1])
                poincare_ys.append(ys1[-1])
                poincare_pxs.append(pxs1[-1])
                poincare_pys.append(pys1[-1])
            
            if len(poincare_xs) >= N:
                return poincare_xs, poincare_ys, poincare_pxs, poincare_pys
        
        t0, tf = tf, 2*tf
       

    return poincare_xs, poincare_ys, poincare_pxs, poincare_pys


def cart_dynamics_bwd(t, z, c = 0, eps = 0.0):
    '''
    Returns the reversed cartesian dynamics 
    '''

    return -cart_dynamics(t, z, c, eps)

def henon_f_cart_bwd(t, z, c = 0.0, eps = 0.0):
    '''
    Returns the reversed Henon dynamics in Cartesian coordinates
    '''
    return -henon_f_cart(t, z, c, eps)

def eigenvectors(c):
    '''For saddle point (x, y, px, py) = (0,0,0,0)
    As jacobian is known, this returns the normalized stable and unstable eigenvectors 
    
    Arguments: 
    c: strength of magnetic field
    
    Returns:
    vu: unstable eigenvector
    vs: stable eigenvector 
    '''

    jac = np.array([
        [0, -c, 1, 0],
        [c, 0, 0, 1], 
        [1, 0, 0, -c],
        [0, 1, c, 0]
    ])

    # Both eigenvalues and eigenvectors can be complex
    eigvals, eigvecs = np.linalg.eig(jac)

    # Takes real parts of eigenvectors
    vu = eigvecs[:, np.argmax(eigvals.real)].real
    vu /= np.linalg.norm(vu)

    vs = eigvecs[:, np.argmin(eigvals.real)].real
    vs /= np.linalg.norm(vs)

    return vu, vs

def seed_directions(c):
    ''' Since the eigenvectors are known, this returns the eigenvectors such that they lie in 
    r>0 for polar coordinates'''
    vu, vs = eigenvectors(c)
    seed_u = -vu
    seed_s = vs

    return seed_u, seed_s

def collect_per_seed(dist, seed_vec, f, hf, alg, T_range, tolerance, N_crossings, max_it, max_step, c, eps):
    '''Places one seed at a distance from the equilibrium along seed_vec, integrates and returns list of crossings'''
    z0 = dist*seed_vec
    xs, _, pxs, _ = alg(f, hf, 0, T_range, z0, tolerance, N_crossings, max_it, max_step, c, eps)

    return list(zip(xs, pxs))

def plot_arc(ax, arc_xpx, color, label=None):
    if not arc_xpx:
        return

    arc_r, arc_pr = [], []

    for x, px in arc_xpx:
        # By how the Poincare section is chosen, the conversion to polar coordinates is equal
        r, pr = x, px

        # To avoid getting too close to the equilibrium point
        if r < 1e-4:
            if arc_r and not math.isnan(arc_r[-1]):
                arc_r.append(float('nan'))
                arc_pr.append(float('nan'))
            continue

        arc_r.append(r)
        arc_pr.append(pr)

    ax.plot(arc_r, arc_pr, color=color, label=label)

def compute_and_plot(min_dist, max_dist, N_seeds, T_range, tolerance, N_crossings, max_it, max_step, N_arcs, c, eps):
    seed_u, seed_s = seed_directions(c)
    # Use geomspace to have initial conditions clustered close to equilibrium 
    distances = np.geomspace(min_dist, max_dist, N_seeds)

    # Wu crossing
    print("Starting computation for the unstable manifold...")
    per_seed_collection_u = []
    for d in distances:
        crossings = collect_per_seed(d, seed_u, cart_dynamics, henon_f_cart, henon_alg_cart, T_range, tolerance, N_crossings, max_it, max_step, c, eps)
        per_seed_collection_u.append(crossings)

    # Ws crossing
    print("Starting computation for the stable manifold...")
    per_seed_collection_s = []
    for d in distances:
        crossings = collect_per_seed(d, seed_s, cart_dynamics_bwd, henon_f_cart_bwd, henon_alg_cart_bwd, T_range, tolerance, N_crossings, max_it, max_step, c, eps)
        per_seed_collection_s.append(crossings)

    fig, ax = plt.subplots(figsize=(7, 6))

    for j in range(1, N_arcs + 1):
        # Natural seed order — seeds near saddle give crossings near origin
        arc_u = [per_seed_collection_u[i][j] for i in range(N_seeds) if len(per_seed_collection_u[i]) > j]
        arc_s = [per_seed_collection_s[i][j] for i in range(N_seeds) if len(per_seed_collection_s[i]) > j]

        plot_arc(ax, arc_u, 'r', "Unstable manifold" if j == 1 else None)
        plot_arc(ax, arc_s, 'b', "Stable manifold" if j == 1 else None)


    ax.set_xlabel("$r$")
    ax.set_ylabel("$p_r$")
    ax.legend(fontsize=12)
    ax.grid()
    plt.tight_layout()
    plt.savefig(f"Stable and unstable manifolds c{c} eps{eps} Narcs{N_arcs}.pdf", format="pdf", bbox_inches = 'tight')
    plt.show()