"""
Utility functions for Langevin regression

Jared Callaham (2020)
"""

import numpy as np
from time import time
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from scipy.optimize import shgo
import functions_regression as fr
import copy
import multiprocessing as mp
import time as t
import netCDF4 as nc

# Return a single expression from a list of expressions and coefficients
#   Note this will give a SymPy expression and not a function


def sindy_model(Xi, expr_list):
    return sum([Xi[i]*expr_list[i] for i in range(len(expr_list))])


def ntrapz(I, dx):
    if isinstance(dx, int) or isinstance(dx, float) or len(dx) == 1:
        return np.trapz(I, dx=dx, axis=0)
    else:
        return np.trapz(ntrapz(I, dx[1:]), dx=dx[0])


def kl_divergence(p_in, q_in, dx=1, tol=None):
    """
    Approximate Kullback-Leibler divergence for arbitrary dimensionality
    """
    if tol == None:
        tol = max(min(p_in.flatten()), min(q_in.flatten()))
    q = q_in.copy()
    p = p_in.copy()
    q[q < tol] = tol
    p[p < tol] = tol
    return ntrapz(p*np.log(p/q), dx)


def KM_avg(X, bins, stride, dt):

    Y = X[::stride]
    tau = stride*dt
    # Step (like a finite-difference derivative estimate)
    dY = (Y[1:] - Y[:-1])/tau
    dY2 = (Y[1:] - Y[:-1])**2/tau  # Conditional variance

    f_KM = np.zeros(len(bins)-1)
    a_KM = np.zeros(f_KM.shape)
    f_err = np.zeros(f_KM.shape)
    a_err = np.zeros(f_KM.shape)

    # At each histogram bin, find time series points where the state falls into this bin
    for i in range(len(bins)-1):
        mask = np.nonzero((Y[:-1] > bins[i]) * (Y[:-1] < bins[i+1]))[0]

        if len(mask) > 0:
            f_KM[i] = np.mean(dY[mask])  # Conditional average  ~ drift
            # Conditional variance  ~ diffusion
            a_KM[i] = 0.5*np.mean(dY2[mask])

            # Estimate error by variance of samples in the bin
            a_KM[i] = 0.5*np.mean(dY2[mask])  # Conditional average
            a_err[i] = np.std(dY2[mask])/np.sqrt(len(mask))

        else:
            f_KM[i] = np.nan
            f_err[i] = np.nan
            a_KM[i] = np.nan
            a_err[i] = np.nan

    return f_KM, a_KM, f_err, a_err


def is_all_ended(Nbr_proc, queue, status_proc, status_algo, solutions):
    """
    Condition function for SSR_loop_multi. The function waits for some results
    from the processes which are kept and printed.
    If all processes finished their work, the function gives True.
    """
    # Get information from on of the processes. Program blocked until results shared
    # proc_nbr : # of process
    test, proc_nbr = queue.get(block=True)

    if type(test) == str:  # == Take processes progress
        status_algo[proc_nbr] = test
    elif type(test) == tuple:  # == One of the processes finished
        prev_len = len(status_algo[proc_nbr])
        status_algo[proc_nbr] = 'Done !' + ' '*(prev_len - 6)
        solutions[proc_nbr] = test
        status_proc[proc_nbr] = True

    print_status(status_algo)

    if np.all(status_proc) == True:  # If all processes stopped => True
        return True

    return False


def print_status(status):
    for i in range(len(status) - 1):
        print(status[i], end=' | ')

    print(status[-1], end='\r')


def compare_solutions(solutions, initial_list):
    """
    Function designed to compare candidate solutions in SSR_loop_multi.
    """
    V_min = solutions[0][0]
    index_min = 0

    for i in range(1, len(solutions)):
        if solutions[i][0] < V_min:
            V_min = solutions[i][0]
            index_min = i

    return solutions[index_min]


def print_results(V, Xi, KMc, pos, in_array, **kwargs):
    print('')
    print("### RESULTS ###")
    print("Cost: {0}".format(V[pos]))

    print(Xi[:, pos])

    if len(kwargs) != 0:
        file = kwargs.get("file")

        file.write('\n')
        file.write("### RESULTS ### \n")
        file.write("Cost: {0} \n".format(V[pos]))

        file.write(str(Xi[:, pos]) + "\n")

        KMc.print_filter_KM(in_array, file=file)

        file.write('### END ### \n')
    else:
        KMc.print_filter_KM(in_array)
        print('### END ###')
        print('')


def write_checkpoint(file_save, active, Xi0, V, Xi, k0):
    netcdf_file = nc.Dataset(file_save, "w", format="NETCDF4")

    netcdf_file.createDimension("poly_size", len(Xi))
    netcdf_file.createDimension("step_size", len(V))
    netcdf_file.createDimension("actual_size", len(active))

    _active = netcdf_file.createVariable("active", "i4", ("actual_size",))
    _Xi0 = netcdf_file.createVariable("Xi0", "f8", ("poly_size",))
    _k0 = netcdf_file.createVariable("k0", "i4")
    _V = netcdf_file.createVariable("V", "f8", ("step_size",))
    _Xi = netcdf_file.createVariable("Xi", "f8", ("poly_size", "step_size"))

    _active[:] = active
    _Xi0[:] = Xi0
    _k0[:] = k0
    _V[:] = V
    _Xi[:] = Xi

    netcdf_file.close()


def read_checkpoint(file_save):
    """
    Function to read nc file in which are saved all status reached by previous
    use of the program
    """
    data = nc.Dataset(file_save, "r")

    active = data["active"][:]
    Xi0 = data["Xi0"][:]
    V = data["V"][:]
    Xi = data["Xi"][:]
    k0 = data["k0"][:]

    return active, Xi0, V, Xi, k0

    # Return optimal coefficients for finite-time correction


def AFP_opt(cost, params):
    """
    Optimisation function. Keep a track of the whole optimisation problem.
    Called in main program. Other optimisation algorithms can be chosen from
    Scipy library : Powell, Nelder-mead, differential-evolution...
    """
    start_time = time()
    Xi0 = params["Xi0"]
    Nbr_iter_max = params.get("Nbr_iter_max")

    is_complex = np.iscomplex(Xi0)  # Dunno why... -Antoine

    params['track'] = 0  # Keep track of iteration #
    if np.any(is_complex):
        # Split vector in two for complex
        Xi0 = np.concatenate((np.real(Xi0), np.imag(Xi0)))
        def opt_fun(Xi): return cost(
            Xi[:len(Xi)//2] + 1j*Xi[len(Xi)//2:], params)

    else:
        def opt_fun(Xi): return cost(Xi, params)

    res = minimize(opt_fun, Xi0, method='Nelder-Mead',
                   options={'disp': False, 'maxfev': int(Nbr_iter_max)})

    if params["print_cost"]:
        print('%%%% Optimization time: {0} seconds,   Cost: {1} %%%%'.format(
            time() - start_time, res.fun))
        print(res.message)
        print('Number of iterations : '
              + str(res.nfev) + '/' + str(int(Nbr_iter_max)))

    # Return coefficients and cost function
    if np.any(is_complex):
        # Return to complex number
        return res.x[:len(res.x)//2] + 1j*res.x[len(res.x)//2:], res.fun
    else:
        if res.nfev >= Nbr_iter_max:  # Handle the case where does not converge
            return res.x, res.fun, 'Max_reached'
        return res.x, res.fun


def SSR_loop(opt_fun, params):
    """
    Stepwise sparse regression: general function for a given optimization problem
       opt_fun should take the parameters and return coefficients and cost

    Requires a list of drift and diffusion expressions,
        (although these are just passed to the opt_fun)

    Same regression type for every KM coeff - One CPU kernel
    """

    KMc = fr.KM_list(KM_copy=params['KMc'])
    Xi0 = params['Xi0'].copy()
    Max_it = len(KMc.powers)  # Max of remaining terms in polynomes at the end

    m = len(Xi0)  # Number of fit coefficients

    Xi = np.zeros((m, m - Max_it+1))  # Output results
    V = np.zeros((m - Max_it+1))      # Cost at each step

    # Full regression problem as baseline
    # Not done if loading from checkpoint
    if not params["checkpoint_load"]:
        Xi[:, 0], V[0] = opt_fun(params)[0:2]
        print(Xi[:, 0])

    # Start with all candidates
    active = np.arange(m)

    k0 = 1  # Load data from checkpoint if required
    if params["checkpoint_load"]:
        active, Xi0, V, Xi, k0 = read_checkpoint(params["checkpoint_file"])

    # Iterate and threshold
    for k in range(k0, m - Max_it+1):
        # Loop through remaining terms and find the one that increases the cost
        # function the least
        min_idx = -1
        V[k] = np.inf

        for j in range(len(active)):
            tmp_active = active.copy()
            tmp_active = np.delete(tmp_active, j)  # Try deleting this term

            KMc_j = fr.KM_list(KM_copy=KMc)

            KMc_j.filter_KM(tmp_active)  # Filter selected data
            sizes = KMc_j.get_sizes()

            params['KMc'] = KMc_j
            params['Xi0'] = Xi0[tmp_active]

            # Ensure that there is at least one drift and diffusion term left
            if not (np.any(sizes == 0)):
                tmp_Xi, tmp_V = opt_fun(params)[0:2]  # Optimise !

                # Keep minimum cost
                if tmp_V < V[k]:
                    min_idx = j
                    V[k] = tmp_V
                    min_Xi = tmp_Xi

        active = np.delete(active, min_idx)  # Remove inactive index
        Xi0[active] = min_Xi  # Re-initialize with best results from previous
        Xi[active, k] = min_Xi

        print_results(V, Xi, KMc, k, active)

    return Xi, V


def SSR_loop_multi(opt_fun, params):
    """
    Stepwise sparse regression: general function for a given optimization problem
       opt_fun should take the parameters and return coefficients and cost

    Requires a list of drift and diffusion expressions,
        (although these are just passed to the opt_fun)

    multi : Using multiprocessing module
    Must integrate Nbr_proc : Number of processus
    """

    KMc = fr.KM_list(KM_copy=params['KMc'])
    Xi0 = params['Xi0'].copy()
    Nbr_proc = params['Nbr_proc']  # Number of asked processes
    Max_it = len(KMc.powers)  # Max of remaining terms in polynomes at the end

    file = None  # If file writing
    file = params.get("file_results")
    del params["file_results"]  # !! Error with deepcopy !!

    m = len(Xi0)  # Number of fit coefficients

    Xi = np.zeros((m, m - Max_it+1))  # Output results
    V = np.zeros((m - Max_it+1))      # Cost at each step

    # Full regression problem as baseline
    # Xi[:, 0], V[0] = opt_fun(params)
    # No need if loading from checkpoint
    if not params["checkpoint_load"]:
        opt_val = opt_fun(params)
        Xi[:, 0], V[0] = opt_val[0:2]

    # Disable cost print (will not work with processes)
    params['print_cost'] = False

    # Start with all candidates
    active = np.arange(m)
    # Keep in mind active to give to processes
    params.update({"active": active})

    # All works are transmitted to processes via dedicated queues (in).
    Queues_in = [mp.Queue() for i in range(Nbr_proc)]
    # All work done or progress are all loaded in a single queue (out). This is
    # then analysed in the main process.
    q_out = mp.Queue()
    # Initialise processes
    Processes = [mp.Process(target=SSR_process, args=(
        Queues_in[i], q_out, opt_fun, params)) for i in range(Nbr_proc)]

    # Start processes without any work to do
    for p in Processes:
        p.start()

    k0 = 1  # Load data from checkpoint if required
    if params["checkpoint_load"]:
        active, Xi0, V, Xi, k0 = read_checkpoint(params["checkpoint_file"])

    for k in range(k0, m - Max_it+1):
        t0 = t.time()
        active_cut = np.array_split(active, Nbr_proc)  # Split active
        condition_out = False  # Condition to finish the program
        # variables to keep in mind
        status_proc = np.zeros((Nbr_proc,), dtype=bool)
        # If current work is done in the different processes
        # Solutions obtained in processes (to be compared)
        solutions = [None]*Nbr_proc
        status_algo = [None]*Nbr_proc  # Print processes status

        # Insert new work to do
        for i in range(Nbr_proc):
            Queues_in[i].put(
                (active_cut[i], copy.deepcopy(KMc), active, Xi0, i))

        while not condition_out:
            condition_out = is_all_ended(
                Nbr_proc, q_out, status_proc, status_algo, solutions)

        V[k], min_idx, min_Xi = compare_solutions(solutions, active_cut)
        active = np.delete(active, min_idx)  # Remove inactive index
        Xi0[active] = min_Xi  # Re-initialize with best results from previous
        Xi[active, k] = min_Xi

        # Save data to create checkpoints
        write_checkpoint(params["checkpoint_file"], active, Xi0, V, Xi, k+1)

        if file is not None:
            # Write into a file
            print_results(V, Xi, KMc, k, active, file=file)
        else:
            print_results(V, Xi, KMc, k, active)

        print("Duration :" + str(t.time() - t0))

    # End all processes
    for i in range(Nbr_proc):
        Queues_in[i].put('END')

    # Join all processes
    for p in Processes:
        p.join()

    return Xi, V


def SSR_process(q_in, q_out, opt_fun, params):
    """
    Function for SSR_loop_multi, used for multi-processing.
    *q_in : Queue only for putting work to do (not shared with other processes)
    *q_out : Queue only for putting results or progress (shared with other processes)
    """
    for active_proc, KMc, active, Xi0, proc_nbr in iter(q_in.get, 'END'):
        V = np.inf
        min_idx = -1
        params_proc = copy.deepcopy(params)
        max_handle = ''  # Handle when optimisation does not converge

        for j in range(len(active_proc)):
            tmp_active = active.copy()
            # As work are splitted, need to find which part corresponds to the
            # original array
            idx = np.argwhere(tmp_active == active_proc[j])
            tmp_active = np.delete(tmp_active, idx)  # Try deleting this term

            # Give information of progress made
            q_out.put(['Step {it}/{max}'.format(it=j+1,
                      max=len(active_proc)) + max_handle, proc_nbr])

            KMc_j = fr.KM_list(KM_copy=KMc)

            KMc_j.filter_KM(tmp_active)  # Filter selected data
            sizes = KMc_j.get_sizes()

            params_proc['KMc'] = KMc_j  # Update params for optimisation
            params_proc['Xi0'] = Xi0[tmp_active]

            # Ensure that there is at least one drift and diffusion term left
            if not (np.any(sizes == 0)):
                opt_val = opt_fun(params_proc)  # Optimise !

                if len(opt_val) == 3:  # Print a star when there is no convergence
                    tmp_Xi, tmp_V = opt_val[0:2]
                    max_handle += '*'
                else:
                    tmp_Xi, tmp_V = opt_val

                # Keep minimum cost
                if tmp_V < V:
                    min_idx = idx
                    V = tmp_V
                    min_Xi = tmp_Xi

        q_out.put([(V, min_idx, min_Xi), proc_nbr])


def cost_reg(Xi, params):
    """
    Least-squares cost function for optimization
    Xi - current coefficient estimates
    param - inputs to optimization problem: grid points, list of candidate expressions, regularizations
        W, KMc, x_pts, y_pts, x_msh, y_msh, f_expr, a_expr, l1_reg, l2_reg, kl_reg, p_hist, etc
    """
    ### Unpack parameters ###
    W = params['W']  # Optimization weights
    track = params['track']  # Track number of optimisation iterations
    Nbr_iter_max = params['Nbr_iter_max']
    track += 1

    if params['print_cost']:
        print('{0}/{1}'.format(track, int(Nbr_iter_max)), flush=True, end='\r',)
    params['track'] = track
    # Kramers-Moyal coefficients
    KMc = fr.KM_list(KM_copy=params['KMc'])

    powers = KMc.powers
    ndims = len(powers[0])

    fp, afp = params['fp'], params['afp']  # Fokker-Planck solvers
    N = params['N']

    ### Get all KM coefficients fit values ###
    f_vals = KMc.get_drift_fun(Xi=Xi)
    g_vals = KMc.get_diff_fun(Xi=Xi)

    ### Convert to FP ###
    if ndims == 1:
        a_vals = fr.FP_diff_1D(np.array(g_vals))[0]
    if ndims == 2:
        a_vals = fr.FP_diff_2D(np.array(g_vals))[0]

    ### FP regression ###
    # Solve AFP equation to find finite-time corrected drift/diffusion
    #    corresponding to the current parameters Xi
    afp.precompute_operator(np.squeeze(f_vals), np.squeeze(a_vals))

    if params.get("mod_multi") == "diff":
        f_tau, a_tau = afp.solve(params['tau'], Nbr_proc=params["Nbr_proc"])
    else:
        f_tau, a_tau = afp.solve(params['tau'])

    KM_tau = np.concatenate((f_tau, a_tau))  # Concatenate results

    # Histogram points without data have NaN values in K-M average - ignore
    # these in the average
    mask = np.nonzero(np.isfinite(KMc.get_exp_value(0)))[0]
    V = 0

    KM_exp = np.concatenate((KMc.get_drift(), KMc.get_diff()))

    for k in range(len(powers)):  # Compute cost
        V += np.sum(W[k, mask]*(KM_tau[k][mask]
                    - KM_exp[k, mask])**2)

    # Include PDF constraint via Kullbeck-Leibler divergence regularization
    if params['kl_reg'] > 0:
        p_hist = params['p_hist']  # Empirical PDF
        dims_f = [ndims] + [N]*ndims  # Reconstruct dims
        dims_a = [ndims*(ndims+1)//2] + [N]*ndims
        f_vals = np.reshape(f_vals, dims_f)
        a_vals = np.reshape(a_vals, dims_a)

        # Solve Fokker-Planck equation for steady-state PDF
        p_est = fp.solve(np.squeeze(f_vals), np.squeeze(a_vals))

        kl = kl_divergence(p_hist, p_est, dx=fp.dx, tol=1e-6)
        # Numerical integration can occasionally produce small negative values
        kl = max(0, kl)
        V += params['kl_reg']*kl

    return V


# 1D Markov test
def markov_test(X, lag, N=32, L=2):
    # Lagged time series
    X1 = X[:-2*lag:lag]
    X2 = X[lag:-lag:lag]
    X3 = X[2*lag::lag]

    # Two-time joint pdfs
    bins = np.linspace(-L, L, N+1)
    dx = bins[1]-bins[0]
    p12, _, _ = np.histogram2d(X1, X2, bins=[bins, bins], density=True)
    p23, _, _ = np.histogram2d(X2, X3, bins=[bins, bins], density=True)
    p2, _ = np.histogram(X2, bins=bins, density=True)
    p2[p2 < 1e-4] = 1e-4

    # Conditional PDF (Markov assumption)
    pcond_23 = p23.copy()
    for j in range(pcond_23.shape[1]):
        pcond_23[:, j] = pcond_23[:, j]/p2

    # Three-time PDFs
    p123, _ = np.histogramdd(np.array([X1, X2, X3]).T, bins=np.array(
        [bins, bins, bins]), density=True)
    p123_markov = np.einsum('ij,jk->ijk', p12, pcond_23)

    # Chi^2 value
    #return utils.ntrapz( (p123 - p123_markov)**2, [dx, dx, dx] )/(np.var(p123.flatten()) + np.var(p123_markov.flatten()))
    return kl_divergence(p123, p123_markov, dx=[dx, dx, dx], tol=1e-6)


### FAST AUTOCORRELATION FUNCTION
# From https://dfm.io/posts/autocorr/

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

    # Chi^2 value
    #return utils.ntrapz( (p123 - p123_markov)**2, [dx, dx, dx] )/(np.var(p123.flatten()) + np.var(p123_markov.flatten()))
    return kl_divergence(p123, p123_markov, dx=[dx, dx, dx], tol=1e-6)


### FAST AUTOCORRELATION FUNCTION
# From https://dfm.io/posts/autocorr/

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf
    return acf
