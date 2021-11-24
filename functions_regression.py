import numpy as np
import sympy as sym
import copy
import time as t
import itertools as it
from openpiv import validation, filters


def arg_sorted(X, reverse):
    """
    Function that returns all indices to sort a list
    """
    return [i[0] for i in sorted(enumerate(X), key=lambda x: x[1],
                                 reverse=reverse)]


def Langevin_diff_2D_sym(*args):
    """
    Function that computes Langevin diffusion terms from FP diffusion terms.
    !! This is the case when a and g are assumed symmetric !!
    """
    retour = []

    for a_KM in args:
        a_xx = a_KM[0]
        a_yy = a_KM[2]
        a_xy = a_KM[1]

        Interm = np.sqrt(a_xx*a_yy - a_xy**2)
        # Always real : Cauchy-Schwarz

        g1 = (a_xx - Interm)/np.sqrt(a_xx + a_yy - 2*Interm)
        g2 = (a_yy - Interm)/np.sqrt(a_xx + a_yy - 2*Interm)
        g3 = a_xy/np.sqrt(a_xx + a_yy - 2*Interm)

        G = np.array([g1, g3, g2])

        retour.append(np.sqrt(2)*np.real(G))

    return tuple(retour)


def FP_diff_2D(*args):
    """
    Function that computes FP diffusion terms from Langevin diffusion terms.
    !! This is the case when a and g are assumed symmetric !!
    """
    retour = []

    for g_KM in args:
        G_KM = np.array([[g_KM[0, :], g_KM[1, :]], [g_KM[1, :], g_KM[2, :]]])
        dim_len = len(G_KM.shape)

        axis = np.arange(dim_len)
        mat_perm = np.concatenate((axis[2:], axis[:2]))
        mat_perm_T = np.concatenate((axis[2:], np.flip(axis[:2])))

        G_KM_T = np.transpose(G_KM, tuple(mat_perm))
        G_KM = np.transpose(G_KM, tuple(mat_perm_T))

        a_KM = 1/2*(G_KM @ G_KM_T)

        mat_perm = np.concatenate((axis[-2:], axis[:-2]))
        a_KM = np.transpose(a_KM, tuple(mat_perm))

        a_KM = np.array([a_KM[0, 0], a_KM[1, 0], a_KM[1, 1]])

        retour.append(a_KM)

    return tuple(retour)


def FP_diff_1D(*args):
    retour = []

    for g_KM in args:
        retour.append(0.5*g_KM**2)

    return tuple(retour)


def Langevin_diff_1D(*args):
    retour = []

    for a_KM in args:
        retour.append(np.sqrt(2*a_KM))

    return tuple(retour)


def build_sympy_poly(*args, **kwargs):
    """
    Create sympy polynomes in different dimensions (max 3 : x,y,z).

    *args : list of polynomes degrees ex : [1,2] -> x,1 and y**2,y,1.

    *kwargs : allow for cross_terms to be counted. Cross terms are computed
    so that order of the cross term is max equal to the max order for single
    variables.
    -> [x,y] = [3,3] -> y*x**2 // x*y**2 // x*y
    """
    retour = []
    cross_terms = False
    N = len(args)
    if N > 3:
        print('Dimension sup to 3 not supported')
        return None

    x = sym.symbols('x')
    y = sym.symbols('y')
    z = sym.symbols('z')
    symbols = np.array((x, y, z))

    # Compute single variable terms
    for k in range(len(args)):
        for i in range(args[k] + 1):
            retour.append(symbols[k]**i)

    cross_terms = kwargs.get("cross_terms")

    if cross_terms:
        test = np.array(args)
        if np.any(test != test[0]):
            print('Not symmetric cross terms not supported')
            return None

        # Generate the cross_terms using itertools product
        products = list(it.product(range(args[0]), repeat=len(args)))
        # Supposed to be same order for every coefficient

        for coeffs in products:
            coeff_arr = np.array(coeffs)
            if len(np.nonzero(coeff_arr)[0]) > 1 and np.sum(coeff_arr) <= args[0]:
                # Numpy wise multiplication
                retour.append(np.prod(symbols[:len(args)]**coeff_arr))

    retour = np.array(retour)
    return retour


def filter_2D_signal(u, v, u_thresh, v_thresh, kernel_size=3, method='localmean', max_iter=3):
    """
    Reconstruct porious vectors in 2D space. Use 2D Median method from PIV
    post-processing to localise wrong data.
    Then uses a specific method (ex : "localmean"), to reconstruct the vector
    using the other correct one around.
    => Uses Open Piv functions

    *u,v - np.ndarray : Vector field to filter

    *u_thresh,v_thresh - float : Thresholds for the local median detection
    method

    *kernel_size - int : Size of the kernel used in the reconstruction method

    *method - str : Method to use to reconstruct the vector field.
    ex : localmean, disk, distance. See openpiv documentation for further
    details
    """

    u_proc = copy.deepcopy(u)
    v_proc = copy.deepcopy(v)
    # NECESSARY, shallow copy in openpiv methods

    # Localise purious vectors
    u_filt, v_filt, mask = validation.local_median_val(
        u_proc, v_proc, u_thresh, v_thresh)

    # Set NaN where 0 to reconstruct the whole vector field from kernel method
    filtre = u_proc == 0.
    u_filt[filtre] = np.NaN
    v_filt[filtre] = np.NaN

    # Reconstruct the whole vector field to keep continuity
    u_rep, v_rep = filters.replace_outliers(u_filt, v_filt,
                                            method=method,
                                            max_iter=max_iter,
                                            kernel_size=kernel_size)

    # Set outside data as NaN
    u_rep[filtre] = np.NaN
    v_rep[filtre] = np.NaN

    return u_rep, v_rep


###---###---###---###---###---###---###---###---###
###---###---###---###---###---###---###---###---###
###---###---###---###---###---###---###---###---###


class KM_coefficient:
    """
    Class designed to keep any KM coefficient :
    *order - gives the method to obtain the corresponding coeff
    Ex : [1,1] = <(x - <x>)(y - <y>)> in 2Dcoeff

    *fit_fun - list : function to fit KM_coeff (Symbolic - SYMPY)

    *fit_values - np.ndarray : values of the fit function applied on the
    histogram mapping

    *exp_values - np.ndarray : experimental KM coefficient values
    """

    def __init__(self, Dims_order, fun, KM_exp):

        # Order of the KM coeff [1,1] = x,y [2,0] = x^2
        self.order = Dims_order
        self.fit_fun = fun  # function to fit
        self.fit_values = np.zeros(0)  # Default value

        self.exp_values = KM_exp  # Keep exp values at the same time

    def compute_sym_values(self, Maps):
        """
        Compute values of the fit function onto the histogram mapping
        Maps = tuple of maps used for histogram
        -> Ex 2D : Maps = (X_map,Y_map)
        """
        x = sym.symbols('x')
        y = sym.symbols('y')
        z = sym.symbols('z')

        List_symb = [x, y, z]

        if type(Maps) == np.ndarray:  # 1 = array, 2 = Matrix, 3+ = Tensor
            dim_system = 1
            dim_map = len(Maps)
        else:
            dim_system = len(Maps)
            dim_map = len(Maps[0])

        Retour = np.zeros((len(self.fit_fun), dim_map**dim_system))

        for k in range(len(self.fit_fun)):
            lamb_expr = sym.lambdify(
                tuple(List_symb[:dim_system]), self.fit_fun[k])
            Retour[k, :] = np.array(lamb_expr(*Maps)).flatten()

        self.fit_values = Retour


class KM_list:
    """
    Class designed to keep all KM coefficients and do global operations.

    *powers - list : keeps a track of the powers of KM_coefficients in KM_list

    *KM_coeffs - list : list of KM_coefficient

    *type_diff - str : keeps into memory the type diffusion terms that is computed
    in KM_coefficient, i.e either "Langevin" or "Fokker-Planck".
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            self.powers = []
            self.KM_coeffs = []
            self.type_diff = "Fokker-Planck"

        if len(args) != 0:
            self.powers = args[0]
            self.KM_coeffs = args[1]
            self.type_diff = "Fokker-Planck"

        if len(kwargs) != 0:
            _KM_list = kwargs.get("KM_copy")
            self.powers = copy.deepcopy(_KM_list.powers)
            self.KM_coeffs = copy.deepcopy(_KM_list.KM_coeffs)
            self.type_diff = _KM_list.type_diff

    def add_KM_coeff(self, Dim_order, coefficient):
        """
        Method to add a KM_coefficient to KM_list.

        *Dim_order : list, the power of the coefficient

        *coefficient : KM_coefficient
        """
        self.powers.append(Dim_order)
        self.KM_coeffs.append(coefficient)

    def get_drift(self):
        """
        Method that identifies drift terms and give them to user.
        """
        retour = []
        powers_drift = []

        for i in range(len(self.powers)):
            if sum(self.powers[i]) == 1:
                retour.append(list(self.KM_coeffs[i].exp_values))
                powers_drift.append(list(self.powers[i]))

        # Sort by Python : reverse = x first then y then z
        return np.array([retour[i] for i in arg_sorted(powers_drift, reverse=True)])

    def get_diff(self):
        """
        Method that identifies diffusion terms and give them to user.
        """
        retour = []
        powers_diff = []

        for i in range(len(self.powers)):
            if sum(self.powers[i]) == 2:
                retour.append(list(self.KM_coeffs[i].exp_values))
                powers_diff.append(list(self.powers[i]))

        # Sort by Python : reverse = x first then y then z
        return np.array([retour[i] for i in arg_sorted(powers_diff, reverse=True)])

    def compute_sym_values(self, Maps):
        # Method to compute symbolic values in every KM_list element
        for KM_coeff in self.KM_coeffs:
            KM_coeff.compute_sym_values(Maps)

    def convert_to_langevin_1D(self):
        """
        Convert exp diffusion terms in KM_list from Fokker-Planck to Langevin.
        This function is only applied on the experimental data.
        -> 1D case using Langevin_diff_1D
        """
        if len(self.powers[0]) != 1:
            print("KM coefficients not in 1 dimension")
            return None

        values_exp = self.KM_coeffs[1].exp_values

        ##Convert
        G_exp = np.squeeze(np.array(Langevin_diff_1D(values_exp)))

        ##Change inside self
        self.KM_coeffs[1].exp_values = G_exp

    def convert_to_langevin_2D_sym(self):
        """
        Convert exp diffusion terms in KM_list from Fokker-Planck to Langevin.
        This function is only applied on the experimental data.
        -> 2D case using Langevin_diff_2D_sym
        """
        if len(self.powers[0]) != 2:
            print("KM coefficients not in 2 dimensions")
            return None

        ##First : identify diffusion terms in KM_list and order them
        powers_diff = []
        pwr_indices = []

        N = len(self.KM_coeffs[0].exp_values)

        for i in range(len(self.powers)):
            if sum(self.powers[i]) == 2:
                powers_diff.append(list(self.powers[i]))
                pwr_indices.append(i)

        # Sort
        pwr_order = np.array(arg_sorted(powers_diff, reverse=True))
        pwr_indices = np.array(pwr_indices)[pwr_order]

        ##Get_values
        values_exp = np.zeros((len(pwr_indices), N))

        for k in range(len(pwr_indices)):
            values_exp[k, :] = self.KM_coeffs[pwr_indices[k]].exp_values

        ##Compute Langevin sym
        G_exp = np.squeeze(np.array(Langevin_diff_2D_sym(values_exp)))

        ##Change inside self
        for i in range(len(pwr_indices)):
            index = pwr_indices[i]
            self.KM_coeffs[index].exp_values = G_exp[i, :]

        self.type_diff = "Langevin"

    def convert_to_FP_1D(self):
        """
        Convert exp diffusion terms in KM_list from Langevin to Fokker-Planck.
        This function is only applied on the experimental data.
        -> 1D case using FP_diff_1D
        """
        if len(self.powers[0]) != 1:
            print("KM coefficients not in 1 dimension")
            return None

        ##Convert
        values_exp = self.KM_coeffs[1].exp_values
        a_exp = np.squeeze(np.array(FP_diff_1D(values_exp)))

        ##Change inside self
        self.KM_coeffs[1].exp_values = a_exp

    def convert_to_FP_2D(self):
        """
        Convert exp diffusion terms in KM_list from Langevin to Fokker-Planck.
        This function is only applied on the experimental data.
        -> 2D case using FP_diff_2D
        """
        if len(self.powers[0]) != 2:
            print("KM coefficients not in 2 dimensions")
            return None

        ##First : identify diffusion terms in KM_list and order them
        powers_diff = []
        pwr_indices = []

        N = len(self.KM_coeffs[0].exp_values)

        for i in range(len(self.powers)):
            if sum(self.powers[i]) == 2:
                powers_diff.append(list(self.powers[i]))
                pwr_indices.append(i)

        # Sort
        pwr_order = np.array(arg_sorted(powers_diff, reverse=True))
        pwr_indices = np.array(pwr_indices)[pwr_order]

        ##Get_values
        values_exp = np.zeros((len(pwr_indices), N))

        for k in range(len(pwr_indices)):
            values_exp[k, :] = self.KM_coeffs[pwr_indices[k]].exp_values

        ##Convert
        a_exp = np.squeeze(np.array(FP_diff_2D(values_exp)))

        ##Change inside self
        for i in range(len(pwr_indices)):
            index = pwr_indices[i]
            self.KM_coeffs[index].exp_values = a_exp[i, :]

        self.type_diff = "Fokker-Planck"

    def get_fun_value(self, i):
        return self.KM_coeffs[i].fit_values

    def get_exp_value(self, i):
        return self.KM_coeffs[i].exp_values

    def filter_KM(self, in_array):
        sizes = self.get_sizes(cumul=True)
        sizes = np.concatenate((np.zeros((1,), dtype=int), sizes), axis=0)

        for i in range(len(self.powers)):
            filter = (in_array >= sizes[i])*(in_array < sizes[i + 1])
            mask = in_array[filter] - sizes[i]
            self.KM_coeffs[i].fit_values = self.KM_coeffs[i].fit_values[mask]
            self.KM_coeffs[i].fit_fun = self.KM_coeffs[i].fit_fun[mask]

    def print_filter_KM(self, in_array, **kwargs):
        sizes = self.get_sizes(cumul=True)
        sizes = np.concatenate((np.zeros((1,), dtype=int), sizes), axis=0)

        if len(kwargs) != 0:
            file = kwargs.get("file")

        for i in range(len(self.powers)):
            filter = (in_array >= sizes[i])*(in_array < sizes[i + 1])
            mask = in_array[filter] - sizes[i]
            print(str(self.powers[i]) + ' : '
                  + str(self.KM_coeffs[i].fit_fun[mask]), end=' // ')

            if len(kwargs) != 0:
                file.write('{powers} : {func} // '.format(powers=self.powers[i],
                                                          func=self.KM_coeffs[i].fit_fun[mask]))

        print('')
        if len(kwargs) != 0:
            file.write('\n')

    def get_sizes(self, **kwargs):
        """
        Method of KM_list used to get function sizes (number of polynomial terms)
        in KM_list.

        **Kwargs : if cumul = True, give a cumulated list of sizes
        """
        sizes = []
        cumul = False
        if len(kwargs) != 0:
            cumul = kwargs.get("cumul")

        for i in range(len(self.powers)):
            if len(sizes) == 0 or not cumul:
                sizes.append(len(self.KM_coeffs[i].fit_fun))
            else:
                sizes.append(len(self.KM_coeffs[i].fit_fun) + sizes[-1])

        return np.array(sizes)

    def get_diff_fun(self, **kwargs):
        """
        Method of KM_list used to get all model diffusion terms.
        If kwargs = None, then only gives all rough values (mostly useless)
        If kwargs = Xi, computes model values by doing fun_values * coefficients
        """
        retour = []
        powers_diff = []
        if len(kwargs) != 0:
            Xi = kwargs['Xi']
            sizes = self.get_sizes()
            size_tot = 0

        for i in range(len(self.powers)):
            if len(kwargs) != 0:
                length = sizes[i]

                if sum(self.powers[i]) == 2:
                    retour.append(
                        self.KM_coeffs[i].fit_values.T @ Xi[size_tot:size_tot + length])
                    powers_diff.append(list(self.powers[i]))

                size_tot += length
            else:
                retour.append(self.KM_coeffs[i].fit_values)
                powers_diff.append(list(self.powers[i]))

        # Sort by Python : reverse = x first then y then z
        return [retour[i] for i in arg_sorted(powers_diff, reverse=True)]

    def get_drift_fun(self, **kwargs):
        """
        Method of KM_list used to get all model drift terms.
        If kwargs = None, then only gives all rough values (mostly useless)
        If kwargs = Xi, computes model values by doing fun_values * coefficients
        """
        retour = []
        powers_drift = []
        if len(kwargs) != 0:
            Xi = kwargs['Xi']
            sizes = self.get_sizes()
            size_tot = 0

        for i in range(len(self.powers)):
            if len(kwargs) != 0:
                length = sizes[i]

                if sum(self.powers[i]) == 1:
                    retour.append(
                        self.KM_coeffs[i].fit_values.T @ Xi[size_tot:size_tot + length])
                    powers_drift.append(list(self.powers[i]))

                size_tot += length
            else:
                retour.append(self.KM_coeffs[i].fit_values)
                powers_drift.append(list(self.powers[i]))

        # Sort by Python : reverse = x first then y then z
        return [retour[i] for i in arg_sorted(powers_drift, reverse=True)]
