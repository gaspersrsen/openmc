from collections import defaultdict
from collections.abc import Callable, Iterable
from numbers import Real, Integral

import scipy.optimize as sopt
import numpy as np
import copy

import openmc
import openmc.model
from openmc import Material, Tally, MaterialFilter
import openmc.checkvalue as cv
import openmc.lib
from openmc.mpi import comm

import warnings


_SCALAR_BRACKETED_METHODS = {'brentq', 'brenth', 'ridder', 'bisect'}


def _search_keff(guess, target, model_builder, model_args, print_iterations,
                 run_args, guesses, results):
    """Function which will actually create our model, run the calculation, and
    obtain the result. This function will be passed to the root finding
    algorithm

    Parameters
    ----------
    guess : Real
        Current guess for the parameter to be searched in `model_builder`.
    target_keff : Real
        Value to search for
    model_builder : collections.Callable
        Callable function which builds a model according to a passed
        parameter. This function must return an openmc.model.Model object.
    model_args : dict
        Keyword-based arguments to pass to the `model_builder` method.
    print_iterations : bool
        Whether or not to print the guess and the resultant keff during the
        iteration process.
    run_args : dict
        Keyword arguments to pass to :meth:`openmc.Model.run`.
    guesses : Iterable of Real
        Running list of guesses thus far, to be updated during the execution of
        this function.
    results : Iterable of Real
        Running list of results thus far, to be updated during the execution of
        this function.

    Returns
    -------
    float
        Value of the model for the current guess compared to the target value.

    """

    # Build the model
    model = model_builder(guess, **model_args)

    # Run the model and obtain keff
    sp_filepath = model.run(**run_args)
    with openmc.StatePoint(sp_filepath) as sp:
        keff = sp.keff

    # Record the history
    guesses.append(guess)
    results.append(keff)

    if print_iterations:
        text = 'Iteration: {}; Guess of {:.2e} produced a keff of ' + \
            '{:1.5f} +/- {:1.5f}'
        print(text.format(len(guesses), guess, keff.n, keff.s))

    return keff.n - target


def search_for_keff(model_builder, initial_guess=None, target=1.0,
                    bracket=None, model_args=None, tol=None,
                    bracketed_method='bisect', print_iterations=False,
                    run_args=None, **kwargs):
    """Function to perform a keff search by modifying a model parametrized by a
    single independent variable.

    Parameters
    ----------
    model_builder : collections.Callable
        Callable function which builds a model according to a passed
        parameter. This function must return an openmc.model.Model object.
    initial_guess : Real, optional
        Initial guess for the parameter to be searched in
        `model_builder`. One of `guess` or `bracket` must be provided.
    target : Real, optional
        keff value to search for, defaults to 1.0.
    bracket : None or Iterable of Real, optional
        Bracketing interval to search for the solution; if not provided,
        a generic non-bracketing method is used. If provided, the brackets
        are used. Defaults to no brackets provided. One of `guess` or `bracket`
        must be provided. If both are provided, the bracket will be
        preferentially used.
    model_args : dict, optional
        Keyword-based arguments to pass to the `model_builder` method. Defaults
        to no arguments.
    tol : float
        Tolerance to pass to the search method
    bracketed_method : {'brentq', 'brenth', 'ridder', 'bisect'}, optional
        Solution method to use; only applies if
        `bracket` is set, otherwise the Secant method is used.
        Defaults to 'bisect'.
    print_iterations : bool
        Whether or not to print the guess and the result during the iteration
        process. Defaults to False.
    run_args : dict, optional
        Keyword arguments to pass to :meth:`openmc.Model.run`. Defaults to no
        arguments.

        .. versionadded:: 0.13.1
    **kwargs
        All remaining keyword arguments are passed to the root-finding
        method.

    Returns
    -------
    zero_value : float
        Estimated value of the variable parameter where keff is the
        targeted value
    guesses : List of Real
        List of guesses attempted by the search
    results : List of 2-tuple of Real
        List of keffs and uncertainties corresponding to the guess attempted by
        the search

    """

    if initial_guess is not None:
        cv.check_type('initial_guess', initial_guess, Real)
    if bracket is not None:
        cv.check_iterable_type('bracket', bracket, Real)
        cv.check_length('bracket', bracket, 2)
        cv.check_less_than('bracket values', bracket[0], bracket[1])
    if model_args is None:
        model_args = {}
    else:
        cv.check_type('model_args', model_args, dict)
    cv.check_type('target', target, Real)
    cv.check_type('tol', tol, Real)
    cv.check_value('bracketed_method', bracketed_method,
                   _SCALAR_BRACKETED_METHODS)
    cv.check_type('print_iterations', print_iterations, bool)
    if run_args is None:
        run_args = {}
    else:
        cv.check_type('run_args', run_args, dict)
    cv.check_type('model_builder', model_builder, Callable)

    # Run the model builder function once to make sure it provides the correct
    # output type
    if bracket is not None:
        model = model_builder(bracket[0], **model_args)
    elif initial_guess is not None:
        model = model_builder(initial_guess, **model_args)
    cv.check_type('model_builder return', model, openmc.model.Model)

    # Set the iteration data storage variables
    guesses = []
    results = []

    # Set the searching function (for easy replacement should a later
    # generic function be added.
    search_function = _search_keff

    if bracket is not None:
        # Generate our arguments
        args = {'f': search_function, 'a': bracket[0], 'b': bracket[1]}
        if tol is not None:
            args['rtol'] = tol

        # Set the root finding method
        if bracketed_method == 'brentq':
            root_finder = sopt.brentq
        elif bracketed_method == 'brenth':
            root_finder = sopt.brenth
        elif bracketed_method == 'ridder':
            root_finder = sopt.ridder
        elif bracketed_method == 'bisect':
            root_finder = sopt.bisect

    elif initial_guess is not None:

        # Generate our arguments
        args = {'func': search_function, 'x0': initial_guess}
        if tol is not None:
            args['tol'] = tol

        # Set the root finding method
        root_finder = sopt.newton

    else:
        raise ValueError("Either the 'bracket' or 'initial_guess' parameters "
                         "must be set")

    # Add information to be passed to the searching function
    args['args'] = (target, model_builder, model_args, print_iterations,
                    run_args, guesses, results)

    # Create a new dictionary with the arguments from args and kwargs
    args.update(kwargs)

    # Perform the search
    zero_value = root_finder(**args)

    return zero_value, guesses, results


def get_ao_mix_materials(materials, fracs, fracs_target=None, percent_type='ao', return_wgts=False):
    """
    Mix materials together based on atom, weight, or volume fractions. Any fractions can be applied on a subset of nuclides for each material, by setting the 'fracs_target' argument to a list of nuclides or elements for each material. If 'fracs_target' is not set, the fractions are applied to the whole material. The 'percent_type' argument specifies whether the provided fractions are atom percent (molar percent), weight percent, or volume percent. One of the fractions can be set to None, and it will be automatically calculated to ensure the fractions sum to 1.

    .. versionadded:: 0.15.4

    Parameters
    ----------
    materials : Iterable of openmc.Material
        Materials to combine
    fracs : Iterable of float
        Fractions of each material to be combined
    fracs_target : Iterable of str, optional
        Fraction target of each material to be combined, can be nuclide (i.e. "B10"), or element (i.e. "B")
        or element (ex. "B")
    percent_type : {'ao', 'wo', 'vo'}
        Type of percentage, must be one of 'ao', 'wo', or 'vo', to signify atom
        percent (molar percent), weight percent, or volume percent,
        optional. Defaults to 'ao'
    return_wgts : bool, optional
        Whether to return the calculated weights of each material in the mixture. Defaults to False.

    Returns
    -------
    dict
        A dictionary of the nuclides in the mixture and their corresponding atom densities in atoms/b-cm
    np.ndarray, optional
        The calculated weights of each material in the mixture, if `return_wgts` is True

    """

    cv.check_type('materials', materials, Iterable, Material)
    # cv.check_type('fracs', fracs, Iterable, Real)
    cv.check_value('percent type', percent_type, {'ao', 'wo', 'vo'})

    fracs = np.array(fracs)
    
    if len(materials) != len(fracs):
        raise ValueError(f"Number of provided materials: {len(materials)}; does not match the number of provided material fractions: {len(fracs)}")
    if fracs_target is None:
        fracs_target = [None] * len(fracs)

    # Calculate appropriate weights which are what fraction of a cc of each
    # material are found in 1cc of the composite material
    
    # Read material properties and target nuclides for fraction application
    # ao_mats = {}
    ao_fr_mats = {}
    wo_fr_mats = {}
    for mat in materials:
        # ao_mats[mat] = mat.get_mass_density()
        ao_fr_mats[mat] = mat.get_ao_fraction()
        wo_fr_mats[mat] = mat.get_wo_fraction()
    target_nucs = {}
    for (mat, target) in zip(materials, fracs_target):
        if target is None:
            target_nucs[mat] = []
            continue
        elif type(target) == str:
            if target.isalpha():
                element = openmc.Element(target)
                element_nucs = []
                for nuc in element.expand(1, "ao"):
                    element_nucs += [nuc[0]]
            else:
                element_nucs = [target]
        else:
            element_nucs = target
        target_nucs[mat] = element_nucs
    
    norm_wgt = []
    def process_new_frac_target(mat, p_t):
        if not target_nucs[mat]:
            return 1
        if p_t == 'ao' or p_t == 'vo':
            sum_ao_fracs = np.sum([ao_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])
            return 1 / sum_ao_fracs if sum_ao_fracs > 0 else 1
            # return 1 / np.sum([ao_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])
        elif p_t == 'wo':
            sum_wo_fracs = np.sum([wo_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])
            return 1 / sum_wo_fracs if sum_wo_fracs > 0 else 1
            # return 1 / np.sum([wo_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])
        # elif p_t == 'vo':
        #     return 1 / np.sum([ao_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])
    
    # Process weights based on ao
    for mat in materials:
        norm_wgt += [process_new_frac_target(mat, percent_type)]
    
    fracs = np.array([frac * wgt if frac is not None else None for (frac,wgt) in zip(fracs,norm_wgt)])
    
    # If one of the fractions is None, calculate it to ensure the fractions sum to 1
    if None in fracs:
        index_none = np.argwhere(fracs == None)
        fracs[index_none] = 0
        fracs[index_none] = 1 - np.sum(fracs)
    else:
        if not np.abs(np.sum(fracs)-1) < 1e6:
            warnings.warn(f"Resulting weights do not sum to one: {np.sum(wgts)}.\n Please set set one of 'fracs' to None for automatic correction")
    
    # Scale weights based on type of fraction
    amms = np.asarray([mat.average_molar_mass for mat in materials])
    mass_dens = np.asarray([mat.get_mass_density() for mat in materials])
    if percent_type == 'ao':
        wgts = fracs * amms / mass_dens
        wgts /= np.sum(wgts)
    elif percent_type == 'wo':
        wgts = fracs / mass_dens
        wgts /= np.sum(wgts)
    elif percent_type == 'vo':
        wgts = fracs
    
    # Mix materials
    nuclides_per_barncm = defaultdict(float)
    for (mat, wgt) in zip(materials, wgts):
        for nuc, atoms_per_bcm in mat.get_nuclide_atom_densities().items():
            nuc_per_bmc = wgt * atoms_per_bcm
            nuclides_per_barncm[nuc] += nuc_per_bmc
    nuclide_ao_fr_per_submat = defaultdict(float)
    for nuc, _ in nuclides_per_barncm.items():
        nuclide_ao_fr_per_submat[nuc] = [0] * len(wgts)
    for (mat, wgt, index) in zip(materials, wgts, range(len(wgts))):
        for nuc, atoms_per_bcm in mat.get_nuclide_atom_densities().items():
            nuc_per_bmc = wgt * atoms_per_bcm
            nuclide_ao_fr_per_submat[nuc][index] = nuc_per_bmc / nuclides_per_barncm[nuc]
    
    if return_wgts:        
        return nuclides_per_barncm, nuclide_ao_fr_per_submat, wgts
    else:        
        return nuclides_per_barncm, nuclide_ao_fr_per_submat
 

class GeneralizedKalmanFilter:
    def __init__(self, n_states=1, n_measurements=1, transform_type='identity', 
                 custom_transform=None, custom_inverse=None, custom_jacobian=None):
        """
        Generalized Kalman Filter. 
        Defaults to a 1D (0-th order) noise-free running average filter.
        """
        self.n_states = n_states
        self.n_measurements = n_measurements
        self.transform_type = transform_type  # Ensure transform type is saved
        self.custom_jacobian = custom_jacobian
        
        # State estimates
        self.x = np.zeros((n_states, 1))          
        self.x_pred = np.zeros((n_states, 1))     
        
        # Covariance matrices
        self.P = np.eye(n_states)                 
        self.M = np.eye(n_states)                 
        self.K = np.zeros((n_states, n_measurements)) 
        
        self.initialized = False
        self.n_iterations = 0
        
        # Set up measurement space transforms
        self._setup_transforms(transform_type, custom_transform, custom_inverse)
        
    def _setup_transforms(self, transform_type, custom_transform, custom_inverse):
        if transform_type == 'custom':
            if not custom_transform or not custom_inverse:
                raise ValueError("Both custom_transform and custom_inverse are required for 'custom' type.")
            self.transform = custom_transform
            self.inverse_transform = custom_inverse
        elif transform_type in ['square_root', 'sqrt']:
            self.transform = lambda z: np.sqrt(z)
            self.inverse_transform = lambda z: z**2
        elif transform_type == 'square':
            self.transform = lambda z: z**2
            self.inverse_transform = lambda z: np.sqrt(z)
        elif transform_type == 'anscombe_3_8':
            self.transform = lambda z: 2 * np.sqrt(z + 3/8)
            self.inverse_transform = lambda z: (np.maximum(z, 0) / 2)**2 - 3/8
        elif transform_type == 'anscombe_1_8':
            self.transform = lambda z: 2 * np.sqrt(z + 3/8)
            self.inverse_transform = lambda z: (np.maximum(z, 0) / 2)**2 - 1/8
        elif transform_type == 'identity':
            self.transform = lambda z: z
            self.inverse_transform = lambda z: z
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")

    def set_initial_state(self, x_initial, P_initial):
        self.x = np.array(x_initial).reshape(self.n_states, 1)
        self.P = np.array(P_initial).reshape(self.n_states, self.n_states)
        self.initialized = True
        self.n_iterations = 0

    def predict(self, Phi=None, C=None, Gamma=None, Q=None):
        if not self.initialized:
            raise RuntimeError("Filter must be initialized via set_initial_state or first measurement update.")
            
        if Phi is None:
            Phi = np.eye(self.n_states)
        else:
            Phi = np.array(Phi).reshape(self.n_states, self.n_states)
        
        if C is not None:
            C = np.array(C).reshape(self.n_states, 1)
            self.x_pred = (Phi @ self.x) + C
        else:
            self.x_pred = Phi @ self.x
        
        if Q is not None:
            Q_arr = np.array(Q)
            if Q_arr.ndim == 0:  
                Q = Q_arr.reshape(1, 1)
            else:
                Q = Q_arr.reshape(Q_arr.shape[0], -1) if Q_arr.ndim == 1 else Q_arr
                
            if Gamma is not None:
                Gamma = np.array(Gamma).reshape(self.n_states, -1)
                process_noise_cov = Gamma @ Q @ Gamma.T
            else:
                process_noise_cov = Q
                
            self.M = (Phi @ self.P @ Phi.T) + process_noise_cov
        else:
            self.M = Phi @ self.P @ Phi.T
    
    def update(self, z, H=None, R=None):
        # 1. Standardize the input as a numpy array without stripping away extra states/vectors
        z_arr = np.atleast_1d(z)
        
        # 2. Apply your element-wise non-linear transformations across the vector
        # This keeps the full array length intact for tracking multi-state vectors
        z_transformed = self.transform(z_arr).reshape(self.n_measurements, 1)

        if not self.initialized:
            if H is None:
                H = np.eye(self.n_measurements, self.n_states)
            else:
                H = np.array(H).reshape(self.n_measurements, self.n_states)
            self.x = np.linalg.pinv(H) @ z_transformed
            self.P = np.linalg.pinv(H) @ R @ np.linalg.pinv(H).T
            self.initialized = True
            self.n_iterations = 1
            return

        self.n_iterations += 1

        if H is None:
            H = np.eye(self.n_measurements, self.n_states)
        else:
            H = np.array(H).reshape(self.n_measurements, self.n_states)
            
        if R is None:
            R = np.eye(self.n_measurements)
        else:
            R = np.array(R).reshape(self.n_measurements, self.n_measurements)
            
        # 3. MULTI-DIMENSIONAL ERROR VARIANCE TRANSFORM (Element-wise scaling)
        # Instead of picking up z_scalar, evaluate across the entire active vector array
        if self.transform_type in ['square_root', 'sqrt']:
            # Correct derivative: 1 / (2 * sqrt(z))
            scaling_vector = 1.0 / (2.0 * np.sqrt(np.maximum(z_arr, 1e-5)))
            R_scaled = R * np.outer(scaling_vector, scaling_vector)
            
        elif self.transform_type == 'square':
            scaling_vector = 2.0 * z_arr
            R_scaled = R * np.outer(scaling_vector, scaling_vector)
            
        elif self.transform_type in ['anscombe_3_8', 'anscombe_1_8']:
            c_const = 3/8 if self.transform_type == 'anscombe_3_8' else 1/8
            # Correct derivative: 1.0 / sqrt(z + c)
            scaling_vector = 1.0 / np.sqrt(np.maximum(z_arr + c_const, 1e-5))
            R_scaled = R * np.outer(scaling_vector, scaling_vector)

        elif self.transform_type == 'custom':
            h = 1e-20
            # Broadcast the complex step approximation securely across the entire state vector input
            df_dz = np.imag(self.transform(z_arr + h*1j)) / h
            R_scaled = R * np.outer(df_dz, df_dz)
        else:
            R_scaled = R
            
        R = R_scaled.reshape(self.n_measurements, self.n_measurements)
        
        # 4. Standard linear multi-dimensional Kalman updating loop
        S = (H @ self.M @ H.T) + R
        self.K = self.M @ H.T @ np.linalg.inv(S)
        
        innovation = z_transformed - (H @ self.x_pred)
        self.x = self.x_pred + (self.K @ innovation)
        
        I = np.eye(self.n_states)
        self.P = (I - (self.K @ H)) @ self.M
    
    @property
    def optimal_value(self):
        # Safely unwrap multi-dimensional arrays or nested matrices down to a single float scalar
        if isinstance(self.x, np.ndarray):
            x_scalar = float(self.x.item() if hasattr(self.x, 'item') else self.x.ravel()[0])
        else:
            x_scalar = float(self.x)
            
        physical_value = float(self.inverse_transform(x_scalar))
        
        if self.transform_type in ['square_root', 'sqrt']:
            dh_dx = 2.0 * x_scalar
        elif self.transform_type == 'square':
            dh_dx = 1.0 / (2.0 * np.sqrt(x_scalar)) if x_scalar > 0 else 1e-5
        elif self.transform_type in ['anscombe_3_8', 'anscombe_1_8']:
            dh_dx = x_scalar / 2.0
        elif self.transform_type == 'custom':
            if self.custom_jacobian is not None:
                dh_dx = self.custom_jacobian(x_scalar)
            else:
                h_step = 1e-20
                dh_dx = np.imag(self.inverse_transform(x_scalar + h_step * 1j)) / h_step
        else:
            dh_dx = 1.0
            
        transformed_variance = float(np.ravel(self.P)[0])
        physical_variance = (dh_dx ** 2) * transformed_variance
        physical_uncertainty = np.sqrt(max(physical_variance, 0.0))
        
        return physical_value, physical_uncertainty


    @property
    def linear_P(self):
        return self.P


class CDI:
    """
    Class to perform critical density iteration (CDI) inside the OpenMC. CDI is a method used to find the critical concentration of a nuclide in a material such that the effective multiplication factor (k_eff) of a nuclear system is equal to a target value (usually 1). The class provides a method to perform CDI by iteratively adjusting the concentration of the nuclide and running the number of prescribed batches.
    
    The class allows for more convenient usage of CDI within a depletion simulation, as it can be called as a function and maintains the state of the last concentration value for subsequent calls.
    
    Returns
    -------
    openmc.model.model
        Updated model with converged concentrations.

    .. versionadded:: 0.15.4
    """
    def __init__(self, model, iso=None, batches=30, bracket=None, 
                        materials=None, initial_value=1.0, target=1.,
                        mat_builder=None, prefer_model_xml=False,
                        max_step_change=4, debug=False, force_initial_value=False,
                        other_execution_functions=None, activate_all_tallies=False,
                        skip8890=True, dynamic_sigma_min=None, kf_kwargs={}):

            self.nuc_fractions = None
            self.nuc_fractions_replace = None
            self.activate_all_tallies = activate_all_tallies
            
            # Check input arguments and prepare the model for CDI
            if mat_builder is not None:
                mat_builder_res = mat_builder(initial_value)
                if type(mat_builder_res) == dict:
                    keys = mat_builder_res.keys()
                else:
                    raise ValueError("mat_builder function must return a dictionary")
                if "materials" in keys:
                    materials = [mat for mat in mat_builder_res["materials"]]
                # else:
                #     raise ValueError("'materials' not found in mat_builder return dictionary")
                    if iso is None:
                        iso = []
                        for mat in materials:
                            for nuc in mat.nuclides:
                                if nuc.name not in iso:
                                    iso += [nuc.name]
                if "nuc_fractions" in keys:
                    self.nuc_fractions = mat_builder_res["nuc_fractions"]
                if "nuc_fractions_replace" in keys:
                    self.nuc_fractions_replace = mat_builder_res["nuc_fractions_replace"]
            # if iso is None:
            #     raise ValueError("'iso' argument is empty")
            if batches is not None:
                cv.check_type('batches', batches, Integral)
            else:
                raise ValueError("'batches' argument is empty")
            if bracket is not None:
                cv.check_iterable_type('bracket', bracket, Real)
                cv.check_length('bracket', bracket, 2)
                cv.check_less_than('bracket values', bracket[0], bracket[1])
            cv.check_greater_than("max_step_change", max_step_change, 1.0)
            cv.check_type('initial_value', initial_value, Real)
            
            
            self.mat_ids = None
            if materials is not None:
                self.mat_ids=[]
                for mat in materials:
                    self.mat_ids += [mat.id]
                
            # Add CDI batches to the simulation settings        
            if model.settings.inactive is None:
                model.settings.inactive = 0
            model.settings.inactive += batches
            model.settings.batches += batches
            
            # Create tallies if not already created
            tally_ids = [tally.id for tally in model.tallies]
            if 8888 not in tally_ids: 
                tallyTest = Tally(tally_id=8888, name="CDI_tally1")
                tallyTest.scores = ["nu-fission", "absorption", "nu-scatter", "scatter"]
                model.tallies += [tallyTest]
            if 8889 not in tally_ids:
                tallyTest2 = Tally(tally_id=8889, name="CDI_tally2")
                if iso is None:
                    raise ValueError("'iso' is empty")
                tallyTest2.nuclides = iso
                tallyTest2.scores = ["nu-fission", "absorption", "nu-scatter", "scatter"]
                if materials is not None:
                    tallyTest2.filters = [MaterialFilter(materials,filter_id=8889)]
                model.tallies += [tallyTest2]
            
            # Export modified model
            if prefer_model_xml:
                model.export_to_model_xml()
            else:
                model.export_to_xml()
            
            self.batches = batches
            self.starting_batch = max(model.settings.inactive - self.batches, 1)
                
            self.other_exec = []
            if other_execution_functions is not None:
                for exec_func_properties in other_execution_functions:
                    exec_func, position, exec_strategy = exec_func_properties
                    exec_start, exec_end = position
                    if exec_start == "CDI":
                        exec_start = self.starting_batch
                    elif type(exec_start) == int:
                        exec_start = exec_start
                    else:
                        exec_start = 0
                    
                    if type(exec_end) == int:
                        exec_end = exec_end
                    else:
                        exec_end = model.settings.inactive
                        
                    if exec_strategy not in [0,1,2]:
                        raise ValueError(f"Provided execution strategy {exec_strategy} is not valid, must be 0, 1, or 2")
                    if not callable(exec_func):
                        raise ValueError(f"Provided execution function {exec_func} is not callable")
                    self.other_exec.append([exec_func, [exec_start, exec_end], exec_strategy])
            
            kf_kwargs.setdefault('n_states', 1)
            kf_kwargs.setdefault('n_measurements', 1)
            kf_kwargs.setdefault('transform_type', 'identity')
            self.kf = GeneralizedKalmanFilter(**kf_kwargs)
            
            self.model = model
            self.iso = iso
            self.bracket = bracket
            self.materials = materials
            self.initial_value = initial_value
            self.target = target
            self.mat_builder = mat_builder
            self.prefer_model_xml = prefer_model_xml
            self.max_step_change = max_step_change
            self.debug = debug
            self.last_result = None
            self.force_initial_value = force_initial_value
            self.dynamic_sigma_min = dynamic_sigma_min
            
            self.guesses = []
            self.guess_unc = []
            self.guess_ks = []
            self.f = 1
            self.g = 1
            self.f_prev = 1
            self.prev_res = [[],[],[]]
            self.prev_leak = 0
            self.skip_steps = False
            self.skip8890 = skip8890
            
    def _get_model(self):
        return self.model
    
    def __call__(self):
        comm.barrier()
        if not openmc.lib.is_initialized:
            if self.debug is True: print("Initializing OpenMC library for CDI...")
            openmc.lib.init(intracomm=comm)
        openmc.lib.reset()
        openmc.lib.simulation_init()
        
        # Set required tallies to active
        for t_id, _tally in openmc.lib.tallies.items():
            if t_id == 8888 or t_id == 8889 or t_id == 8890:
                _tally.active = True
        if self.activate_all_tallies:
            for t_id, _tally in openmc.lib.tallies.items():
                _tally.active = True
        
        # Run simulation
        for _ in openmc.lib.iter_batches():
            next(self)
        openmc.lib.simulation_finalize()
        
        # Final output extraction using class properties
        final_val, final_unc = self.kf.optimal_value
        print(f"CDI: Solve converged to value: {final_val} +/- {final_unc}")
        
        if len(self.guesses) > 1:
            cdi_Cs = np.array(self.guesses)[1:]
            cdi_ks = np.array(self.guess_ks)[1:] * 1e5
            
            def linF(x, n, k):
                return n + k * x
                
            cdi_res, cdi_res_cov = sopt.curve_fit(linF, cdi_Cs, cdi_ks, sigma=np.array(self.guess_unc)[1:])
            cdi_res_sig = np.sqrt(np.diag(cdi_res_cov))
            print(f"CDI: Estimated reactivity coefficient: {cdi_res[1]:.05e} +/- {cdi_res_sig[1]:.05e} pcm/unit")
        else:
            print("CDI: Not enough tracking history captured for reactivity coefficient fitting.")
        
        return self.model
    
    
    def __next__(self):
        M = openmc.lib.current_batch()
        for exec_func, exec_range, exec_strategy in self.other_exec:
            if exec_strategy not in [0,2]:
                continue
            exec_start, exec_end = exec_range
            if M >= exec_start and M < exec_end:
                exec_func()
        
        # Get tallies
        # Tally results are added (summed) in each batch, batch result is the difference
        _tallies = copy.copy(openmc.lib.tallies)
        # global_tallies = copy.copy(openmc.lib.global_tallies())
        
        if not self.skip8890:
            CDI_tally_ids = {8888: 0, 8889: 1, 8890: 2}
            self.curr_res = [[],[],[]]
        else:
            CDI_tally_ids = {8888: 0, 8889: 1}
            self.curr_res = [[],[]]
        if M == 1:
            for _tally in _tallies.values():
                if _tally.id in CDI_tally_ids:
                    # prev_res += [_tally.results - _tally.results]
                    # curr_res[0 if _tally.id == 8888 else 1] = copy.copy(_tally.results)
                    self.curr_res[CDI_tally_ids[_tally.id]] = copy.copy(_tally.results)
                    self.prev_res[CDI_tally_ids[_tally.id]] = copy.copy(_tally.results) * 0.0
        else:
            for _tally in _tallies.values():
                if _tally.id in CDI_tally_ids:
                    self.curr_res[CDI_tally_ids[_tally.id]] = copy.copy(_tally.results) - self.prev_res[CDI_tally_ids[_tally.id]]
                    self.prev_res[CDI_tally_ids[_tally.id]] = copy.copy(_tally.results)
        # Leakage is a running average
        # leak = global_tallies[3][0]*M - prev_leak
        # prev_leak = global_tallies[3][0]*M
                
        # Only change concentrations during the inactive CDI batches
        if ((M < self.starting_batch + self.batches)
            and (M >= self.starting_batch)
            and (not self.skip_steps)):
            
            if self.debug is True: print(f"\n Batch: {M}")
            
            if M == self.starting_batch: 
                absolute_initial_guess = float(self.initial_value)  
                
                # 1. Transform the initial physical guess into the filter's tracking domain
                if self.kf.transform_type in ['square_root', 'sqrt']:
                    transformed_initial_state = np.sqrt(absolute_initial_guess)
                    dh_dx = 2.0 * transformed_initial_state
                elif self.kf.transform_type == 'square':
                    transformed_initial_state = absolute_initial_guess ** 2
                    dh_dx = 1.0 / (2.0 * np.sqrt(transformed_initial_state)) if transformed_initial_state > 0 else 1e-5
                elif self.kf.transform_type in ['anscombe_3_8', 'anscombe_1_8']:
                    c_const = 3/8 if self.kf.transform_type == 'anscombe_3_8' else 1/8
                    transformed_initial_state = 2.0 * np.sqrt(absolute_initial_guess + c_const)
                    dh_dx = transformed_initial_state / 2.0
                elif self.kf.transform_type == 'custom':
                    transformed_initial_state = float(self.kf.transform(absolute_initial_guess))
                    if self.kf.custom_jacobian is not None:
                        dh_dx = self.kf.custom_jacobian(transformed_initial_state)
                    else:
                        h_step = 1e-20
                        dh_dx = np.imag(self.kf.inverse_transform(transformed_initial_state + h_step * 1j)) / h_step
                else:
                    transformed_initial_state = absolute_initial_guess
                    dh_dx = 1.0

                # 2. COMPUTE THE EXACT SCALE VARIANCE REQUIRED FOR PERFECT HANDOVER
                # To clear the initial guess entirely without causing non-linear squashing,
                # the initial variance must scale with the physical measurement variance target.
                
                # Define a safe, uninformative physical variance baseline relative to the guess size
                physical_variance_baseline = 100.0 * (absolute_initial_guess ** 2)

                if self.kf.transform_type in ['identity', 'custom']:
                    initial_variance = physical_variance_baseline
                else:
                    # Scale the physical baseline cleanly relative to the Jacobian coordinate distortion
                    # This protects against numerical underflow bugs when R_measure is small
                    initial_variance = physical_variance_baseline / (dh_dx ** 2)
                
                # Seed the filter states safely
                self.kf.set_initial_state(x_initial=np.array([[transformed_initial_state]]), 
                                          P_initial=np.array([[initial_variance]]))
                
                # Seed predictive arrays manually matching the transformed tracking space
                self.kf.x_pred = copy.deepcopy(self.kf.x)
                self.kf.M = copy.deepcopy(self.kf.P)
                
                self.kf.initialized = True
                self.f = 1.0
                self.f_prev = 1.0
            else:
                # --- PAPER-ACCURATE CONSTANT PARAMETER ESTIMATION (Q=0) ---
                # According to Equation (5), the filter tracks a constant scalar value.
                # Process noise is 0.0 unless an explicit floor is requested by the user.
                if self.dynamic_sigma_min is not None:
                    q_physical = float(self.dynamic_sigma_min ** 2)
                else:
                    q_physical = 0.0
                
                if self.debug is True and q_physical > 0:
                    print(f"Predictive step: Propagating user physical process noise Q = {q_physical:.02f} ppm^2")
                
                # Pass directly to the engine. With Q=0, P_next = P_old, allowing 
                # successive measurement updates to smoothly drive P down toward zero.
                self.kf.predict(Phi=np.array([[1.0]]), Q=q_physical)
            
            ### P = production of neutrons, L = loss of neutrons
            # Neutrons produced by fission (prompt and delayed)
            P_fiss = self.curr_res[0][0][0][1]
            P_fiss = P_fiss if P_fiss > 0 else 0
            # Additional neutrons produced by (n,xn) reactions
            P_nxn = self.curr_res[0][0][2][1] - self.curr_res[0][0][3][1]
            P_nxn = P_nxn if P_nxn > 0 else 0
            
            # Total neutron absorption             
            L_abs = self.curr_res[0][0][1][1]                                
            L_abs = L_abs if L_abs > 0 else 0
            # WARNING: L_abs - P_nxn + L_leak === 1; by OpenMC def
            # >0, for when floating point errors cause negative values
            
            ### Neutron leakage fraction
            # It is calculated implicitly by OpenMC as 1 = P_nxn + L_abs + L_leak
            # L_leak = leak if leak > 0 else 0
            L_leak = 1 - (L_abs - P_nxn)

            def parse_flagged_nuclide_tally(tally_results, nuc_fractions):
                P_fiss_nucs = np.sum((tally_results[...,0::4,1]) * np.array(nuc_fractions))
                P_nxn_nucs = np.sum((tally_results[...,2::4,1] - tally_results[...,3::4,1]) * np.array(nuc_fractions))
                L_abs_nucs = np.sum((tally_results[...,1::4,1]) * np.array(nuc_fractions))
                return P_fiss_nucs, P_nxn_nucs, L_abs_nucs
            
            # Same as above but summed for all flagged nuclides, weighted by weights if provided by 'mat_builder'

            P_fiss_nucs = 0
            P_fiss_nucs_absolute = 0
            P_nxn_nucs = 0
            P_nxn_nucs_absolute = 0
            L_abs_nucs = 0
            L_abs_nucs_absolute = 0
            if self.materials:
                for index, mat in enumerate(self.materials):
                    if self.debug: print(f"Tally partial fractions for mat with id={mat.id}:",np.array(self.nuc_fractions[index]))
                    _P_fiss_nucs, _P_nxn_nucs, _L_abs_nucs = parse_flagged_nuclide_tally(np.array(self.curr_res[1][index]),
                                                               self.nuc_fractions[index])
                    P_fiss_nucs += _P_fiss_nucs
                    P_nxn_nucs += _P_nxn_nucs
                    L_abs_nucs += _L_abs_nucs
                    P_fiss_nucs_absolute += np.abs(_P_fiss_nucs)
                    P_nxn_nucs_absolute += np.abs(_P_nxn_nucs)
                    L_abs_nucs_absolute += np.abs(_L_abs_nucs)
                    if len(self.curr_res) == 3:
                        if len(self.curr_res[2]) != 0:
                            _P_fiss_nucs, _P_nxn_nucs, _L_abs_nucs = parse_flagged_nuclide_tally(np.array(self.curr_res[2][index]),
                                                                    self.nuc_fractions_replace[index])
                            P_fiss_nucs -= _P_fiss_nucs
                            P_nxn_nucs -= _P_nxn_nucs
                            L_abs_nucs -= _L_abs_nucs
                            P_fiss_nucs_absolute += np.abs(_P_fiss_nucs)
                            P_nxn_nucs_absolute += np.abs(_P_nxn_nucs)
                            L_abs_nucs_absolute += np.abs(_L_abs_nucs)
            else:
                _P_fiss_nucs, _P_nxn_nucs, _L_abs_nucs = parse_flagged_nuclide_tally(np.array(self.curr_res[1]),
                                                           (self.nuc_fractions if self.nuc_fractions is not None else 1) )
                P_fiss_nucs += _P_fiss_nucs
                P_nxn_nucs += _P_nxn_nucs
                L_abs_nucs += _L_abs_nucs
                P_fiss_nucs_absolute += np.abs(_P_fiss_nucs)
                P_nxn_nucs_absolute += np.abs(_P_nxn_nucs)
                L_abs_nucs_absolute += np.abs(_L_abs_nucs)
                if len(self.curr_res) == 3:
                    if len(self.curr_res[2]) != 0:
                            _P_fiss_nucs, _P_nxn_nucs, _L_abs_nucs = parse_flagged_nuclide_tally(np.array(self.curr_res[2]),
                                                                    self.nuc_fractions_replace)
                            P_fiss_nucs -= _P_fiss_nucs
                            P_nxn_nucs -= _P_nxn_nucs
                            L_abs_nucs -= _L_abs_nucs
                            P_fiss_nucs_absolute += np.abs(_P_fiss_nucs)
                            P_nxn_nucs_absolute += np.abs(_P_nxn_nucs)
                            L_abs_nucs_absolute += np.abs(_L_abs_nucs)
            
            # Predict concentration change
            bot = L_abs_nucs - P_fiss_nucs/self.target - P_nxn_nucs
            top = P_fiss/self.target - 1 + bot 
            if bot == 0:
                print(f"CDI: Combined effect of absorption, fission and (n,xn) reaction of flagged is 0, skipping step {M}")
            else:
                g_est = top / bot  
                
                # Fetch absolute concentration cleanly
                absolute_current_concentration = float(self.f_prev * self.initial_value)
                
                                # 1. Compute statistical error from Monte Carlo
                rel_err_MC = 1 / np.sqrt(self.model.settings.particles * (self.model.settings.generations_per_batch if self.model.settings.generations_per_batch is not None else 1))
                prod = P_fiss + P_nxn
                loss = L_abs + L_leak
                
                # sig_nucs = rel_err_MC * np.sqrt(prod * (P_fiss_nucs**2 / P_fiss_nucs_absolute if P_fiss_nucs_absolute != 0 else 0) / self.target
                #                                 + prod * (P_nxn_nucs**2 / P_nxn_nucs_absolute if P_nxn_nucs_absolute != 0 else 0)
                #                                 + loss * (L_abs_nucs**2 / L_abs_nucs_absolute if L_abs_nucs_absolute != 0 else 0))
                sig_nucs = rel_err_MC * np.sqrt(
                    (P_fiss/self.target) * P_fiss_nucs_absolute 
                    + P_fiss * P_nxn_nucs_absolute 
                    + loss * L_abs_nucs_absolute
                )
                sig_fiss = rel_err_MC * np.sqrt(prod * P_fiss)
                
                # --- NEW CORRELATION AND COVARIANCE PROPAGATION INJECTION ---
                # 1. Estimate rho based on your theoretical fraction bounds (Page 6 of paper)
                # We enforce an un-biased nu_bar approximation (typically ~2.5 for PWR profiles)
                nu_bar_est = 2.5
                if P_fiss > 0 and bot > 0:
                    rho_fiss_nucs = -np.sqrt(np.abs(bot / (P_fiss / nu_bar_est)))
                else:
                    rho_fiss_nucs = 0.0
                    
                # 2. Compute the raw covariance metric explicitly
                cov_fiss_nucs = rho_fiss_nucs * sig_fiss * sig_nucs
                
                # 3. Calculate full correlated multi-variable Taylor expansion (Delta Method)
                var_term_fiss = (sig_fiss / (self.target * bot)) ** 2
                var_term_nucs = ((P_fiss / self.target - 1) * sig_nucs / (bot ** 2)) ** 2
                
                # Cross covariance mapping term
                var_term_cross = -2.0 * ((P_fiss / self.target - 1) / (self.target * (bot ** 3))) * cov_fiss_nucs
                
                # Sum terms securely and enforce a mathematical floor check
                sig_res_squared = var_term_fiss + var_term_nucs + var_term_cross
                sig_res = np.sqrt(max(sig_res_squared, 1e-12))
                
                sig_g_est = np.abs(sig_res)
                
                z_measurement = absolute_current_concentration * float(g_est)
                sig_absolute_est = absolute_current_concentration * float(sig_g_est)
                p_measure = float(sig_absolute_est ** 2)
                
                # Enforce step limits on the absolute target value if out of safe bounds
                if (g_est < 1 / (self.max_step_change - 0.5)) or (g_est > self.max_step_change):
                    if g_est <= 1 / (self.max_step_change - 0.5):
                        g_est_bounded = 1 / (self.max_step_change - 0.5)
                    else:
                        g_est_bounded = self.max_step_change
                    
                    z_measurement = absolute_current_concentration * float(g_est_bounded)
                    # Inflate measurement uncertainty because the linear step assumption failed
                    p_measure = absolute_current_concentration**2 #p_measure * 100.0 

                # Handle tracking bracket bounds constraints in physical units if specified
                if self.bracket is not None:
                    if z_measurement > self.bracket[1]:
                        z_measurement = self.bracket[1]
                    elif z_measurement < self.bracket[0]:
                        z_measurement = self.bracket[0]
                
                if self.debug is True: 
                    print(f"Propagating absolute uncertainty: R_measure={p_measure}, Measurement Target={z_measurement}")

                # --- 1. EXECUTE THE UPDATE ONLY USING PHYSICAL METRICS ---
                # Let the filter engine handle the internal transformation and noise scaling
                z_input = np.array([[z_measurement]])
                R_input = np.array([[p_measure]])
                
                self.kf.update(z=z_input, R=R_input)
                
                # --- 2. EXTRACT HEALTHY PHYSICAL CONCENTRATIONS VIA PROPERTIES ---
                # This uses the Delta Method property to cleanly read the physical value and uncertainty
                absolute_filtered_concentration, physical_uncertainty = self.kf.optimal_value
                
                # --- 3. RE-SYNCHRONIZE MULTIPLIERS FOR MATERIAL BUILDER ENDPOINTS ---
                self.f = absolute_filtered_concentration / self.initial_value
                self.g = self.f / self.f_prev
                self.f_prev = copy.copy(self.f)

                # --- 4. RECORD RAW PHYSICAL METRICS IN THE HISTORY LOGS ---
                # Save the physical value and physical uncertainty so curve_fit receives the correct scales
                self.guesses += [absolute_filtered_concentration]
                self.guess_unc += [physical_uncertainty]
                self.guess_ks += [(P_fiss - P_nxn) / (L_abs + L_leak - P_nxn)]

                if self.debug is True:
                    # Print the internal transformed variance alongside the true physical uncertainty
                    print(f"Batch System Transformed Variance P: {float(np.ravel(self.kf.P))}")
                    print(f"Batch values top: {top}, bot: {bot}")
                    print(f"Batch estimated correction - 1: {g_est-1}")
                    print(f"Batch filtered correction - 1: {self.g-1}")
                    print(f"Batch filtered Absolute Value: {absolute_filtered_concentration} +/- {physical_uncertainty} ppm")


                # Rebuild the material with the given function at provided concentration
                if self.mat_builder is not None:
                    mat_builder_res = self.mat_builder(self.f * self.initial_value)
                    keys = mat_builder_res.keys() if type(mat_builder_res) == dict else None
                    if keys:
                        if "materials" in keys:
                            materials = [mat for mat in mat_builder_res["materials"]]
                        if "nuc_fractions" in keys:
                            self.nuc_fractions = mat_builder_res["nuc_fractions"]
                        if "nuc_fractions_replace" in keys:
                            self.nuc_fractions_replace = mat_builder_res["nuc_fractions_replace"]
                
                # Update densities on C API side
                for rank in range(comm.size):
                    C_API_mats = comm.bcast(openmc.lib.materials, root=rank)
                    for mat in C_API_mats:
                        if self.materials is not None:
                            if int(mat) not in self.mat_ids:
                                continue
                        nuclides=[]
                        densities=[]
                        all_nuc = np.array(C_API_mats[int(mat)].nuclides)
                        mat_internal = C_API_mats[int(mat)]
                        
                        if self.mat_builder is None: # Change all materials with the flagged nuclides, as in option A
                            all_dens = (np.array(C_API_mats[int(mat)].densities)).astype(float)
                            for nuc in all_nuc:
                                val = float((all_dens[all_nuc==str(nuc)])[0])
                                # If nuclide is zero, do not add to the problem.
                                if val > 0:
                                    if str(nuc) in self.iso:
                                        val *= self.g
                                    nuclides.append(nuc)
                                    densities.append(val)
                                elif str(nuc) in self.iso:
                                    val *= self.g
                                    nuclides.append(nuc)
                                    densities.append(val)
                        else:
                            for matpy in materials: # Change only the flagged materials
                                matpy_nuc_dict = matpy.get_nuclide_atom_densities()
                                if matpy.id == int(mat):
                                    for nuc in all_nuc:
                                        val = matpy_nuc_dict.get(str(nuc),0)
                                        # If nuclide is zero, do not add to the problem.
                                        # 16 bit float limit, to avoid overflow in OpenMC C API
                                        # May need to be changed to > 1e-38 or similar
                                        if val > 0: 
                                            nuclides.append(nuc)
                                            densities.append(val)
                                    break
                        mat_internal.set_densities(nuclides, densities)

        for exec_func, exec_range, exec_strategy in self.other_exec:
            if exec_strategy not in [1,2]:
                continue
            exec_start, exec_end = exec_range
            if M >= exec_start and M < exec_end:
                exec_func()
        
        if M == self.model.settings.inactive:
            openmc.lib.reset()
        
        return 0
    
    def run(self):
        return self()