from collections import defaultdict, namedtuple, Counter
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


def critical_density_iteration(model, iso=None, batches=None, bracket=None, 
                        materials=None, initial_value=None, target=1., perfer_all_xml=True, debug=False): # TODO allow for change in number of neutrons and then return settings to previous values
    """
    Runs a simulation where 'iso' nuclide values converge in such a way to obtain the desired k_eff.
    Operator.model materials are updated in the process
    Optional initial value is the value given in your material building process, must be strictly bigger than 0.
    All 'iso' nuclides are multiplied by the same scaling factor.
    Higher (>10 000) particle numbers in 'openmc.settings' are recommended for more accurate simulation.
    Atleast 30 batches are required for adequate convergence, recommended >50.

    Parameters
    ----------
    model: openmc.model, required
    iso: array of str, required
        Nuclide name, ex. ["B10", "B11"]
    batches: int, optional
        Number of inactive batches added to the begining of simulation where
        'iso' concentration converges.
        Defaults to 50 extra inactive cycles.
    bracket: array of 2 floats > 0, optional
        Lower and upper bounds for concentrations.
        Inteded to be used in tandem with initial_value.
        Otherwise just a multiple of initial concentration at step 0.
    materials: materials in which nuclide concentrations are changed, optional
        Defaults to all materials.
    initial_value: float > 0, optional
        Only used in first call, used for intermediate critical concentration message output.
        Defaults to 1.0.
    target: float, optional
        Target k_eff, defaults to 1.0
    debug: Bool, optional
        Wether to print out batch number, tally results of each batch,
        batch k_absorption and current concentration.
        Defaults to False.

    Returns
    -------
    openmc.model.model, with updated critical density concentrations of flagged nuclides in flagged materials

    """
    if iso is None:
        raise ValueError("'iso' argument is empty")
    if initial_value is not None:
        cv.check_type('initial_value', initial_value, Real)
    else:
        raise ValueError("'initial_value' argument is empty")
    if batches is not None:
        cv.check_type('batches', batches, Integral)
    else:
        raise ValueError("'batches' argument is empty")
    if bracket is not None:
        cv.check_iterable_type('bracket', bracket, Real)
        cv.check_length('bracket', bracket, 2)
        cv.check_less_than('bracket values', bracket[0], bracket[1])
        
    #Create tallies if not already created
    if model.settings.inactive < batches + 10:
        model.settings.inactive += batches
        model.settings.batches += batches
    tally_ids = [tally.id for tally in model.tallies]
    if 8888 not in tally_ids or 8889 not in tally_ids:
        tallyTest = Tally(tally_id=8888, name="search_crit_conc_tally_1")
        tallyTest.scores = ["nu-fission", "absorption", "nu-scatter", "scatter"]
        model.tallies += [tallyTest]

        tallyTest2 = Tally(tally_id=8889, name="search_crit_conc_tally_2")
        if iso is None:
            raise ValueError("'iso' argument in conc_args is empty")
        tallyTest2.nuclides = iso
        tallyTest2.scores = ["absorption"]
        if materials is not None:
            tallyTest2.filters = [MaterialFilter(materials,filter_id=8888)]
        model.tallies += [tallyTest2]
    
    if not perfer_all_xml:
        model.export_to_xml()
    else:
        model.settings.export_to_xml()
        model.materials.export_to_xml()
        model.geometry.export_to_xml()
        if model.plots:
            model.plots.export_to_xml()
        if model.tallies:
            model.tallies.export_to_xml()

    if initial_value is None:
        initial_value = 1.0
    else:           
        initial_value = initial_value
        
    if materials is not None:
        mat_ids=[]
        for mat in materials:
            mat_ids += [mat.id]

    f = 1
    g = 1
    f_prev = 1
    prev_res = []
    prev_leak = 0
    skip_steps = False
    starting_batch = model.settings.inactive - batches
    # Initialize OpenMC library
    comm.barrier()
    # if not openmc.lib.is_initialized:
    openmc.lib.init(intracomm=comm)
    openmc.lib.reset()
    openmc.lib.simulation_init()
    # Run simulation
    for _ in openmc.lib.iter_batches():
        M = openmc.lib.current_batch()
        if M < starting_batch: continue
        # Only change concentrations during the additional batches
        if M <= starting_batch + batches and not skip_steps:
            #k = openmc.lib.keff()[0]
            talliez = copy.copy(openmc.lib.tallies)
            curr_res = []
            if M == starting_batch:
                for tally_ in talliez.values():
                    if tally_.id in [8888,8889]:
                        prev_res += [tally_.results - tally_.results]
            i=0
            for tally_ in talliez.values():
                if tally_.id in [8888,8889]:
                    curr_res += [tally_.results - prev_res[i]]
                    prev_res[i] = copy.copy(tally_.results)
                    i+=1
                    
            # Tally results are added (summed) in each batch - measurement is the difference
            glob_tall = copy.copy(openmc.lib.global_tallies())
            leak = glob_tall[3][0]*M - prev_leak
            prev_leak = glob_tall[3][0]*M
            
            P_fiss = curr_res[0][0][0][1]                               # Neutrons produced by fission (prompt and delayed)
            P_nxn = curr_res[0][0][2][1] - curr_res[0][0][3][1]         # Additional neutrons produced by (n,xn) reactions
            L_leak = (leak if leak > 0 else 0)                          # Neutron leakage fraction, very low, may happen to be negative due to floating point percision
            L_abs = curr_res[0][0][1][1]                                # Total neutron absorption
            # Total flagged nuclide absorption
            if materials is not None:
                L_abs_nucs = np.sum(np.array(np.sum(curr_res[1], axis=0)).T, axis=1)[1]
            else:
                L_abs_nucs = np.sum(np.array(curr_res[1][0]).T, axis=1)[1]
            if L_abs_nucs == 0:
                if not skip_steps: print(f"No nuclide absorption tallied, skipping from step {M} onwards")
                skip_steps = True
                continue
            
            # Predict concentration change
            top = (P_fiss/target + P_nxn) - (L_abs - L_abs_nucs) - (P_fiss + P_nxn) * L_leak
            bot = L_abs_nucs
            g_est = top / bot
            # Optimal following (Kalman filter for narrowing to a scalar value):
            if M == starting_batch: #Start the iteration at step 10, handled before, this is only K.f initialization
                x = 1
                p = 1e16
                p_n = 1e16
                p_measure = 1e16
            if (g_est >= 0.1 and g_est <= 10.0):
                rel_err_MC = 1/np.sqrt(model.settings.particles)
                prod = P_fiss + P_nxn
                loss = L_abs + prod * L_leak
                # rel_err = sqrt(1/N_part_tally) = 1/sqrt(N_tot) * sqrt(N_tot/N_part_tally) = rel_err_MC * sqrt(N_tot/N_part_tally) = rel_err_MC * sqrt(tot_tally/part_tally)
                # Sig = part_tally * rel_err = part_tally * rel_err_MC * sqrt(tot/part_tally) = rel_err_MC * sqrt(tot*part_tally)
                sig1 = rel_err_MC * (np.sqrt(prod * P_fiss)/target + np.sqrt(prod * P_nxn)) #sig for (P_fiss + P_nxn)/target
                sig2 = rel_err_MC * (np.sqrt(loss * L_abs) + np.sqrt(loss * L_abs_nucs) ) #sig for (L_abs - L_abs_nucs)
                #logic: prod = loss = abs + leak; leak = prod - abs
                #sig_leak = rel_err_MC*(np.sqrt(prod/P_fiss) + np.sqrt(prod/P_nxn)) + rel_err_MC*np.sqrt(loss/L_abs) #sig for L_leak
                #sig3 = (sig1/prod + sig_leak/loss) * prod * L_leak  #sig for (P_fiss + P_nxn)/target * L_leak; Sig = L_leak * prod * (rel_err(prod) + rel_err(L_leak)) / target
                sig3 = rel_err_MC * (np.sqrt(prod * P_fiss) + np.sqrt(prod * P_nxn)) * L_leak
                #Division by target must not influence relative errors
                rel_err_top = (np.abs(sig1) + np.abs(sig2) + np.abs(sig3)) / top
                rel_err_bot = rel_err_MC * np.sqrt(loss * L_abs_nucs) / bot
                rel_err_g_est = (rel_err_top + rel_err_bot)  * (1 + (100*np.exp(-(M - starting_batch)**2 / (batches / 6)) if (M - starting_batch) < (batches / 3) else 0)) #Slowly relax uncertainty, as first are inaccurate, 2/3 of batches do not extra uncertainty, this improves convergence when initial guess is bad, but increases final uncertainty
                sig_g_est = f_prev * g_est * rel_err_g_est
                p_measure = sig_g_est**2
                
                
                # p_measure = ((batches+10)/M)/np.sqrt(self.model.settings.particles) #Slowly relax uncertainty; OLD version
                
                # Estimate the accuracy of the measurement with a quadratic difference of k and target
                # p_measure = (1 + self.model.settings.particles * (k-target)**2)**2 / np.sqrt(self.model.settings.particles) #OLD version
            else:
                if g_est <= 0.1: g_est = 0.1
                elif g_est >= 2.5: g_est = 2.5
                p_measure = 1e16
            
            if p_n >= 1e16 and p_measure >= 1e16:
                pass
            else:
                p_n = 1/(1/p + 1/p_measure)
                print(f"Propagating uncertainty: p_prev {p}, p_measure {p_measure}, p_next {p_n}")
            z = f_prev * g_est
            
            if bracket is not None:
                if z*initial_value > bracket[1]:
                    z = bracket[1]/initial_value
                elif z*initial_value < bracket[0]:
                    z = bracket[0]/initial_value
            print(f"Changing concentration mult from {x} to {x + p_n/p_measure * (z - x)}, by {p_n/p_measure * (z - x)}, innovation factor: {p_n/p_measure}")
            x = x + p_n/p_measure * (z - x)
            p = copy.copy(p_n)
            f = copy.copy(x)
            g = f/f_prev
            f_prev = copy.copy(f)

            if debug is True:
                k = (P_fiss) / (L_abs + (P_fiss + P_nxn)*L_leak - P_nxn)
                print(f"Batch: {M}")
                print(f"k_absorption: {k}")
                print(f"Batch uncertainty: p: {p_measure}, sig_g: {sig_g_est}")
                print(f"top: {top}, bot: {bot}")
                print(f"Batch estimated correction - 1: {g_est-1}")
                print(f"Batch filtered correction - 1: {g-1}")
                # print(f"Search algorithm internal tally:\n{curr_res}")
                print(f"Correction coefficients [P_fiss, P_nxn, L_leak, L_abs, L_abs_nucs]: {P_fiss, P_nxn, L_leak, L_abs, L_abs_nucs}")
                print(f"Sigmas: [sig1, sig2, sig3]: {sig1, sig2, sig3}")
                print(f"Batch estimated concentration: {f*initial_value} +/- {f*initial_value*(p**(1/2))}")

            # Update densities on C API side
            for mat in openmc.lib.materials:
                if materials is not None:
                    if int(mat) not in mat_ids:
                        continue
                nuclides=[]
                densities=[]
                all_dens = (np.array(openmc.lib.materials[int(mat)].densities)).astype(float)
                all_nuc = np.array(openmc.lib.materials[int(mat)].nuclides)
                
                for nuc in all_nuc:
                    val = float((all_dens[all_nuc==str(nuc)])[0])
                    # If nuclide is zero, do not add to the problem.
                    if val > 0: # 1 atom/barn-cm
                        if str(nuc) in iso:
                            val *= g
                        nuclides.append(nuc)
                        densities.append(val)
                    elif str(nuc) in iso:
                        val *= g
                        nuclides.append(nuc)
                        densities.append(val)
                # Update densities on C API side
                mat_internal = openmc.lib.materials[int(mat)]
                mat_internal.set_densities(nuclides, densities)
        # if M == model.settings.inactive:
        #     openmc.lib.reset()
    openmc.lib.simulation_finalize()
        
    # Finaly update densities on Python API side
    for mat in openmc.lib.materials:
        all_dens = (np.array(openmc.lib.materials[int(mat)].densities)).astype(float)
        all_nuc = np.array(openmc.lib.materials[int(mat)].nuclides)
        i = 0
        for matPY in model.materials:#TODO check if model.materials[i] or model.materials[matPY.id]
            if matPY.id == int(mat):
                for nuc in all_nuc:
                    val = (all_dens[all_nuc==str(nuc)])[0]
                    model.materials[i].remove_nuclide(nuc)
                    model.materials[i].add_nuclide(nuc,val)
            i += 1
    if not perfer_all_xml:
        model.export_to_xml()
    else:
        model.settings.export_to_xml()
        model.materials.export_to_xml()
        model.geometry.export_to_xml()
        if model.plots:
            model.plots.export_to_xml()
        if model.tallies:
            model.tallies.export_to_xml()
    
    return model


def get_ao_mix_materials(materials, fracs, fracs_target=None, percent_type='ao'):#TODO also handle chemical equations, ex. CO2
        """Mix materials together based on atom, weight, or volume fractions

        .. versionadded:: 0.12

        Parameters
        ----------
        materials : Iterable of openmc.Material
            Materials to combine
        fracs : Iterable of float
            Fractions of each material to be combined
        fracs_target : Iterable of str, optional
            Fraction target of each material to be combined, can be nuclide (ex. "B10")
            or element (ex. "B")
        percent_type : {'ao', 'wo', 'vo'}
            Type of percentage, must be one of 'ao', 'wo', or 'vo', to signify atom
            percent (molar percent), weight percent, or volume percent,
            optional. Defaults to 'ao'

        Returns
        -------

        """

        cv.check_type('materials', materials, Iterable, Material)
        # cv.check_type('fracs', fracs, Iterable, Real)
        # cv.check_type('fracs', fracs_target, Iterable, str)
        # if type(fracs_target) ==  str:
        #     cv.check_value('percent type', percent_type, {'ao', 'wo', 'vo'})
        # else:
        #     for f_t in fracs_target:
        cv.check_value('percent type', percent_type, {'ao', 'wo', 'vo'})

        fracs = np.array(fracs)
        
        if len(materials) != len(fracs):
            raise ValueError(f"Number of provided materials: {len(materials)}; does not match the number of provided material fractions: {len(fracs)}")
        
        # if type(percent_type) == str:
        #     percent_type = [percent_type] * len(fracs)
        if fracs_target is None:
            fracs_target = [None] * len(fracs)

        # Calculate appropriate weights which are how many cc's of each
        # material are found in 1cc of the composite material
        # avg_mol_mass = np.asarray([mat.average_molar_mass for mat in materials])
        # mass_dens = np.asarray([mat.get_mass_density() for mat in materials])
        ao_mats = {}
        ao_fr_mats = {}
        wo_fr_mats = {}
        for mat in materials:
            ao_mats[mat] = mat.get_mass_density()
            ao_fr_mats[mat] = get_ao_fraction(mat)
            wo_fr_mats[mat] = get_wo_fraction(mat)
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
        
        # def _average_molar_mass(mat,nucs):
        #     # Using the sum of specified atomic or weight amounts as a basis, sum
        #     # the mass and moles of the material
        #     mass = 0.
        #     moles = 0.
        #     for nuc in mat.nuclides:
        #         if nuc in nucs:
        #             if nuc.percent_type == 'ao':
        #                 mass += nuc.percent * openmc.data.atomic_mass(nuc.name)
        #                 moles += nuc.percent
        #             else:
        #                 moles += nuc.percent / openmc.data.atomic_mass(nuc.name)
        #                 mass += nuc.percent

        #     # Compute and return the molar mass
        #     return mass / moles
        
        norm_wgt = []
        def process_new_frac_target(mat, p_t):
            if target_nucs[mat] != []:
                return 1
            print((mat.name,(mat.get_mass_density(), np.sum(mat.get_mass_density([nuc for nuc in target_nucs[mat]]))), target_nucs[mat]))
            if p_t == 'ao':
                return ((1 / np.sum([ao_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])) if target_nucs[mat] != [] else 1)
                #norm_mat = ((mat.average_molar_mass / _average_molar_mass(mat,target_nucs[mat]))  if target_nucs[mat] != [] else 1)
                #print(norm_mat)
                #return frac * mat.average_molar_mass / mat.get_mass_density() * norm_mat
            elif p_t == 'wo':
                #norm_mat = ((np.sum(list(mat.get_nuclide_atom_densities())) / np.sum([ao_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])) if target_nucs[mat] != [] else 1)
                #return ((mat.average_molar_mass / _average_molar_mass(mat,target_nucs[mat]))  if target_nucs[mat] != [] else 1)
                return mat.get_mass_density() / np.sum(mat.get_mass_density([nuc for nuc in target_nucs[mat]]))
                # print(norm_mat)
                # return frac / mat.get_mass_density() * norm_mat
            elif p_t == 'vo':
                return 1 / np.sum([ao_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])
                # print(norm_mat)
                # return frac * norm_mat
                
        for mat in materials:
            norm_wgt += [process_new_frac_target(mat, percent_type)]
            
        fracs = np.array([frac * wgt if frac is not None else None for (frac,wgt) in zip(fracs,norm_wgt)])
        print("norm_wgts",norm_wgt)
        
        if None in fracs:
            index_none = np.argwhere(fracs == None)#[0]
            print(index_none, fracs)
            print(fracs[index_none])
            fracs[index_none] = 0
            fracs[index_none] = 1 - np.sum(fracs)
        else:
            if not np.abs(np.sum(fracs)-1) < 1e8:
                warnings.warn(f"Resulting weights do not sum to one: {np.sum(wgts)}.\n Please set set one of 'fracs' to None for automatic correction")
        
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

        # # if None not in wgts and (np.abs(np.sum(wgts) - 1) < 1e-6): #TODO make the proper checks
        # #     warnings.warn(f"Resulting weights do not sum to one: {np.sum(wgts)}.\n Please set set one of 'fracs' to None for automatic correction")
        # AVOGADRO = openmc.data.AVOGADRO
        # BARN = 1e-24
        # sum_wo = 0.0
        # sum_ao = 0.0
        # sum_vo = 0.0
        # for (p_t, frac) in zip(percent_type, fracs):
        #     if p_t == 'ao': sum_ao += frac
        #     elif p_t == 'wo': sum_wo += frac
        #     elif p_t == 'vo': sum_vo += frac
            
        # # First mix to get only 1 or 0 of each ao, wo, vo
        # # def mix_all_ao(materials, fracs, fracs_target, percent_type):
        # # Calculate the missing fracton in vo
        # wgts_A = [] #All wo
        # wgts_B = [] #All ao
        # wgts_C = [] #All vo + None
        # for (mat, p_t, frac) in zip(materials, percent_type, fracs):#TODO handle frac=None
        #     if p_t == 'ao':
        #         wgts_B += [frac * mat.average_molar_mass / mat.get_mass_density()]
        #     elif p_t == 'wo':
        #         wgts_B += [frac / mat.get_mass_density()]
        #     elif p_t == 'vo':
        #         wgts_C += [frac]
        # wgts_A /= np.sum(wgts_A)
        # wgts_B /= np.sum(wgts_B)
        # wgts_C /= np.sum(wgts_C)
        # mix_A = defaultdict(float)
        # mix_B = defaultdict(float)
        # mix_C = defaultdict(float)
        # for (mat, p_t, frac) in zip(materials, percent_type, fracs):#TODO handle frac=None
        #     nuc_dict = mat.get_nuclide_atom_densities()
        #     for nuc, val in nuc_dict.items():
        #         if p_t == 'ao':
        #             mix_A[nuc] += val
        #         elif p_t == 'wo':
        #             mix_B[nuc] += val
        #         elif p_t == 'vo':
        #             mix_C[nuc] += val
        
        
        # def _average_molar_mass(nuc_dict):
        #     # Using the sum of specified atomic or weight amounts as a basis, sum
        #     # the mass and moles of the material
        #     mass = 0.
        #     moles = 0.
        #     for nuc in mat.nuclides:
        #         if (nuc,val) in nuc_dict.values():
        #                 mass += val * openmc.data.atomic_mass(nuc)
        #                 moles += val
        #     # Compute and return the molar mass
        #     return mass / moles
        
        # M_A = _average_molar_mass(mix_A)
        # M_B = _average_molar_mass(mix_B)
        # M_C = _average_molar_mass(mix_C)
        
        # def _get_mass_density(nuc_dict) -> float:
        #     """Return mass density of one or all nuclides

        #     Parameters
        #     ----------
        #     nuclides : str, optional
        #         Nuclide for which density is desired. If not specified, the density
        #         for the entire material is given.

        #     Returns
        #     -------
        #     float
        #         Density of the nuclide/material in [g/cm^3]

        #     """
        #     mass_density = 0.0
        #     for nuc, atoms_per_bcm in nuc_dict.items():
        #         density_i = 1e24 * atoms_per_bcm * openmc.data.atomic_mass(nuc) \
        #                     / openmc.data.AVOGADRO
        #         mass_density += density_i
        #     return mass_density
        
        # rho_A = _get_mass_density(mix_A)
        # rho_B = _get_mass_density(mix_B)
        # rho_C = _get_mass_density(mix_C)
        # index_None = np.argwhere(np.array(fracs) == None)
        # a = 1 - sum_ao
        # b = -sum_ao * M_B / M_C
        
            
        # # elif sum_wo == 0:
        # #     fracs[index_None] = 1 - sum_ao
        # # else: # wo to ao conversion
        # #     if "wo" in percent_type:  #TODO handle frac=None
        # #         M_avg_ao = 0
        # #         inv_M_avg_wo = 0
        # #         for (mat, p_t, frac) in zip(materials, percent_type, fracs):
        # #             if p_t == 'ao':
        # #                 M_avg += frac * mat.average_molar_mass
        # #             elif p_t == 'wo':
        # #                 inv_M_avg_wo += frac / mat.average_molar_mass
        # #         inv_M_avg_wo /= sum_wo
        # #         M_avg_ao /= (sum_ao if sum_ao != 0 else 1)
        # #         M_avg = sum_ao * M_avg_ao + (1 - sum_ao) / inv_M_avg_wo
                
        # #         for (mat, p_t, frac, index) in zip(materials, percent_type, fracs, range(len(fracs))): # wo to ao conversion
        # #             if p_t == 'wo':
        # #                 fracs[index] = (frac / mat.average_molar_mass * M_avg if frac is not None else None)
        # #                 percent_type[index] = "ao"
            
        
        # if "vo" in percent_type: # Mixing ao, wo, vo
        #     N_mix_top = 0.0
        #     N_mix_bot = 1.0
        #     for (mat, p_t, frac) in zip(materials, percent_type, fracs):#TODO handle frac=None
        #         if p_t == 'ao':
        #             N_mix_bot -= frac
        #         elif p_t == 'wo':
        #             for (mat2, p_t2, frac2) in zip(materials, percent_type, fracs):
        #                 if p_t2 == 'ao':
        #                     N_mix_bot -= frac * mat2.average_molar_mass / mat.average_molar_mass * frac2
        #                 elif p_t2 == 'vo':
        #                     N_mix_top +=  frac / (1 - sum_wo) * AVOGADRO * BARN / mat.average_molar_mass * frac2 * np.sum(list(mat2.get_mass_density().values()))
        #         elif p_t == 'vo':
        #             N_mix_top += frac * np.sum(list(mat.get_nuclide_atom_densities().values()))
        #         print( mat, N_mix_top, N_mix_bot)
        #     if N_mix_bot == 0:
        #         raise ValueError("Unable to calculate fractions: Internal: N_mix_bot = 0)")
        #     N_mix = N_mix_top / N_mix_bot
        #     print( N_mix, N_mix_top, N_mix_bot)
            
        #     m_mix_top = 0.0
        #     m_mix_bot = 1.0 - sum_wo
        #     for (mat, p_t, frac) in zip(materials, percent_type, fracs):
        #         if p_t == 'ao':
        #             m_mix_top += frac * N_mix * mat.average_molar_mass * AVOGADRO * BARN
        #         if p_t == 'vo':
        #             m_mix_top += mat.get_mass_density()

        #     m_mix = m_mix_top / m_mix_bot
        #     print(m_mix, m_mix_top, m_mix_bot)
            
        #     wgts = []
        #     for (mat, p_t, frac) in zip(materials, percent_type, fracs):
        #         if p_t == 'ao':
        #             wgts += [frac * np.sum(list(mat.get_nuclide_atom_densities().values())) / N_mix]
        #         if p_t == 'wo':
        #             wgts += [frac * mat.get_mass_density() / m_mix]
        #         if p_t == 'vo':
        #             wgts += [frac]#TODO
        # else:

            
                    
        #     if "wo" in percent_type: # wo to ao conversion #TODO handle frac=None
        #         M_avg_ao = 0
        #         inv_M_avg_wo = 0
        #         for (mat, p_t, frac) in zip(materials, percent_type, fracs):
        #             if p_t == 'ao':
        #                 M_avg += frac * mat.average_molar_mass
        #             elif p_t == 'wo':
        #                 inv_M_avg_wo += frac / mat.average_molar_mass
        #         inv_M_avg_wo /= sum_wo
        #         M_avg_ao /= (sum_ao if sum_ao != 0 else 1)
        #         M_avg = sum_ao * M_avg_ao + (1 - sum_ao) / inv_M_avg_wo
                
        #         for (mat, p_t, frac, index) in zip(materials, percent_type, fracs, range(len(fracs))): # wo to ao conversion
        #             if p_t == 'wo':
        #                 fracs[index] = (frac / mat.average_molar_mass * M_avg if frac is not None else None)
        #                 percent_type[index] = "ao"
        #     print("fracs", fracs, np.sum(fracs))
        #     wgts = []
        #     for (mat, frac) in zip(materials, fracs): # ao to vo conversion
        #         wgts += [frac * mat.average_molar_mass / mat.get_mass_density()]
        #     wgts /= np.sum(wgts)
        #     print("weights", wgts, np.sum(wgts))
                        
            
        # # for (mat, p_t, wgt, index) in zip(materials, percent_type, wgts, range(len(wgts))):
        # #     if wgt is None:
        # #         wgts[index] = 1 - np.sum(wgts)

                
            
        # print(wgts)
        # print(materials,percent_type,fracs)
        # wgts = mix_ao_wo_vo(materials,percent_type,fracs)
        # Add nuclide densities weighted by appropriate fractions
        nuclides_per_bmc = defaultdict(float)
        
        for (mat, wgt) in zip(materials, wgts):
            for nuc, atoms_per_bcm in mat.get_nuclide_atom_densities().items():
                nuc_per_bmc = wgt * atoms_per_bcm
                nuclides_per_bmc[nuc] += nuc_per_bmc
        nuclide_ao_fr_per_submat = defaultdict(float)
        for nuc, _ in nuclides_per_bmc.items():
            nuclide_ao_fr_per_submat[nuc] = [0] * len(wgts)
        for (mat, wgt, index) in zip(materials, wgts, range(len(wgts))):
            for nuc, atoms_per_bcm in mat.get_nuclide_atom_densities().items():
                nuc_per_bmc = wgt * atoms_per_bcm
                nuclide_ao_fr_per_submat[nuc][index] = nuc_per_bmc / nuclides_per_bmc[nuc]
        return nuclides_per_bmc, nuclide_ao_fr_per_submat
 


def get_ao_fraction(material):
    nuc_dict = material.get_nuclide_atom_densities()
    mat_ao = np.sum(list(nuc_dict.values()))
    new_dict2 = {}
    for key, value in nuc_dict.items():
        # print(type(value))
        # print(key, value)
        new_dict2[key] = value/mat_ao
    return new_dict2

def get_wo_fraction(material):
    nuc_dict= {}
    for nuc in material.nuclides:
        nuc_dict[nuc] = material.get_mass_density(nuc)
    mat_dens = material.get_mass_density()
    new_dict2 = {}
    for key, value in nuc_dict.items():
        # print(type(value))
        # print(key, value)
        new_dict2[key] = value/mat_dens
    return new_dict2


def mix_ao_wo_vo(materials, fraction_types, fraction_values, V_tot=1.0):
    """
    Exact mixture calculation with one filler material.
    
    Returns:
        - vo_fractions: volume fractions
        - wo_fractions: mass fractions
        - ao_fractions: mole fractions
    """

    N = len(materials)
    M = np.array([m.average_molar_mass for m in materials], dtype=float)
    rho = np.array([m.get_mass_density() for m in materials], dtype=float)

    volumes = np.zeros(N)
    masses  = np.zeros(N)
    moles   = np.zeros(N)

    unknown_idx = []

    # Step 1: assign volumes for explicit volume fractions
    for i, (ftype, fval) in enumerate(zip(fraction_types, fraction_values)):
        if ftype == "vo":
            volumes[i] = fval * V_tot
            masses[i] = rho[i] * volumes[i]
            moles[i]  = masses[i] / M[i]
        elif ftype in ("wo", "ao"):
            unknown_idx.append(i)
        elif ftype is None:
            unknown_idx = i
            break
        else:
            raise ValueError(f"Invalid fraction type {ftype}")

    # Step 2: handle the simple case of **one unknown** (filler or mass/mole fraction)
    vec = []
    def calc_values(guess_rho, guess_M):
        s_ao = 0
        s_wo = 0
        s_vo = 0
        for i, (IM, Irho, ftype, fval) in enumerate(zip(M, rho, fraction_types, fraction_values)):
            if ftype == "ao":
                s_ao += fval
                s_wo += fval * Irho / IM / guess_M / guess_rho 
                s_vo += fval * IM / Irho * guess_rho / guess_M
            elif ftype == "wo":
                s_ao += fval * guess_M / IM
                s_wo += fval 
                s_vo += fval * guess_rho / Irho
            elif ftype == "vo":
                if fval == None: continue
                s_ao += fval * Irho / IM * guess_M / guess_rho
                s_wo += fval * Irho / guess_rho
                s_vo += fval
            
        
    
    for i, (ftype, fval) in enumerate(zip(fraction_types, fraction_values)):
        volumes[i] = fval * V_tot
        masses[i] = rho[i] * volumes[i]
        moles[i]  = masses[i] / M[i]
    if len(unknown_idx) == 1:
        idx = unknown_idx[0]
        # Remaining volume goes to the filler
        V_remain = V_tot - volumes.sum()
        volumes[idx] = V_remain
        masses[idx] = volumes[idx] * rho[idx]
        moles[idx]  = masses[idx] / M[idx]

        # Compute exact fractions
        v_fractions = volumes / volumes.sum()
        w_fractions = masses / masses.sum()
        x_fractions = moles / moles.sum()
        return v_fractions#, w_fractions, x_fractions

    # Step 3: multiple unknowns (mass/mole fractions) -> solve exactly
    # For exact results, we solve **linear equations symbolically** if possible
    # Here, for simplicity, raise an error if multiple unknowns are present
    raise ValueError(
        "Exact solution only implemented for one filler or single unknown. "
        "Use simpler constraints or one filler."
    )
