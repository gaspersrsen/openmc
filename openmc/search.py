from collections.abc import Callable
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
                        materials=None, initial_value=None, target=1., perfer_all_xml=False, debug=False): # TODO allow for change in number of neutrons and then return settings to previous values
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
    starting_batch = model.settings.inactive - batches - 1 #Otherwise Obi-Wan error
    # Initialize OpenMC library
    comm.barrier()
    if not openmc.lib.is_initialized:
        openmc.lib.init(intracomm=comm)
    openmc.lib.reset()
    openmc.lib.simulation_init()
    # Run simulation
    for _ in openmc.lib.iter_batches():
        M = openmc.lib.current_batch()
        if M < starting_batch: continue
        # Only change concentrations during the additional batches
        if M < batches+10 and not skip_steps:
            #k = openmc.lib.keff()[0]
            talliez = copy.copy(openmc.lib.tallies)
            curr_res = []
            if M == 10:
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
            if M == 10: #Start the iteration at step 10, handled before, this is only K.f initialization
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
                rel_err_g_est = (rel_err_top + rel_err_bot)  * (1 + (100*np.exp(-(M - 10)**2 / (batches / 6)) if (M - 10) < (batches / 3) else 0)) #Slowly relax uncertainty, as first are inaccurate, 2/3 of batches do not extra uncertainty, this improves convergence when initial guess is bad, but increases final uncertainty
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
        if M == batches:
            openmc.lib.reset()
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
