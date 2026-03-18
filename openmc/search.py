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


def critical_density_iteration(model, iso=None, batches=None, bracket=None, 
                        materials=None, initial_value=1.0, target=1.,
                        mat_builder=None, prefer_model_xml=False,
                        max_step_change=4, debug=False):
    """
    A (without 'mat_builder') - Legacy option: Runs a simulation where 'iso' nuclide values converge in such a way to obtain the desired k_eff.
        Material density remains untouched, therefore additional 'iso' nuclides 'compete' with all other nuclides.
        If materials are not provided all known instances of 'iso' nuclides are multiplied by the same scaling factor.
        Works reasonably well for nuclides with small relative densities (<1/1000 of the total material density).
    B (with 'mat_builder'): Runs a simulation where materials are varied using 'mat_builder' function to obtain the desired k_eff.
        This option is more flexible but relies on the user provided 'mat_builder' function.
    
    Model materials are updated in the process.
    Optional initial value is the value given in your material building process, must be strictly bigger than 0.
   
    Higher (>10 000) particle numbers per batch in 'openmc.settings' are recommended.
    50 or higher batches are recommended for proper convergance.
    CDI batches are added to inactive batches of the simulation, if the number of inactive batches does not exceed, CDI batches + 10.
    These 10 or more inactive batches are used for neutron source convergance before active CDI starts.

    .. versionadded:: 0.15.4
    
    Parameters
    ----------
    model: openmc.model, required
    iso: array of str, required or mat_builder is provided
        Nuclide name, ex. ["B10", "B11"]
        'mat_builder' overwrites isotopes provided by this option.
    batches: int, optional
        Number of inactive batches added to the begining of simulation where
        'iso' concentration converges.
        Defaults to 50 extra inactive cycles.
    bracket: array of 2 floats > 0, optional
        Lower and upper bounds for concentrations.
        Inteded to be used in tandem with initial_value.
        Otherwise just a multiple of initial concentration at step 0.
    materials: materials in which nuclide concentrations are changed, optional
        'mat_builder' overwrites materials provided by this option.
        Defaults to all materials.
    initial_value: float > 0, optional
        Used for additional critical concentration message output.
        Used in first call of 'mat_builder' function
        Required when mat_builder option is used
        Defaults to 1.0.
    target: float, optional
        Target k_eff, defaults to 1.0
    mat_builder: function, optional
        Callable builder function, that returns a dictionary with  'materials' and 'nuc_fractions' keys
        in the mixed material. Meant to be used with 'openmc.search.get_ao_mix_materials'
        and 'openmc.material.update_material' functions.
        
        dictionary of isotope concentrations
        ('nuclide':value in atoms/b-cm) for each flagged material.
        It is called in each step of CDI.
        When used use of 'initial_value' parameter is recommended.
    prefer_model_xml: Bool, optional
        Whether to prefer model XML files.
        Defaults to False.
    max_step_change: float > 1.0, optional
        Maximum allowed change in concentration per iteration step.
        Defaults to 4.0.
    debug: Bool, optional
        Wether to print out batch number, tally results of each batch,
        batch k_absorption and current concentration.
        Defaults to False.

    Returns
    -------
    openmc.model.model
        Updated model with converged concentrations.
    [float, float]
        converged_value, one-sigma uncertainty

    """
    # Check input arguments
    if mat_builder is not None:
        mat_builder_res = mat_builder(initial_value)
        keys = mat_builder_res.keys() if type(mat_builder_res) == dict else None
        if keys:
            if "materials" in keys:
                materials = [mat for mat in mat_builder_res["materials"]]
            if "nuc_fractions" in keys:
                nuc_fractions = mat_builder_res["nuc_fractions"]
        else:
            materials = mat_builder(initial_value)
            nuc_fractions = np.array([[1 for i in iso] for m in materials])
        if iso is None:
            iso = []
            for mat in materials:
                for nuc in mat.nuclides:
                    if nuc.name not in iso:
                        iso += [nuc.name]
    else:
        if materials is not None:
            nuc_fractions = np.array([[1 for i in iso] for m in materials])
        else:
            nuc_fractions = np.array([[1 for i in iso] for m in model.materials])
    if iso is None:
        raise ValueError("'iso' argument is empty")
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
    if materials is not None:
        mat_ids=[]
        for mat in materials:
            mat_ids += [mat.id]
        
    #Create tallies if not already created
    if model.settings.inactive is None:
        model.settings.inactive = 0
    if model.settings.inactive < batches + 10:
        model.settings.inactive += batches + 10
        model.settings.batches += batches + 10
    tally_ids = [tally.id for tally in model.tallies]
    if 8888 not in tally_ids or 8889 not in tally_ids:
        tallyTest = Tally(tally_id=8888, name="CDI_tally1")
        tallyTest.scores = ["nu-fission", "absorption", "nu-scatter", "scatter"]
        model.tallies += [tallyTest]

        tallyTest2 = Tally(tally_id=8889, name="CDI_tally2")
        if iso is None:
            raise ValueError("'iso' is empty")
        tallyTest2.nuclides = iso
        tallyTest2.scores = ["nu-fission", "absorption", "nu-scatter", "scatter"]
        if materials is not None:
            tallyTest2.filters = [MaterialFilter(materials,filter_id=8888)]
        model.tallies += [tallyTest2]
    
    # Export modified model
    if prefer_model_xml:
        model.export_to_model_xml()
    else:
        model.export_to_xml()

    # Initialize variables for iteration
    guesses = []
    guess_unc = []
    guess_ks = []
    f = 1
    g = 1
    f_prev = 1
    prev_res = []
    prev_leak = 0
    skip_steps = False
    starting_batch = model.settings.inactive - batches

    # Initialize OpenMC library
    comm.barrier()
    if not openmc.lib.is_initialized:
        if debug is True: print("Initializing OpenMC library for CDI...")
        openmc.lib.init(intracomm=comm)
    openmc.lib.reset()
    openmc.lib.simulation_init()
    
    # Set required tallies to active
    for t_id, _tally in openmc.lib.tallies.items():
        if t_id == 8888 or t_id == 8889:
            _tally.active = True
    
    # Run simulation
    for _ in openmc.lib.iter_batches():
        M = openmc.lib.current_batch()
        if M > model.settings.inactive: continue
        if skip_steps: continue
        
        # Get tallies
        # Tally results are added (summed) in each batch, batch result is the difference
        _tallies = copy.copy(openmc.lib.tallies)
        global_tallies = copy.copy(openmc.lib.global_tallies())
        curr_res = []
        if M == 1:
            for _tally in _tallies.values():
                if _tally.id == 8888 or _tally.id == 8889:
                    prev_res += [_tally.results - _tally.results]
        else:
            i=0
            for _tally in _tallies.values():
                if _tally.id == 8888 or _tally.id == 8889:
                    curr_res += [_tally.results - prev_res[i]]
                    prev_res[i] = copy.copy(_tally.results)
                    i+=1
        # Leakage is a running average
        leak = global_tallies[3][0]*M - prev_leak
        prev_leak = global_tallies[3][0]*M
        
        # Only change concentrations during the inactive CDI batches
        if M < starting_batch + batches:
            # Skip initial steps for flux convergence
            if M < starting_batch: continue
            if debug is True: print(f"\n Batch: {M}")
            
            ### P = production of neutrons, L = loss of neutrons
            # Neutrons produced by fission (prompt and delayed)
            P_fiss = curr_res[0][0][0][1]
            P_fiss = P_fiss if P_fiss > 0 else 0
            # Additional neutrons produced by (n,xn) reactions
            P_nxn = curr_res[0][0][2][1] - curr_res[0][0][3][1]
            P_nxn = P_nxn if P_nxn > 0 else 0
            # Neutron leakage fraction
            L_leak = leak if leak > 0 else 0
            # Total neutron absorption             
            L_abs = curr_res[0][0][1][1]                                
            L_abs = L_abs if L_abs > 0 else 0
            # WARNING: L_abs - P_nxn + L_leak === 1; by OpenMC def
            # >0, for when floating point errors cause negative values

            # Same as above but summed for all flagged nuclides, weighted by weights if provided by 'mat_builder'
            P_fiss_nucs = 0
            P_nxn_nucs = 0
            L_abs_nucs = 0
            for index, mat in enumerate(materials):
                if debug: print(f"Tally partial fractions for mat with id={mat.id}:",np.array(nuc_fractions[index]))
                Res_nucs_mat = np.array(curr_res[1][index])
                P_fiss_nucs += np.sum((Res_nucs_mat[0::4,1]) * np.array(nuc_fractions[index]))
                P_nxn_nucs += np.sum((Res_nucs_mat[2::4,1] - Res_nucs_mat[3::4,1]) * np.array(nuc_fractions[index]))
                L_abs_nucs += np.sum((Res_nucs_mat[1::4,1]) * np.array(nuc_fractions[index]))
            P_fiss_nucs = P_fiss_nucs if P_fiss_nucs > 0 else 0
            P_nxn_nucs = P_nxn_nucs if P_nxn_nucs > 0 else 0
            L_abs_nucs = L_abs_nucs if L_abs_nucs > 0 else 0
            if L_abs_nucs == 0:
                print(f"CDI: No nuclide absorption tallied, skipping from step {M} onwards")
                skip_steps = True
                continue
            
            # Predict concentration change
            bot = L_abs_nucs - P_fiss_nucs/target - P_nxn_nucs
            top = P_fiss - 1 + bot 
            if bot == 0:
                print(f"CDI: Combined effect of absorption, fission and (n,xn) reaction of flagged is 0, skipping step {M}")
                continue
            g_est = top / bot
            # Optimal following (Kalman filter for scalar value)
            if M == starting_batch: #Kalman filter initialization, at step 10
                x = 1
                p = 1e16
                p_n = 1e16
                p_measure = 1e16
            # Limit the step change and calculate uncertainty based on MC statistics
            if (g_est >= 1/(max_step_change-0.5) and g_est <= max_step_change):
                rel_err_MC = 1/np.sqrt(model.settings.particles * (model.settings.generations_per_batch if model.settings.generations_per_batch is not None else 1))
                prod = P_fiss + P_nxn
                loss = L_abs + L_leak
                ### g_est = (P_fiss/target - 1)/nucs + 1
                ### Covariances between fission (also (n,xn) reactions) and absorption neglected
                ### Would need nu_bar for fiss/abs nucs and rho((n,xn):abs) (=approx 2, as higher reactions are much less probable)
                ### Rho is covariance coefficient
                ### If nuc is the only fissile nuc: rho(fiss_nucs, fiss)=1, ie. rho=np.sqrt(fiss_nucs/fiss)
                ### Following the example rho(abs_nucs, fiss) = -np.sqrt(abs_nucs/(fiss/nu_bar)), minus because when a fission reaction replaces absorption
                ### Following the example rho(nxn_nucs, fiss) = -np.sqrt((nxn_nucs/2)/(fiss/nu_bar)), minus!
                sig_nucs = rel_err_MC * np.sqrt(prod*P_fiss_nucs/target
                                               + prod*P_nxn_nucs
                                               + loss*L_abs_nucs
                                               )
                sig_fiss = rel_err_MC * np.sqrt(prod*P_fiss)
                sig_res = (P_fiss/target-1)/bot *np.sqrt((sig_fiss/(P_fiss/target-1))**2 if (P_fiss/target-1) != 0 else rel_err_MC #Catch div by 0
                                                         + (sig_nucs/bot)**2
                                                         )
                
                sig_g_est = np.abs(sig_res * (1 + (100*np.exp(-(M - starting_batch) / batches * 3 * np.log(100)) - 1
                                                   if (M - starting_batch) < (batches / 3) else 0)))
                ### Slowly relax uncertainty, as first batches are inaccurate, 2/3 of batches do not recieve extra uncertainty,
                ### this improves convergence when initial guess is bad, but increases final uncertainty
                rel_err_g_est = sig_g_est / g_est
                
                sig_g_est = f_prev * sig_g_est
                p_measure = sig_g_est**2
            else:
                if g_est <= 1/(max_step_change-0.5):
                    if debug: print(f"estimated change out of bounds, g_est: {g_est} scaled to {1/(max_step_change-0.5)}")
                    g_est = 1/(max_step_change-0.5)
                    
                elif g_est >= max_step_change:
                    if debug: print(f"estimated change out of bounds, g_est: {g_est} scaled to {max_step_change}")
                    g_est = max_step_change
                p_measure = 1e16
                sig_g_est = p_measure**(1/2)
            
            # Store values for analysis and CDI coefficient estimation at the end of the simulation
            guesses += [f]
            guess_unc += [p_n**(1/2)]
            guess_ks += [(P_fiss-P_nxn)/(L_abs+L_leak-P_nxn)]
            
            # Continue Kalman filter
            if p_n >= 1e16 and p_measure >= 1e16:
                pass
            else:
                p_n = 1/(1/p + 1/p_measure)
                if debug is True: print(f"Propagating uncertainty: p_prev {p}, p_measure {p_measure}, p_next {p_n}")
            z = f_prev * g_est
            
            # Handle bracket
            if bracket is not None:
                if z*initial_value > bracket[1]:
                    z = bracket[1]/initial_value
                elif z*initial_value < bracket[0]:
                    z = bracket[0]/initial_value
            if debug is True: print(f"Changing concentration mult from {x} to {x + p_n/p_measure * (z - x)}, by {p_n/p_measure * (z - x)}, innovation factor: {p_n/p_measure}")
            
            # Finally update the concentration multiplier and uncertainty for the next step
            x = x + p_n/p_measure * (z - x)
            p = copy.copy(p_n)
            f = copy.copy(x)
            g = f/f_prev
            f_prev = copy.copy(f)
            
            # Print debug information
            if debug is True:
                print(f"Batch uncertainty: p: {p_measure}, sig_g: {sig_g_est}")
                print(f"Batch values top: {top}, bot: {bot}")
                print(f"Batch estimated correction - 1: {g_est-1}")
                print(f"Batch filtered correction - 1: {g-1}")
                # if g_est > 1/(max_step_change-0.5) and g_est < max_step_change:
                print(f"Correction coefficients [P_fiss, P_nxn, L_leak, L_abs]: {P_fiss, P_nxn, L_leak, L_abs}")
                print(f"Correction coefficients nucs [L_abs_nucs, P_fiss_nuc, P_nxn_nucs]: {L_abs_nucs, P_fiss_nucs, P_nxn_nucs}")
                # print(f"OpenMC def diff: L_abs+L_leak-P_nxn-1: {(L_abs+L_leak-P_nxn-1):.03e}")
                # print("sig_fiss", sig_fiss, "sig_nucs", sig_nucs)
                print(f"Relative error g_est: {rel_err_g_est}")
                print(f"Batch estimated concentration: {f*initial_value} +/- {initial_value*(p**(1/2))}")

            # Rebuild the material with the given function at provided concentration
            if mat_builder is not None:
                mat_builder_res = mat_builder(f*initial_value)
                keys = mat_builder_res.keys() if type(mat_builder_res) == dict else None
                if keys:
                    if "materials" in keys:
                        materials = [mat for mat in mat_builder_res["materials"]]
                    if "nuc_fractions" in keys:
                        nuc_fractions = mat_builder_res["nuc_fractions"]
            
            # Update densities on C API side
            for rank in range(comm.size):
                C_API_mats = comm.bcast(openmc.lib.materials, root=rank)
                for mat in C_API_mats:
                    if materials is not None:
                        if int(mat) not in mat_ids:
                            continue
                    nuclides=[]
                    densities=[]
                    all_nuc = np.array(C_API_mats[int(mat)].nuclides)
                    mat_internal = C_API_mats[int(mat)]
                    
                    if mat_builder is None: # Change all materials with the flagged nuclides, as in option A
                        all_dens = (np.array(C_API_mats[int(mat)].densities)).astype(float)
                        for nuc in all_nuc:
                            val = float((all_dens[all_nuc==str(nuc)])[0])
                            # If nuclide is zero, do not add to the problem.
                            if val > 0:
                                if str(nuc) in iso:
                                    val *= g
                                nuclides.append(nuc)
                                densities.append(val)
                            elif str(nuc) in iso:
                                val *= g
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
                    mat_internal.set_density(np.sum(densities))
                    mat_internal.set_densities(nuclides, densities)
                    
    openmc.lib.simulation_finalize()
    # Finaly update densities on Python API side
    for mat in openmc.lib.materials:
        all_dens = (np.array(openmc.lib.materials[int(mat)].densities)).astype(float)
        all_nuc = np.array(openmc.lib.materials[int(mat)].nuclides)
        for i, matPY in enumerate(model.materials):
            if matPY.id == int(mat):
                for nuc in all_nuc:
                    val = (all_dens[all_nuc==str(nuc)])[0]
                    model.materials[i].remove_nuclide(nuc)
                    model.materials[i].add_nuclide(nuc,val)
    
    # Output results and estimated CDI coefficient
    print(f"CDI: Solve converged to concentration: {f*initial_value} +/- {initial_value*(p**(1/2))}")
    cdi_Cs = np.array(guesses)[1:]*initial_value
    cdi_ks = np.array(guess_ks)[1:]*1e5
    def linF(x,n,k):
        return n+k*x
    cdi_res, cdi_res_cov = sopt.curve_fit(linF,cdi_Cs,cdi_ks,sigma=np.array(guess_unc)[1:])#,absolute_sigma=True)
    cdi_res_sig = np.diag(cdi_res_cov)
    # poly_res=np.polyfit(cdi_Cs,cdi_ks, 1,w=cdi_wgts) #Wgts are UN-squared
    print(f"CDI: Estimated concentration reactivity coefficient: {cdi_res[1]:.05e} +/- {cdi_res_sig[1]:.05e} pcm/unit of concentration")

    # Update the model with the final concentrations
    if prefer_model_xml:
        model.export_to_model_xml()
    else:
        model.export_to_xml()
    
    return model, [f*initial_value, initial_value*(p**(1/2))]


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
    ao_mats = {}
    ao_fr_mats = {}
    wo_fr_mats = {}
    for mat in materials:
        ao_mats[mat] = mat.get_mass_density()
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
        if p_t == 'ao':
            return 1 / np.sum([ao_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])
        elif p_t == 'wo':
            return 1 / np.sum([wo_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])
        elif p_t == 'vo':
            return 1 / np.sum([ao_fr_mats[mat].get(nuc,0) for nuc in target_nucs[mat]])
    
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
 

class CDI:
    """
    Class to perform critical density iteration (CDI) inside the OpenMC depletion. CDI is a method used to find the critical concentration of a nuclide in a material such that the effective multiplication factor (k_eff) of a nuclear system is equal to a target value (usually 1). The class provides a method to perform CDI by iteratively adjusting the concentration of the nuclide and running OpenMC simulations until convergence is achieved.
    
    CDI is a wrapper around the 'critical_density_iteration' function, which performs the actual iteration process. The class allows for more convenient usage of CDI within a depletion simulation, as it can be called as a function and maintains the state of the last concentration value for subsequent calls.
    
    Returns
    -------
    openmc.model.model
        Updated model with converged concentrations.

    .. versionadded:: 0.15.4
    """
    def __init__(self, model, iso=None, batches=None, bracket=None, 
                        materials=None, initial_value=1.0, target=1.,
                        mat_builder=None, prefer_model_xml=False,
                        max_step_change=4, debug=False, force_initial_value=False):

            # Check input arguments and prepare the model for CDI
            if mat_builder is not None:
                mat_builder_res = mat_builder(initial_value)
                if type(mat_builder_res) == dict:
                    keys = mat_builder_res.keys()
                else:
                    raise ValueError("mat_builder function must return a dictionary with 'materials' key")
                if "materials" in keys:
                    materials = [mat for mat in mat_builder_res["materials"]]
                else:
                    raise ValueError("'materials' not found in mat_builder return dictionary")
                if iso is None:
                    iso = []
                    for mat in materials:
                        for nuc in mat.nuclides:
                            if nuc.name not in iso:
                                iso += [nuc.name]
            if iso is None:
                raise ValueError("'iso' argument is empty")
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
            if materials is not None:
                mat_ids=[]
                for mat in materials:
                    mat_ids += [mat.id]
                
            #Create tallies if not already created
            if model.settings.inactive is None:
                model.settings.inactive = 0
            if model.settings.inactive < batches + 10:
                model.settings.inactive += batches + 10
                model.settings.batches += batches + 10
            tally_ids = [tally.id for tally in model.tallies]
            if 8888 not in tally_ids or 8889 not in tally_ids:
                tallyTest = Tally(tally_id=8888, name="CDI_tally1")
                tallyTest.scores = ["nu-fission", "absorption", "nu-scatter", "scatter"]
                model.tallies += [tallyTest]

                tallyTest2 = Tally(tally_id=8889, name="CDI_tally2")
                if iso is None:
                    raise ValueError("'iso' is empty")
                tallyTest2.nuclides = iso
                tallyTest2.scores = ["nu-fission", "absorption", "nu-scatter", "scatter"]
                if materials is not None:
                    tallyTest2.filters = [MaterialFilter(materials,filter_id=8888)]
                model.tallies += [tallyTest2]
            
            # Export modified model
            if prefer_model_xml:
                model.export_to_model_xml()
            else:
                model.export_to_xml()
                
            self.model = model
            self.iso = iso
            self.batches = batches
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
            
    def _get_model(self):
        return self.model
    
    def __call__(self):
        self.model, self.last_result = critical_density_iteration(model=self.model, iso=self.iso, batches=self.batches, bracket=self.bracket, 
                        materials=self.materials, target=self.target,
                        initial_value=(self.initial_value if (self.last_result is None or self.force_initial_value) else self.last_result[0]), 
                        mat_builder=self.mat_builder, prefer_model_xml=self.prefer_model_xml,
                        max_step_change=self.max_step_change, debug=self.debug)
        return self.model