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
                        materials=None, initial_value=None, target=1.,
                        mat_builder=None, perfer_all_xml=True, debug=False):
    # TODO allow for change in number of neutrons and then return settings to previous values
    # TODO allow mixing for multiple materials
    # TODO CDI for neutron producing nuclides, Be, U, Pu,... Probably a new function
    # TODO overwrite iso option in mix_materials
    """
    Runs a simulation where 'iso' nuclide values converge in such a way to obtain the desired k_eff.
    This option assumes that flagged 'iso' nuclides are absorbers and do not produce neutrons by either fission or (n, xn) reactions.
    For that use function: NOT YET DEVELOPED.
    Operator.model materials are updated in the process
    Optional initial value is the value given in your material building process, must be strictly bigger than 0.
    All 'iso' nuclides are multiplied by the same scaling factor.
    Higher (>10 000) particle numbers in 'openmc.settings' are recommended for more accurate simulation.
    Atleast 30 batches are required for adequate convergence, recommended >50 or roughly
    sqrt(number of particles per batch) to achieve best results.

    .. versionadded:: 0.15.3
    
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
        Used in first call, used for intermediate critical concentration message output.
        Required when mat_builder option is used
        Defaults to 1.0.
    target: float, optional
        Target k_eff, defaults to 1.0
    mat_builder: function optional
        Callable builder function, that returns a list of materials.
        
        
        dictionary of isotope concentrations
        ('nuclide':value in atoms/b-cm) for each flagged material.
        It is called in each step of CDI.
        When used 'initial_value' parameter is required.
    debug: Bool, optional
        Wether to print out batch number, tally results of each batch,
        batch k_absorption and current concentration.
        Defaults to False.

    Returns
    -------
    openmc.model.model, with updated critical density concentrations of flagged nuclides in flagged materials

    """
    if mat_builder is not None:
        materials = mat_builder(initial_value)
        iso = []
        for mat in materials:
            for nuc in mat.nuclides:
                if nuc not in iso:
                    iso += [nuc]
        
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
            if debug is True: print(f"Batch: {M}")
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
                if debug is True: print(f"Propagating uncertainty: p_prev {p}, p_measure {p_measure}, p_next {p_n}")
            z = f_prev * g_est
            
            if bracket is not None:
                if z*initial_value > bracket[1]:
                    z = bracket[1]/initial_value
                elif z*initial_value < bracket[0]:
                    z = bracket[0]/initial_value
            if debug is True: print(f"Changing concentration mult from {x} to {x + p_n/p_measure * (z - x)}, by {p_n/p_measure * (z - x)}, innovation factor: {p_n/p_measure}")
            x = x + p_n/p_measure * (z - x)
            p = copy.copy(p_n)
            f = copy.copy(x)
            g = f/f_prev
            f_prev = copy.copy(f)

            if debug is True:
                k = (P_fiss) / (L_abs + (P_fiss + P_nxn)*L_leak - P_nxn)
                # print(f"Batch: {M}")
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
            if 0:#mat_builder is not None:
                mat_builder(f*initial_value)
            else:
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


def get_ao_mix_materials(materials, fracs, fracs_target=None, percent_type='ao'):
    """Mix materials together based on atom, weight, or volume fractions

    .. versionadded:: 0.15.3

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

    Returns
    -------

    """

    cv.check_type('materials', materials, Iterable, Material)
    # cv.check_type('fracs', fracs, Iterable, Real)
    cv.check_value('percent type', percent_type, {'ao', 'wo', 'vo'})

    fracs = np.array(fracs)
    
    if len(materials) != len(fracs):
        raise ValueError(f"Number of provided materials: {len(materials)}; does not match the number of provided material fractions: {len(fracs)}")
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
            
    for mat in materials:
        norm_wgt += [process_new_frac_target(mat, percent_type)]
        
    fracs = np.array([frac * wgt if frac is not None else None for (frac,wgt) in zip(fracs,norm_wgt)])
    print("norm_wgts",norm_wgt)
    
    if None in fracs:
        index_none = np.argwhere(fracs == None)
        fracs[index_none] = 0
        fracs[index_none] = 1 - np.sum(fracs)
    else:
        if not np.abs(np.sum(fracs)-1) < 1e6:
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
    nuc_dict = {}
    for nuc in material.nuclides:
        nuc_dict[nuc.name] = material.get_mass_density(nuc.name)
    mat_dens = material.get_mass_density()
    new_dict2 = {}
    for key, value in nuc_dict.items():
        # print(type(value))
        # print(key, value)
        new_dict2[key] = value/mat_dens
    return new_dict2

def update_material(mat, nuc_dict):
    nuc_remove = []
    for nuc in mat.nuclides:
        nuc_remove += [nuc.name]
    for nuc in nuc_remove:
        mat.remove_nuclide(nuc)
    for nuc, val in nuc_dict.items():
        mat.add_nuclide(nuc, val)
    nuc_dict2 = mat.get_nuclide_atom_densities()
    mat_ao = np.sum(list(nuc_dict2.values()))
    mat.set_density('atom/b-cm', mat_ao)