import sys
import os
import time
import pickle
from pprint import pformat
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

from desc import set_device
set_device("gpu")

# import jax
# jax.config.update("jax_compilation_cache_dir", "../jax-caches")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np
import matplotlib.pyplot as plt
# Desc imports
import desc
from desc.backend import jnp, jit
from desc.magnetic_fields import (
    FourierCurrentPotentialField,
    SplineMagneticField,
    field_line_integrate,
)
from desc.grid import LinearGrid
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.objectives import (
    QuadcoilFreeBoundaryError,
    QuadcoilProxy,
    QuadraticFlux,
    BoundaryError,
    RotationalTransform,
    VacuumBoundaryError,
    SurfaceCurrentRegularization,
    QuasisymmetryTwoTerm,
    Volume,
    FixBoundaryR,
    FixBoundaryZ,
    FixIota,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile
from desc.vmec import VMECIO
from quadcoil.quantity import f_B, Phi_with_net_current, Phi

# Creating toroidal initial state
vacuum = True
surf = FourierRZToroidalSurface(
    R_lmn=[8.0, 1.0],
    modes_R=[[0, 0], [2, 0]],
    Z_lmn=[-2.0],
    modes_Z=[[-1, 0]],
    NFP=2,
)
eq_init = Equilibrium(
    M=16, N=16, # Psi=1.0, 
    surface=surf, 
    current=PowerSeriesProfile(),
    # pressure=pres, iota=iota
)
eq_init.solve()
eq_init, info = eq_init.solve(verbose=3, copy=False)

# ----- Targets -----

# Field-on-coil at the dipole layer
B2_self_target = 15.**2 
plasma_coil_distance = 1.5 
# Bound for 2-term QS error
qs_bound = 1e-4 # Helios boozer error is between 2.1e-3 and 5.8e-3
# Bounds for iota
iota_l, iota_u = 0.1, 0.4 # helios is 0.15
vol_l, vol_u = 450, 550 # Helios is 493

# ----- Quadcoil Resolution -----

mpol = 8  # Num. poloidal modes for the current potential
ntor = 8  # Num. toroidal modes for the current potential
# Resolution for sampling objectives
quadpoints_phi = jnp.linspace(0, 1 / eq_init.NFP, 32, endpoint=False)
quadpoints_theta = jnp.linspace(0, 1, 32, endpoint=False)

# ----- Creating quadcoil inputs -----

# Creating quadcoil input kwargs. These can be fed into 
# both quadcoil and its DESC interfaces. 
quadcoil_kwargs_basic = {
    "mpol": mpol,
    "ntor": ntor,
    # Resolutions for evaluating winding-surface
    # pointwise objectives and plotting.
    # Note that this does not control the resolution
    # for evaluating winding surface integrals.
    # The integral resolution is controlled using
    # winding_quadpoints_phi and winding_quadpoints_theta.
    # Here, we use the default value of 33x(32xnfp), which
    # is often good enough.
    "quadpoints_phi": quadpoints_phi,
    "quadpoints_theta": quadpoints_theta,
    "plasma_coil_distance": plasma_coil_distance,
}


quadcoil_kwargs_nescoil = quadcoil_kwargs_basic | {
    # The NESCOIL problem only contains the squared
    # flux objective. In QUADCOIL, this quantity is called
    # f_B.
    "objective_name": "f_B",
    # The NESCOIL problem is simple enough to need no normalization
    # constants. In the next example we will discuss how to choose
    # this constant.
    "objective_unit": None,
}

# Solve NESCOIL to estimate some scaling factors
nescoil_objective = QuadcoilProxy(
    eq=eq_init,
    quadcoil_kwargs=quadcoil_kwargs_nescoil,
    vacuum=vacuum,
    # If you have additional filament/planar coils, put the CoilSet here.
    # field=[],
)
nescoil_objective.build()
f_B_nescoil = f_B(qp_nescoil, dofs_nescoil)
# Solving the NESCOIL problem
out_dict_nescoil, qp_nescoil, dofs_nescoil, status_nescoil = \
    nescoil_objective.solve_quadcoil(*nescoil_objective.xs(eq_init))


# Defining problem
quadcoil_kwargs = quadcoil_kwargs_basic | {
    "objective_name": "f_B",
    "objective_unit": f_B_nescoil,
    "constraint_name": ("f_max_B2_self",),
    "constraint_type": ("<=",),
    # Unit is not necessary under normalized=True. Under this mode,
    # they'll be automatically calculated from typical values in DESC.
    "constraint_unit": (B2_self_target,),
    "constraint_value": jnp.array([B2_self_target,]),
    "phi_init_with_nescoil": False,
}

# ----- QSS routine -----

def quasi_single_stage(
    init_eq,
    file_name,
    quadcoil_kwargs_obj,
    objective_mode="free",  # "free" -> QuadcoilFreeBoundaryError, "fixed" -> QuadcoilProxy
    quadcoil_weight=0.0,
    vol_weight=0.0,
    iota_weight=0.0,
    qs_weight=1.0,
    maxiter=500,  # Maximum iteration per Fourier continuation
    step=2,
    max_k=None,  # Maximum continuation boundary mode number
    printout=False,  # Whether to run dummy optimization even when data exists for printout.
):

    # ----- Initializing objects and grids -----
    
    eqfam = EquilibriaFamily(init_eq)
    iotagridaxis = LinearGrid(
        M=init_eq.M_grid, N=init_eq.N_grid, NFP=init_eq.NFP, rho=np.array([0.01]), sym=True
    )
    iotagridedge = LinearGrid(
        M=init_eq.M_grid, N=init_eq.N_grid, NFP=init_eq.NFP, rho=np.array([1.0]), sym=True
    )
    out_list = []

    # ----- Fourier continuation -----
    
    if not max_k:
        max_k = init_eq.M + 1
    else:
        max_k = min(max_k, init_eq.M + 1)
    k_list = range(3, init_eq.M + 1, step)
    print("Boundary mode steps:", k_list)

    # ----- QSS loop -----
    
    for i in range(len(k_list)):
        k = k_list[i]
        filename_eq = file_name + "_eq_" + str(k) + ".h5"
        filename_qf = file_name + "_qf_" + str(k) + ".h5"
        filename_time = file_name + "_time_" + str(k) + ".npy"
        filename_history = file_name + "_history_" + str(k) + ".pickle"
        filename_log = file_name + "_log_" + str(k) + ".txt"
        
        #  ----- Constraints -----
        
        # as opposed to SIMSOPT and STELLOPT where variables are assumed fixed, in DESC
        # we assume variables are free. Here we decide which ones to fix, starting with
        # the major radius (R mode = [0,0,0]) and all modes with m,n > k
        R_modes = np.vstack(
            (
                [0, 0, 0],
                init_eq.surface.R_basis.modes[
                    np.max(np.abs(init_eq.surface.R_basis.modes), 1) > k, :
                ],
            )
        )
        Z_modes = init_eq.surface.Z_basis.modes[
            np.max(np.abs(init_eq.surface.Z_basis.modes), 1) > k, :
        ]
        # next we create the constraints, using the mode number arrays just created
        # if we didn't pass those in, it would fix all the modes (like for the profiles)
        constraints_base = [
            ForceBalance(eq=init_eq_k),
            FixBoundaryR(eq=init_eq_k, modes=R_modes),
            FixBoundaryZ(eq=init_eq_k, modes=Z_modes),
            # FixPsi(init_eq_k),
            FixCurrent(init_eq_k),
        ]
        
        # ----- Objectives -----
        
        init_eq_k = eqfam[-1].copy()
        qs_objective = QuasisymmetryTwoTerm(
            eq=init_eq_k,
            bounds=(-qs_bound, qs_bound),
            normalize_target=False,
            weight=qs_weight,
        )
        qs_objective.build()
        obj_list_base = [
            Volume(
                eq=init_eq_k,
                bound=(vol_l, vol_u)
                normalize_target=False,
                weight=vol_weight,
            ),
            # Axis rotational transform
            RotationalTransform(
                eq=init_eq_k,
                bound=(iota_l, iota_u)
                normalize_target=False,
                weight=iota_weight,
                grid=iotagridaxis,
            ),
            # Edge rotational transform
            RotationalTransform(
                eq=init_eq_k,
                bound=(iota_l, iota_u)
                normalize_target=False,
                weight=iota_weight,
                grid=iotagridedge,
            ),
            qs_objective,
        ]
        # QS objective
        # Mostly used in quasi-single-stage but also
        # used to get single-stage init guess
        if objective_mode == "free":
            quadcoil_fbe = QuadcoilFreeBoundaryError(
                eq=init_eq_k,
                quadcoil_kwargs=quadcoil_kwargs,
                enable_net_current_plasma=True,
                vacuum=vacuum,
                normalize=True,
                normalize_target=False,
                weight=quadcoil_weight,
            )
        elif objective_mode == "fixed":
            quadcoil_fbe = QuadcoilProxy(
                eq=init_eq_k,
                quadcoil_kwargs=quadcoil_kwargs,
                enable_net_current_plasma=True,
                vacuum=vacuum,
                metric_name=('f_B',),
                metric_target=np.array([0.,]),
                metric_weight=np.array([quadcoil_weight/f_B_nescoil,]),
                normalize=False,
                normalize_target=False,
                eq_fixed=False,  # Whether the equilibrium are fixed
            )
        else:
            raise ValueError(
                "objective_mode must be 'free' or 'fixed', got: "
                + repr(objective_mode)
            )
        quadcoil_fbe.build()
        objective = ObjectiveFunction(obj_list_base + [quadcoil_fbe])
        constraints = constraints_base
        # ----- Performing optimization -----
        try:
            # Run continuation step if the save file does not exist
            if not (os.path.exists(filename_history) and os.path.exists(filename_eq)):
                print("\n====================================")
                print("Optimizing boundary modes M,N <= {}".format(k))
                print("====================================")
                time1 = time.time()
                optimizer = Optimizer("proximal-lsq-auglag")
                eq_new, out = init_eq_k.optimize(
                    objective=objective,
                    constraints=constraints,
                    optimizer=optimizer,
                    maxiter=maxiter,
                    verbose=3,
                    ftol=1e-5,
                    copy=True,
                )
                # Printing continuation stage result
                qs_objective1 = qs_objective.compute_scalar(*qs_objective.xs(init_eq_k))
                quadcoil_fbe1 = quadcoil_fbe.compute_scalar(
                    *quadcoil_fbe.xs(init_eq_k)
                )
                qs_objective2 = qs_objective.compute_scalar(*qs_objective.xs(eq_new))
                quadcoil_fbe2 = quadcoil_fbe.compute_scalar(
                    *quadcoil_fbe.xs(eq_new)
                )
                print("Pre-optimization QS value:  ", qs_objective1)
                print("Pre-optimization FBE value: ", quadcoil_fbe1)
                print("Post-optimization QS value: ", qs_objective2)
                print("Post-optimization FBE value:", quadcoil_fbe2)
                time2 = time.time()
                jnp.save(filename_time, time2 - time1)
                eq_new.save(filename_eq)
                # Its important to use binary mode
                with open(filename_history, "wb") as dbfile:  # 'wb' = write binary
                    pickle.dump(out, dbfile)
                with open(filename_log, "w") as f:
                    f.write("Pre-optimization QS value:" + str(qs_objective1))
                    f.write(
                        "Pre-optimization quadcoil value:"
                        + str(quadcoil_fbe1)
                    )
                    f.write("Post-optimization QS value:" + str(qs_objective2))
                    f.write(
                        "Post-optimization quadcoil value:"
                        + str(quadcoil_fbe2)
                    )
                    f.write("=== Optimization Result ===\n")
                    f.write(pformat(dict(out), indent=2))
                    f.write("\n")
            # Load continuation step if the save file exists
            else:
                print("Step", k, "exists.")
                if printout and i == len(k_list) - 1:
                    print("Still running a dummy optimization to print out stuff.")
                    optimizer = Optimizer("proximal-lsq-auglag")
                    eq_new, out = init_eq_k.optimize(
                        objective=objective,
                        constraints=constraints,
                        optimizer=optimizer,
                        maxiter=1,
                        verbose=3,
                        ftol=1e-5,
                        copy=True,
                    )
                eq_new = desc.io.load(filename_eq, file_format="hdf5")
                with open(filename_history, "rb") as dbfile:  # 'wb' = write binary
                    out = pickle.load(dbfile)

            eqfam.append(eq_new)
            out_list.append(out)
            eqfam.save(file_name + "_eqfam_" + ".h5")
        except KeyboardInterrupt:
            break
        _, _, dofs_init, status_init = quadcoil_fbe.solve_quadcoil(
            *quadcoil_fbe.xs(eq_new)
        )

    return eqfam, out_list, quadcoil_fbe
