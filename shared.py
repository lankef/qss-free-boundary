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
    QuasisymmetryBoozer,
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


import jax

# ----- memory instrumentation -----

_DEV = jax.devices()[0]
_GiB = float(2 ** 30)
_HISTORY = []

# Important: This (along with the nvidia-smi call)
# views the pool size, peak use and largest memory 
# allocation request! It seems that most of the memory
# are used by one big request. 
def mem(tag):
    """Print JAX allocator stats. Safe to call anywhere."""
    print('mem')
    # try:
    #     s = _DEV.memory_stats() or {}
    # except Exception as e:                      # CPU backend, or unsupported
    #     print(f"[mem:{tag}] memory_stats unavailable: {e}", flush=True)
    #     return
    # rec = {
    #     'tag': tag,
    #     'in_use': s.get('bytes_in_use', 0) / _GiB,
    #     # bytes currently handed out by the BFC allocator and not yet freed
    #     # i.e. your live JAX arrays at the moment you call memory_stats().
    #     'peak': s.get('peak_bytes_in_use', 0) / _GiB, 
    #     'pool': s.get('pool_bytes', 0) / _GiB,
    #     'largest': s.get('largest_alloc_size', 0) / _GiB,
    #     'num_allocs': s.get('num_allocs', 0),
    # }
    # _HISTORY.append(rec)
    # print(
    #     f"[mem:{tag:>16}] in_use={rec['in_use']:7.3f} GiB  "
    #     f"peak={rec['peak']:7.3f} GiB  pool={rec['pool']:7.3f} GiB  "
    #     f"largest={rec['largest']:7.3f} GiB  n_alloc={rec['num_allocs']}",
    #     flush=True,
    # )


def mem_summary():
    print('mem summary')
    # if not _HISTORY:
    #     return
    # print("\n===== JAX memory summary =====", flush=True)
    # print(f"{'stage':>18}  {'in_use(GiB)':>11}  {'peak(GiB)':>10}", flush=True)
    # for r in _HISTORY:
    #     print(f"{r['tag']:>18}  {r['in_use']:11.3f}  {r['peak']:10.3f}", flush=True)
    # print(f"\nJAX peak_bytes_in_use = {_HISTORY[-1]['peak']:.3f} GiB", flush=True)
    # print("Compare against the max of mem_<jobid>.csv; the difference is "
    #       "non-JAX memory (cuDSS workspace, PETSc, CUDA context).", flush=True)
    
mem('*******     start')


# Creating toroidal vacuum initial state
vacuum = True
surf = FourierRZToroidalSurface(
    R_lmn=[8.0, 1.8, 1.7],
    Z_lmn=[-1.8, 1.7],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    NFP=2,
)
filename_init = 'init_eq.h5'
if os.path.exists(filename_init):
    eq_init = desc.io.load(filename_init, file_format="hdf5")
else:
    eq_init = Equilibrium(
        L=8, M=8, N=8, # L=16, M=16, N=16, 
        Psi=50.0, 
        surface=surf, 
        current=PowerSeriesProfile(),
        pressure=PowerSeriesProfile(),
        # pressure=pres, iota=iota
    )
    eq_init.solve(verbose=3)
    eq_init.save(filename_init)
# eq_init, info = eq_init.solve(, copy=False)

# ----- Targets -----

# Field-on-coil at the dipole layer
B2_self_target = 20.**2 
plasma_coil_distance = 1.5 
# Bound for 2-term QS error
qs_bound = 5e-5 # Helios boozer error is between 2.1e-3 and 5.8e-3
# Bounds for iota
iota_l, iota_u = 0.1, 0.4 # helios is 0.15
vol_l, vol_u = 450, 550 # Helios is 493
optimizer_name = "proximal-lsq-auglag"
jac_chunk_size = 32 
bs_chunk_size = 32

# ----- Quadcoil Resolution -----

mpol = 10  # Num. poloidal modes for the current potential
ntor = 10  # Num. toroidal modes for the current potential
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
    eq=eq_init.copy(),
    quadcoil_kwargs=quadcoil_kwargs_nescoil,
    vacuum=vacuum,
    # If you have additional filament/planar coils, put the CoilSet here.
    # field=[],
)
nescoil_objective.build()
# Solving the NESCOIL problem
out_dict_nescoil, qp_nescoil, dofs_nescoil, status_nescoil = \
    nescoil_objective.solve_quadcoil(*nescoil_objective.xs(eq_init))
# Calculating a reference f_B
f_B_nescoil = f_B(qp_nescoil, dofs_nescoil)
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

mem('*******     nescoil')

# ----- QSS routine -----

def quasi_single_stage(
    eq,
    file_name,
    quadcoil_kwargs_obj,
    objective_mode="free",  # "free" -> QuadcoilFreeBoundaryError, "fixed" -> QuadcoilProxy
    quadcoil_weight=0.0,
    vol_weight=0.0,
    iota_weight=0.0,
    qs_weight=1.0,
    maxiter=500,  # Maximum iteration per Fourier continuation
    step=4,
    max_k=None,  # Maximum continuation boundary mode number
    printout=False,  # Whether to run dummy optimization even when data exists for printout.
):

    # ----- Initializing objects and grids -----
    

    eqfam = EquilibriaFamily(eq)
    iotagridaxis = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.01]), sym=True
    )
    iotagridedge = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([1.0]), sym=True
    )
    out_list = []
    quadcoil_fbe = None

    
    mem('*******     eqfam')
    
    # ----- Fourier continuation -----
    
    if not max_k:
        max_k = eq.M + 1
    else:
        max_k = min(max_k, eq.M + 1)
    k_list = range(4, eq.M + 1, step)
    print("Boundary mode steps:", k_list)

    # ----- QSS loop -----
    # Rebuild objectives and constraints on eq_k each step. ProximalProjection
    # requires them to target the same Equilibrium object passed to optimize().

    if objective_mode not in ("free", "fixed", None):
        raise ValueError(
            "objective_mode must be 'free', 'fixed', or None, got: "
            + repr(objective_mode)
        )

    for i in range(len(k_list)):
        k = k_list[i]
        eq_k = eqfam[-1].copy()

        # ----- Objectives -----
        qs_objective = QuasisymmetryBoozer(
            eq=eq_k,
            bounds=(-qs_bound, qs_bound),
            normalize_target=False,
            weight=qs_weight,
        )
        qs_objective.build()
        obj_list_base = [
            Volume(
                eq=eq_k,
                bounds=(vol_l, vol_u),
                # normalize_target=False,
                weight=vol_weight,
            ),
            # Axis rotational transform
            RotationalTransform(
                eq=eq_k,
                bounds=(iota_l, iota_u),
                # normalize_target=False,
                weight=iota_weight,
                grid=iotagridaxis,
            ),
            # Edge rotational transform
            RotationalTransform(
                eq=eq_k,
                bounds=(iota_l, iota_u),
                # normalize_target=False,
                weight=iota_weight,
                grid=iotagridedge,
            ),
            qs_objective,
        ]
        if objective_mode == "free":
            quadcoil_fbe = QuadcoilFreeBoundaryError(
                eq=eq_k,
                quadcoil_kwargs=quadcoil_kwargs_obj,
                enable_net_current_plasma=True,
                vacuum=vacuum,
                normalize=True, # This combination: target in normalized
                normalize_target=False,
                weight=quadcoil_weight,
                bs_chunk_size=bs_chunk_size,
            )
            quadcoil_fbe.build()
            objective = ObjectiveFunction(obj_list_base + [quadcoil_fbe]) #, deriv_mode="batched", jac_chunk_size=1)
        elif objective_mode == "fixed":
            quadcoil_fbe = QuadcoilProxy(
                eq=eq_k,
                quadcoil_kwargs=quadcoil_kwargs_obj,
                enable_net_current_plasma=True,
                vacuum=vacuum,
                metric_name=('f_B',),
                metric_target=np.array([0.,]),
                metric_weight=np.array([quadcoil_weight/f_B_nescoil,]),
                normalize=False,
                normalize_target=False,
                eq_fixed=False,  # Whether the equilibrium are fixed
                bs_chunk_size=bs_chunk_size,
            )
            quadcoil_fbe.build()
            objective = ObjectiveFunction(obj_list_base + [quadcoil_fbe]) #, deriv_mode="batched", jac_chunk_size=1)
        else:
            quadcoil_fbe = None
            objective = ObjectiveFunction(obj_list_base) #, deriv_mode="batched", jac_chunk_size=1)

        # ----- Constraints -----
        # as opposed to SIMSOPT and STELLOPT where variables are assumed fixed, in DESC
        # we assume variables are free. Here we decide which ones to fix, starting with
        # the major radius (R mode = [0,0,0]) and all modes with m,n > k
        R_modes = np.vstack(
            (
                [0, 0, 0],
                eq_k.surface.R_basis.modes[
                    np.max(np.abs(eq_k.surface.R_basis.modes), 1) > k, :
                ],
            )
        )
        Z_modes = eq_k.surface.Z_basis.modes[
            np.max(np.abs(eq_k.surface.Z_basis.modes), 1) > k, :
        ]
        # next we create the constraints, using the mode number arrays just created
        # if we didn't pass those in, it would fix all the modes (like for the profiles)
        constraints = [
            ForceBalance(eq=eq_k, jac_chunk_size=jac_chunk_size),
            FixBoundaryR(eq=eq_k, modes=R_modes),
            FixBoundaryZ(eq=eq_k, modes=Z_modes),
            FixPsi(eq_k),
            FixCurrent(eq_k),
            FixPressure(eq_k),
        ]

        mem('*******     objectives and constraints built for k=' + str(k))
        
        filename_eq = file_name + "_eq_" + str(k) + ".h5"
        filename_qf = file_name + "_qf_" + str(k) + ".h5"
        filename_time = file_name + "_time_" + str(k) + ".npy"
        filename_history = file_name + "_history_" + str(k) + ".pickle"
        filename_log = file_name + "_log_" + str(k) + ".txt"
        
        # ----- Performing optimization -----
        
        try:
            # Run continuation step if the save file does not exist
            if not (os.path.exists(filename_history) and os.path.exists(filename_eq)):
                print("\n====================================")
                print("Optimizing boundary modes M,N <= {}".format(k))
                print("====================================")
                qs_objective1 = qs_objective.compute_scalar(*qs_objective.xs(eq_k))
                print("Pre-optimization QS value:  ", qs_objective1)
                if objective_mode is not None:
                    quadcoil_fbe1 = quadcoil_fbe.compute_scalar(*quadcoil_fbe.xs(eq_k))
                    print("Pre-optimization FBE value: ", quadcoil_fbe1)
                mem('*******     starting step: '+str(k))
                time1 = time.time()
                optimizer = Optimizer(optimizer_name)
                eq_new, out = eq_k.optimize(
                    objective=objective,
                    constraints=constraints,
                    optimizer=optimizer,
                    maxiter=maxiter,
                    verbose=3,
                    ftol=1e-5,
                    copy=True,
                )
                mem('*******     finishing step: '+str(k))
                # Printing continuation stage result
                qs_objective2 = qs_objective.compute_scalar(*qs_objective.xs(eq_new))
                print("Post-optimization QS value: ", qs_objective2)
                if objective_mode is not None:
                    quadcoil_fbe2 = quadcoil_fbe.compute_scalar(*quadcoil_fbe.xs(eq_new))
                    print("Post-optimization FBE value:", quadcoil_fbe2)
                time2 = time.time()
                jnp.save(filename_time, time2 - time1)
                eq_new.save(filename_eq)
                # Its important to use binary mode
                with open(filename_history, "wb") as dbfile:  # 'wb' = write binary
                    pickle.dump(out, dbfile)
                with open(filename_log, "w") as f:
                    f.write("Pre-optimization QS value:" + str(qs_objective1))
                    f.write("Post-optimization QS value:" + str(qs_objective2))
                    if objective_mode is not None:
                        f.write(
                            "Pre-optimization quadcoil value:"
                            + str(quadcoil_fbe1)
                        )
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
                    optimizer = Optimizer(optimizer_name)
                    eq_new, out = eq_k.optimize(
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
        
        if objective_mode is not None:
            _, _, dofs_init, status_init = quadcoil_fbe.solve_quadcoil(
                *quadcoil_fbe.xs(eq_new)
            )
            
    return eqfam, out_list, quadcoil_fbe
