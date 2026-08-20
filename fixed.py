import sys
import os
import time
import pickle
from pprint import pformat
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

from desc import set_device
set_device("gpu")

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
    SurfaceCurrentRegularization,
    QuasisymmetryTripleProduct,
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
# Quadcoil import 
from quadcoil.quantity import f_B, Phi_with_net_current, Phi

qa_eq = desc.examples.get("reactor_QA")
qa_eq.change_resolution(L=8, M=8, N=8)
qa_eq.current = PowerSeriesProfile() 
qa_eq, info = qa_eq.solve(verbose=3, copy=False)
B2_self_target = 20.**2
vacuum = True

# Settings
mpol = 8  # Num. poloidal modes for the current potential
ntor = 8  # Num. toroidal modes for the current potential
# Controls the resolution of the plasma surface integration.
# Integration in quadcoil is naively performed using summation
# so we recommend at least 16 here.
# This corresponds to a (33 x 33) grid.
plasma_coil_distance = 1.3 # 1.3
# coil_coil_distance = 0.77  # 1.10 is the Wiedman value
# coil_per_half_fp = 3  # 3 is the Wiedman value
# curvature_target = 0.88  # 0.88 is the Wiedman value
# f_B_target_norm = 1e-4
# radius_of_curvature = 0.5 # 0.5m is the infinity two value
# Resolution for sampling objectives
quadpoints_phi = jnp.linspace(0, 1 / qa_eq.NFP, 32, endpoint=False)
quadpoints_theta = jnp.linspace(0, 1, 32, endpoint=False)
qs_multiple = 10


# In[9]:


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


# In[10]:


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
# Define a QuadcoilProxy with the simplest possible
# signature.
nescoil_objective = QuadcoilProxy(
    eq=qa_eq,
    quadcoil_kwargs=quadcoil_kwargs_nescoil,
    vacuum=vacuum,
    # If you have additional filament/planar coils, put the CoilSet here.
    # field=[],
)
nescoil_objective.build()

# Solving the NESCOIL problem
out_dict_nescoil, qp_nescoil, dofs_nescoil, status_nescoil = (
    nescoil_objective.solve_quadcoil(
        # Like Objective.compute(), the xs of the equilibrium
        # must be passed in as the *arg.
        *nescoil_objective.xs(qa_eq)
    )
)

# Defining problem
quadcoil_kwargs = quadcoil_kwargs_basic | {
    "objective_name": "f_B",
    "objective_unit": f_B(qp_nescoil, dofs_nescoil),
    "constraint_name": ("f_max_B2_self",),
    "constraint_type": ("<=",),
    # Unit is not necessary under normalized=True. Under this mode,
    # they'll be automatically calculated from typical values in DESC.
    "constraint_unit": (B2_self_target,),
    "constraint_value": jnp.array([B2_self_target,]),
    "phi_init_with_nescoil": False,
}


# Quasi-single-stage with continuation
def quasi_single_stage(
    init_eq,
    file_name,
    quadcoil_kwargs_obj,
    quadcoil_weight=0.0,
    vol_weight=0.0,
    iota_weight=0.0,
    qs_weight=1.0,
    maxiter=500,  # Maximum iteration per Fourier continuation
    step=1,
    max_k=None,  # Maximum continuation boundary mode number
    printout=False,  # Whether to run dummy optimization even when data exists for printout.
):

    # ----- Calculating targets -----
    # Equilibrium optimization targets in this examples are
    # calculated from the initial state, so that the QUADCOIL
    # proxy is minimized while maintaining other plasma parameters.
    # Building grids
    # Volume
    vol = init_eq.compute(["V"])["V"]
    eqfam = EquilibriaFamily(init_eq)
    # # Triple product QS
    qs_objective_init = QuasisymmetryTripleProduct(
        init_eq,
    )
    qs_objective_init.build()
    qs_bound = jnp.abs(
        qs_objective_init.compute(*qs_objective_init.xs(init_eq))
        / qs_objective_init.normalization
    ) * qs_multiple
    iotagridaxis = LinearGrid(
        M=init_eq.M_grid, N=init_eq.N_grid, NFP=init_eq.NFP, rho=np.array([0.01]), sym=True
    )
    iotagridedge = LinearGrid(
        M=init_eq.M_grid, N=init_eq.N_grid, NFP=init_eq.NFP, rho=np.array([1.0]), sym=True
    )
    iota_axis = init_eq.compute(['iota'], grid=iotagridaxis)['iota'][0]
    iota_edge = init_eq.compute(['iota'], grid=iotagridedge)['iota'][0]
    
    out_list = []

    # ----- Fourier continuation -----
    if not max_k:
        max_k = init_eq.M + 1
    else:
        max_k = min(max_k, init_eq.M + 1)
    k_list = range(3, init_eq.M + 1, step)
    print("Boundary mode steps:", k_list)
    for i in range(len(k_list)):
        k = k_list[i]
        filename_eq = file_name + "_eq_" + str(k) + ".h5"
        filename_qf = file_name + "_qf_" + str(k) + ".h5"
        filename_time = file_name + "_time_" + str(k) + ".npy"
        filename_history = file_name + "_history_" + str(k) + ".pickle"
        filename_log = file_name + "_log_" + str(k) + ".txt"

        # ----- Objectives -----
        init_eq_k = eqfam[-1].copy()
        qs_objective = QuasisymmetryTripleProduct(
            eq=init_eq_k,
            bounds=(-qs_bound, qs_bound),
            normalize_target=False,
            weight=qs_weight,
        )
        qs_objective.build()
        obj_list_base = [
            Volume(
                eq=init_eq_k,
                target=vol,
                weight=vol_weight,
            ),
            # Axis rotational transform
            RotationalTransform(
                eq=init_eq_k,
                target=iota_axis,
                weight=iota_weight,
                grid=iotagridaxis,
            ),
            # Edge rotational transform
            RotationalTransform(
                eq=init_eq_k,
                target=iota_edge,
                weight=iota_weight,
                grid=iotagridedge,
            ),
            qs_objective,
        ]
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
            FixPsi(init_eq_k),
            FixCurrent(init_eq_k),
            # FixPressure(init_eq_k),
            # Equilibrium is now loaded with fixed iota because we don't
            # care about enforcing vacuum field any more.
            # FixIota(init_eq_k),
        ]
        # QS objective
        # Mostly used in quasi-single-stage but also
        # used to get single-stage init guess
        quadcoil_fbe = QuadcoilProxy(
            eq=init_eq_k,
            quadcoil_kwargs=quadcoil_kwargs,
            enable_net_current_plasma=True,
            vacuum=vacuum,
            # WARNING: If changing this impacts the sln, then the unit 
            # generators may not be working!!!
            metric_name=('f_B',),
            metric_target=np.array([0.,]),
            metric_weight=np.array([quadcoil_weight/f_B(qp_nescoil, dofs_nescoil),]),
            normalize=False,
            normalize_target=False,
            eq_fixed=False, # Whether the equilibrium are fixed
        )
        quadcoil_fbe.build()
        objective = ObjectiveFunction(obj_list_base + [quadcoil_fbe])
        constraints = constraints_base
        # ----- Performing optimization -----
        try:
            # Run continuation step if the save file does not exist
            if not (os.path.exists(filename_history) and os.path.exists(filename_eq)):
                print("\n==================================")
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
                print("Pre-optimization QS value:", qs_objective1)
                print("Pre-optimization quadcoil value:", quadcoil_fbe1)
                print("Post-optimization QS value:", qs_objective2)
                print("Post-optimization quadcoil value:", quadcoil_fbe2)
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


# ## Optimization

# In[79]:


quadcoil_weight = 10. # 5. # 50. -> 90% improvement, 10x degradation in QS
qs_weight = 5000. # 5000.
vol_weight = 30.
iota_weight = 30.


# In[80]:


data_dir = 'data_fixed' 
plot_dir = 'plots'
init_eq = qa_eq.copy()
# init_eq.iota = init_eq.get_profile("iota")

eqfam_f_max_phi, out_list_f_max_phi, quadcoil_objective_f_max_phi = quasi_single_stage(
    init_eq=init_eq, 
    file_name=data_dir + '/' + 'f_max_B2',
    quadcoil_kwargs_obj=quadcoil_kwargs,
    quadcoil_weight=quadcoil_weight,
    qs_weight=qs_weight,
    vol_weight=vol_weight,
    iota_weight=iota_weight,
    printout=True
)


# In[ ]:




