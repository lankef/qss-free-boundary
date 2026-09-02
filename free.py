from shared import *


# ## Optimization
print('OPTIMIZING FOR FBE')

quadcoil_weight = 250. # 5. # 50. -> 90% improvement, 10x degradation in QS
qs_weight = 500. # 5000.
vol_weight = 30.
iota_weight = 30.

data_dir = 'data'
plot_dir = 'plots'
# eq_init.iota = eq_init.get_profile("iota")
with jax.profiler.trace("./tmp/jax-trace", create_perfetto_link=True):
    eqfam_f_max_phi, out_list_f_max_phi, quadcoil_objective_f_max_phi = quasi_single_stage(
        eq=eq_init,
        file_name=data_dir + '/' + 'f_max_B2',
        quadcoil_kwargs_obj=quadcoil_kwargs,
        objective_mode="free",
        quadcoil_weight=quadcoil_weight,
        qs_weight=qs_weight,
        vol_weight=vol_weight,
        iota_weight=iota_weight,
        printout=True,
        maxiter=2,
    )
