from shared import *


# ## Optimization

quadcoil_weight = 5000. # 5. # 50. -> 90% improvement, 10x degradation in QS
qs_weight = 5000. # 5000.
vol_weight = 30.
iota_weight = 30.


data_dir = 'data'
plot_dir = 'plots'
init_eq = qa_eq.copy()
# init_eq.iota = init_eq.get_profile("iota")

eqfam_f_max_phi, out_list_f_max_phi, quadcoil_objective_f_max_phi = quasi_single_stage(
    init_eq=init_eq,
    file_name=data_dir + '/' + 'f_max_B2',
    quadcoil_kwargs_obj=quadcoil_kwargs,
    objective_mode="free",
    quadcoil_weight=quadcoil_weight,
    qs_weight=qs_weight,
    vol_weight=vol_weight,
    iota_weight=iota_weight,
    printout=True
)
