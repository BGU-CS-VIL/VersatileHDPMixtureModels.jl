#Global Setting
use_gpu = false
use_darrays = false #Only relevant if use_gpu = false


hard_clustering = false
ignore_local = false

global_weight = 1.0
local_weight= 1.0

split_delays = false



# #Model Parameters
# iterations = 10
#
# total_dim = 3
# local_dim = 3
#
# α = 4.0
# γ = 2.0
# global_weight = 1.0
# local_weight= 1.0
#
#
# global_hyper_params = niw_hyperparams(5.0,
#     ones(3)*2.5,
#     10.0,
#     [[0.8662817 0.78323282 0.41225376];[0.78323282 0.74170384 0.50340258];[0.41225376 0.50340258 0.79185577]])
#
# local_hyper_params = niw_hyperparams(1.0,
#     [217.0,510.0],
#     10.0,
#     Matrix{Float64}(I, 2, 2)*0.5)
