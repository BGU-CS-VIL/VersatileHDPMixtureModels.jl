include("ds.jl")
include("distributions/niw.jl")

using LinearAlgebra


#Global Setting
use_gpu = false
use_darrays = false #Only relevant if use_gpu = false



random_seed = 1234567
random_seed = nothing

#Data Loading specifics
data_path = "data/multi-d-gaussians//"
data_prefix = "g"
groups_count = 9
global_preprocessing = nothing
local_preprocessing = nothing



#Model Parameters
iterations = 100
hard_clustering = false

total_dim = 3
local_dim = 3

α = 500.0
γ = 10000.0
global_weight = 1.0
local_weight= 1.0

initial_global_clusters = 1
initial_local_clusters = 1 #this is per group


use_dict_for_global = false

split_delays = false

ignore_local = false

split_stop = 10
argmax_sample_stop = 10

use_mean_for_merge = false


global_hyper_params = niw_hyperparams(1.0,
    zeros(2),
    20.0,
    Matrix{Float64}(I, 2, 2)*1)

local_hyper_params = niw_hyperparams(1.0,
    zeros(1),
    20.0,
    Matrix{Float64}(I, 1, 1)*1)





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
