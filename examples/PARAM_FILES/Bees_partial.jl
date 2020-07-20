include("../CODE/ds.jl")
include("../CODE/distributions/niw.jl")

using LinearAlgebra


#Global Setting
use_gpu = false
use_darrays = false #Only relevant if use_gpu = false



random_seed = nothing

#Data Loading specifics
data_path = "../DATA/Bees/"
data_prefix = "XXX"
groups_count = 4
global_preprocessing = nothing
local_preprocessing = nothing



#Model Parameters
iterations = 100
hard_clustering = false

total_dim = 10
local_dim = 7

α = 10.0
γ = 1000000000000.0
global_weight = 1.0
local_weight= 1.0

initial_global_clusters = 1
initial_local_clusters = 1


use_dict_for_global = false

ignore_local = true

split_stop = 5
argmax_sample_stop = 5


glob_dim = 6
global_hyper_params = niw_hyperparams(1.0,
    zeros(glob_dim),
    glob_dim+3,
    Matrix{Float64}(I, glob_dim, glob_dim)*0.1)



local_mult = 0.1

count_list = [1,1,1,1]






local_hyper_params = [niw_hyperparams(1.0, zeros(i),i+3,Matrix{Float64}(I, i, i)*local_mult) for i in count_list]
#local_hyper_params = [niw_hyperparams(1.0, zeros(5),8+3,Matrix{Float64}(I, 5, 5)*local_mult) for i in count_list]

# local_hyper_params = [niw_hyperparams(1.0,
#     zeros(3),
#     6,
#     Matrix{Float64}(I, 3, 3)*local_mult),
#     niw_hyperparams(1.0,
#         zeros(6),
#         9.0,
#         Matrix{Float64}(I, 6, 6)*local_mult),
#     niw_hyperparams(1.0,
#         zeros(9),
#         12.0,
#         Matrix{Float64}(I, 9, 9)*local_mult),
#     niw_hyperparams(1.0,
#         zeros(12),
#         15.0,
#         Matrix{Float64}(I, 12, 12)*local_mult),
#     niw_hyperparams(1.0,
#         zeros(15),
#         18.0,
#         Matrix{Float64}(I, 15, 15)*local_mult),
#     niw_hyperparams(1.0,
#         zeros(18),
#         21.0,
#         Matrix{Float64}(I, 18, 18)*local_mult),
#     niw_hyperparams(1.0,
#         zeros(21),
#         24.0,
#         Matrix{Float64}(I, 21, 21)*local_mult),
#     niw_hyperparams(1.0,
#         zeros(24),
#         27.0,
#         Matrix{Float64}(I, 24, 24)*local_mult),
#     niw_hyperparams(1.0,
#         zeros(27),
#         30.0,
#         Matrix{Float64}(I, 27, 27)*local_mult)]


# local_hyper_params = [niw_hyperparams(1.0,
#     zeros(3),
#     100.0,
#     Matrix{Float64}(I, 3, 3)*10.0),
#     niw_hyperparams(1.0,
#         zeros(6),
#         100.0,
#         Matrix{Float64}(I, 6, 6)*10.0),
#     niw_hyperparams(1.0,
#         zeros(9),
#         100.0,
#         Matrix{Float64}(I, 9, 9)*10.0),
#     niw_hyperparams(1.0,
#         zeros(12),
#         100.0,
#         Matrix{Float64}(I, 12, 12)*10.0),
#     niw_hyperparams(1.0,
#         zeros(15),
#         100.0,
#         Matrix{Float64}(I, 15, 15)*10.0),
#     niw_hyperparams(1.0,
#         zeros(18),
#         100.0,
#         Matrix{Float64}(I, 18, 18)*10.0),
#     niw_hyperparams(1.0,
#         zeros(21),
#         100.0,
#         Matrix{Float64}(I, 21, 21)*10.0),
#     niw_hyperparams(1.0,
#         zeros(24),
#         100.0,
#         Matrix{Float64}(I, 24, 24)*10.0),
#     niw_hyperparams(1.0,
#         zeros(27),
#         100.0,
#         Matrix{Float64}(I, 27, 27)*10.0)]
