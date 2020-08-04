__precompile__()

module VersatileHDPMixtureModels

using Distributed
using StatsBase
using Distributions
using SpecialFunctions
using LinearAlgebra
using JLD2
using Clustering
using Random
using NPZ
using Base
using PDMats

#DS:
include("ds.jl")

#Distributions:
include("distributions/mv_gaussian.jl")
include("distributions/mv_group_gaussian.jl")
include("distributions/multinomial_dist.jl")
include("distributions/topic_modeling_dist.jl")

#Priors:
include("priors/multinomial_prior.jl")
include("priors/topic_modeling_prior.jl")
include("priors/niw.jl")
include("priors/niw_stable_var.jl")
include("priors/bayes_network_model.jl")

#Rest:
include("params_base.jl")
include("utils.jl")
include("shared_actions.jl")
include("local_clusters_actions.jl")
include("global_clusters_actions.jl")
include("crf_hdp.jl")
include("gaussian_generator.jl")
include("hdp_shared_features.jl")

#Data Generators:
export 
    generate_sph_gaussian_data,
    generate_gaussian_data,
    generate_mnmm_data,
    generate_grouped_mnm_data,
    generate_grouped_gaussian_data,
    create_mnmm_data,
    create_gaussian_data,
    create_grouped_gaussian_data,
    create_grouped_mnmm_data,
    hdp_prior_crf_draws,
    generate_grouped_gaussian_from_hdp_group_counts,
#Model DS:    
    hdp_shared_features,
    multinomial_dist,
    mv_gaussian, 
    mv_group_gaussian,
    topic_modeling_dist,
    bayes_network_model,
    niw_hyperparams,
    niw_stable_hyperparams,
    topic_modeling_hyper,
#Functions
    hdp_fit,
    vhdp_fit, 
    create_default_priors,
    get_model_global_pred

end # module
