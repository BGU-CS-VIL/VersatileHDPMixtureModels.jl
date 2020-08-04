abstract type distribution_hyper_params end
#Suff statistics must contain N which is the number of points associated with the cluster
abstract type sufficient_statistics end
abstract type distibution_sample end
import Base.copy



struct model_hyper_params
    global_hyper_params::distribution_hyper_params
    local_hyper_params
    α::Float64
    γ::Float64
    η::Float64
    global_weight::Float64
    local_weight::Float64
    total_dim::Int64
    local_dim::Int64
end

mutable struct cluster_parameters
    hyperparams::distribution_hyper_params
    distribution::distibution_sample
    suff_statistics::sufficient_statistics
    posterior_hyperparams::distribution_hyper_params
end

mutable struct splittable_cluster_params
    cluster_params::cluster_parameters
    cluster_params_l::cluster_parameters
    cluster_params_r::cluster_parameters
    lr_weights::AbstractArray{Float64, 1}
    splittable::Bool
    logsublikelihood_hist::AbstractArray{Float64,1}
end

mutable struct global_cluster
    cluster_params::splittable_cluster_params
    total_dim::Int64
    local_dim::Int64
    points_count::Int64
    clusters_count::Int64
    clusters_sub_counts::AbstractArray{Int64, 1}
end

mutable struct local_cluster
    cluster_params::splittable_cluster_params
    total_dim::Int64
    local_dim::Int64
    points_count::Int64
    global_weight::Float64
    local_weight::Float64
    globalCluster::Int64
    globalCluster_subcluster::Int64 #1 for left, 2 for right
    global_suff_stats::AbstractArray{Int64, 1}
end

mutable struct local_group
    model_hyperparams::model_hyper_params
    points::AbstractArray{Float64,2}
    labels::AbstractArray{Int64,2}
    labels_subcluster::AbstractArray{Int64,2}
    local_clusters::Vector{local_cluster}
    weights::Vector{Float64}
    group_num::Int64
end

mutable struct local_group_stats
    labels::AbstractArray{Int64,2}
    labels_subcluster::AbstractArray{Int64,2}
    local_clusters::Vector{local_cluster}
end


mutable struct hdp_shared_features
    model_hyperparams::model_hyper_params
    groups_dict::Dict
    global_clusters::Vector{global_cluster}
    weights::AbstractArray{Float64, 1}
end

function copy_local_cluster(c::local_cluster)
    return deepcopy(c)
end

function update_group_from_stats!(group::local_group, stats::local_group_stats)
    group.labels = stats.labels
    group.labels_subcluster = stats.labels_subcluster
    group.local_clusters = stats.local_clusters
end
