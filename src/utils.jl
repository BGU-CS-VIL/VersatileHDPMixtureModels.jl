# We expects the data to be in npy format, return a dict of {group: items}, each file is a different group
function load_data(path::String,groupcount::Number; prefix::String="", swapDimension::Bool = true)
    groups_dict = Dict()
    for i = 1:groupcount
        arr = npzread(path * prefix * string(i) * ".npy")
        for (index, value) in enumerate(arr)
            if isnan(value)
                arr[index] = 0.0
            end
        end
        groups_dict[i] = swapDimension ? transpose(arr) : arr
    end
    return groups_dict
end

# Preprocessing on the samples, global_preprocessing and local_preprocessing are functions. only same dimesions input output are supported atm
function preprocessing!(samples_dict,local_dim::Number, global_preprocessing, local_preprocessing)
    gp = x->Base.invokelatest(global_preprocessing,x)
    lp = x->Base.invokelatest(local_preprocessing,x)
    if global_preprocessing == nothing && local_preprocessing == nothing
        return
    end
    for (k,v) in samples_dict
        if global_preprocessing != nothing
            samples_dict[k][1 : local_dim-1,:] = mapslices(gp,v[1 : local_dim-1,:], dims= [2])
        end
        if local_preprocessing != nothing
            samples_dict[k][local_dim : end,:] = mapslices(lp,v[local_dim : end,:], dims= [2])
        end
    end
end


function dcolwise_dot!(r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    n = length(r)
    for j = 1:n
        v = zero(promote_type(eltype(a), eltype(b)))
        for i = 1:size(a, 1)
            @inbounds v += a[i, j]*b[i, j]
        end
        r[j] = v
    end
end

function distributer_factory(arr::AbstractArray)
    function distributer!(a,b)
        #for arg in args
        try
            show(a)
            show(b)
            arr[b[1]] = b[2]
        catch y
            show(a)
            show(b)
            println("Exception: ", y)
        end
        return -1
        #end

    end
    return distributer!
end

# Note that we expect the log_likelihood_array to be in rows (samples) x columns (clusters) , this is due to making it more efficent that way.
# function sample_log_cat_array!(labels::AbstractArray{Int64,2}, log_likelihood_array::AbstractArray{Float64,2})
#     # println("lsample log cat" * string(log_likelihood_array))
#     max_log_prob_arr = maximum(log_likelihood_array, dims = 2)
#     log_likelihood_array .-= max_log_prob_arr
#     map!(exp,log_likelihood_array,log_likelihood_array)
#     # println("lsample log cat2" * string(log_likelihood_array))
#     #sum_prob_arr = sum(log_likelihood_array, dims =[2])
#     sum_prob_arr = (cumsum(log_likelihood_array, dims =2))
#     randarr = rand(length(labels)) .* sum_prob_arr[:,(size(sum_prob_arr,2))]
#     sum_prob_arr .-= randarr
#     sum_prob_arr[sum_prob_arr .< 0] .= maxintfloat()
#     #replace!(x -> x < 0 ? maxintfloat() : x, sum_prob_arr)
#     labels .= mapslices(argmin, sum_prob_arr, dims= [2])
# end

function sample_log_cat_array!(labels::AbstractArray{Int64,2}, log_likelihood_array::AbstractArray{Float64,2})
    # println("lsample log cat" * string(log_likelihood_array))
    log_likelihood_array[isnan.(log_likelihood_array)] .= -Inf #Numerical errors arent fun
    max_log_prob_arr = maximum(log_likelihood_array, dims = 2)
    log_likelihood_array .-= max_log_prob_arr
    map!(exp,log_likelihood_array,log_likelihood_array)
    # println("lsample log cat2" * string(log_likelihood_array))
    sum_prob_arr = sum(log_likelihood_array, dims =[2])
    log_likelihood_array ./=  sum_prob_arr
    for i=1:length(labels)
        labels[i,1] = sample(1:size(log_likelihood_array,2), ProbabilityWeights(log_likelihood_array[i,:]))
    end
end


function sample_log_cat(logcat_array::AbstractArray{Float64, 1})
    max_logprob::Float64 = maximum(logcat_array)
    for i=1:length(logcat_array)
        logcat_array[i] = exp(logcat_array[i]-max_logprob)
    end
    sum_logprob::Float64 = sum(logcat_array)
    i::Int64 = 1
    c::Float64 = logcat_array[1]
    u::Float64 = rand()*sum(logcat_array)
    while c < u && i < length(logcat_array)
        c += logcat_array[i += 1]
    end
    return i
end


function create_sufficient_statistics(dist::distribution_hyper_params, pts::Array{Any,1})
    return create_sufficient_statistics(dist,dist, Array{Float64}(undef, 0, 0))
end


function get_labels_histogram(labels)
    hist_dict = Dict()
    for v in labels
        if haskey(hist_dict,v) == false
            hist_dict[v] = 0
        end
        hist_dict[v] += 1
    end
    return sort(collect(hist_dict), by=x->x[1])
end

function create_global_labels(group::local_group)
    clusters_dict = Dict()
    for (i,v) in enumerate(group.local_clusters)
        clusters_dict[i] = v.globalCluster
    end
    return [clusters_dict[i] for i in group.labels]
end


function print_global_sub_cluster(group::local_group)
    println([v.globalCluster for v in group.local_clusters])
    println([v.globalCluster_subcluster for v in group.local_clusters])
end


function print_groups_global_clusters(model::hdp_shared_features)
    for (k,g) in model.groups_dict
        println("Group: " * string(k) * "Global Clusters: " * string([x.globalCluster for x in g.local_clusters]))
    end
end

function axes_swapper(groups_pts_dict::Dict, axes_swap_vector)
    for (k,v) in groups_pts_dict
        groups_pts_dict[k] = v[axes_swap_vector,:]
    end
    return groups_pts_dict
end



function create_params_jld(jld_path,
     random_seed,
     data_path,
     data_prefix,
     groups_count,
     global_preprocessing,
     local_preprocessing,
     iterations,
     hard_clustering,
     total_dim,
     local_dim,
     split_stop,
     argmax_sample_stop,
     α,
     γ,
     global_weight,
     local_weight,
     initial_global_clusters,
     initial_local_clusters,
     global_hyper_params,
     local_hyper_params)
     @save jld_path random_seed data_path data_prefix groups_count global_preprocessing local_preprocessing iterations hard_clustering total_dim local_dim split_stop argmax_sample_stop α γ global_weight local_weight initial_global_clusters initial_local_clusters global_hyper_params local_hyper_params
end


function print_params_to_files(file_path,
     random_seed,
     iterations,
     hard_clustering,
     split_stop,
     argmax_sample_stop,
     α,
     γ,
     global_weight,
     local_weight,
     initial_global_clusters,
     initial_local_clusters,
     global_hyper_params,
     local_hyper_params,
     global_multiplier = 0,
     local_multiplier = 0)
    io = open(file_path, "w+")
    println(io, "random_seed = " * string(random_seed))
    println(io, "iterations = " * string(iterations))
    println(io, "hard_clustering = " * string(hard_clustering))
    println(io, "split_stop = " * string(split_stop))
    println(io, "argmax_sample_stop = " * string(argmax_sample_stop))
    println(io, "α = " * string(α))
    println(io, "γ = " * string(γ))
    println(io, "global_weight = " * string(global_weight))
    println(io, "local_weight = " * string(local_weight))
    println(io, "initial_global_clusters = " * string(initial_global_clusters))
    println(io, "initial_local_clusters = " * string(initial_local_clusters))
    println(io, "global_hyper_params = " * string(global_hyper_params))
    println(io, "local_hyper_params = " * string(local_hyper_params))
    println(io, "global_multiplier = " * string(global_multiplier))
    println(io, "local_multiplier = " * string(local_multiplier))
    close(io)
 end


 function get_node_leaders_dict()
     leader_dict = Dict()
     cur_leader = 2
     leader_dict[cur_leader] = []
     for i in workers()
         if i in procs(cur_leader)
             push!(leader_dict[cur_leader], i)
         else
             cur_leader = i
             leader_dict[cur_leader] = [i]
         end
     end
     return leader_dict
 end

function assign_group_leaders(groups_count, leader_dict)
    group_assignments = zeros(groups_count)
    group_leaders = collect(keys(leader_dict))
    for i=1:length(group_assignments)
        group_assignments[i] = group_leaders[i%length(group_leaders) +1 ]
    end
    return group_assignments
end

function log_multivariate_gamma(x::Number, D::Number)
    res::Float64 = D*(D-1)/4*log(pi)
    for d = 1:D
        res += logabsgamma(x+(1-d)/2)[1]
    end
    return res
end
