function create_first_local_cluster(group::local_group, max_global::Int64 = 1)
    suff = create_sufficient_statistics(group.model_hyperparams.local_hyper_params, [])
    post = group.model_hyperparams.local_hyper_params
    dist = sample_distribution(post)
    cp = cluster_parameters(group.model_hyperparams.local_hyper_params, dist, suff, post)
    cpl = deepcopy(cp)
    cpr = deepcopy(cp)
    splittable = splittable_cluster_params(cp,cpl,cpr,[0.5,0.5], false, ones(20).*-Inf)
    update_splittable_cluster_params!(splittable, (@view group.points[group.model_hyperparams.local_dim : end,:]), (@view group.labels_subcluster[:]), false)
    cluster = local_cluster(splittable, group.model_hyperparams.total_dim,
        group.model_hyperparams.local_dim,splittable.cluster_params.suff_statistics.N,1.0,1.0,rand(1:max_global),rand(1:2),[])
    return cluster
end


function sample_sub_clusters!(group::local_group, final::Bool)
    for (i,v) in enumerate(group.local_clusters)
        create_subclusters_labels!(reshape((@view group.labels_subcluster[group.labels .== i]),:,1),
            (@view group.points[:,(@view (group.labels .== i)[:])]), v.cluster_params, global_clusters_vector[v.globalCluster].cluster_params, v.local_dim, group.group_num, final)
    end
end

function create_subclusters_labels!(labels::AbstractArray{Int64,2},
        points::AbstractArray{Float64,2},
        cluster_params::splittable_cluster_params,
        global_cluster_params::splittable_cluster_params,
        local_dim::Int64,
        group_num::Int64, final::Bool)
    #lr_arr = create_array(zeros(length(labels), 2))
    if size(labels,1) == 0
        return
    end
    parr = zeros(length(labels), 6)
    log_likelihood!((@view parr[:,5]),(@view points[1:local_dim-1, : ]),global_cluster_params.cluster_params_l.distribution, group_num)
    log_likelihood!((@view parr[:,6]),(@view points[1:local_dim-1, : ]),global_cluster_params.cluster_params_r.distribution, group_num)
    if ignore_local == false
        log_likelihood!((@view parr[:,1]),(@view points[local_dim:end, : ]),cluster_params.cluster_params_l.distribution)
        log_likelihood!((@view parr[:,4]),(@view points[local_dim:end, : ]),cluster_params.cluster_params_r.distribution)
    end
    # println(global_cluster_params.cluster_params_l.distribution)
    # println(global_cluster_params.cluster_params_r.distribution)
    parr[:,5] .+= log(global_cluster_params.lr_weights[1])
    parr[:,6] .+= log(global_cluster_params.lr_weights[2])
    parr[:,1] .+= log(cluster_params.lr_weights[1])
    parr[:,4] .+= log(cluster_params.lr_weights[2])
    parr[:,3] .= parr[:,1]+ parr[:,6]
    parr[:,1] .+= parr[:,5]
    parr[:,2] .= parr[:,5] + parr[:,4]
    parr[:,4] .+= parr[:,6]
    if final
        labels .= mapslices(argmax, parr, dims= [2])
    else
        sample_log_cat_array!(labels,parr[:,1:4])
    end
end

function get_local_cluster_likelihood!(parr::AbstractArray{Float64,2}, points::AbstractArray{Float64,2}, cluster::local_cluster, group_num::Int64)
    global_p = zeros(size(parr))
    local_dim = cluster.local_dim
    log_likelihood!(global_p, (@view points[1 : local_dim-1,:]),global_clusters_vector[cluster.globalCluster].cluster_params.cluster_params.distribution, group_num)
    if ignore_local == false
        log_likelihood!(parr, (@view points[local_dim:end,:]),cluster.cluster_params.cluster_params.distribution)
    end
    parr .*= (1 / cluster.local_weight)
    parr .+= (global_p .* (1 / cluster.global_weight))
end


function sample_labels!(group::local_group, final::Bool)
    sample_labels!(group.labels, group.points, group.local_clusters, group.weights, final, group.group_num)
end

function sample_labels!(labels::AbstractArray{Int64,2},
        points::AbstractArray{Float64,2},
        local_clusters::Vector{local_cluster},
        weights::Vector{Float64},
        final::Bool,
        group_num::Int64)
    parr = zeros(length(labels), length(local_clusters))
    for (k,v) in enumerate(local_clusters)
        get_local_cluster_likelihood!(reshape((@view parr[:,k]),:,1),points,v, group_num)
    end
    for (k,v) in enumerate(weights)
        parr[:,k] .+= log(v)
    end
    if final
        labels .= mapslices(argmax, parr, dims= [2])
    else
        sample_log_cat_array!(labels,parr)
    end
end


function update_local_cluster_params!(cluster::local_cluster,
        points::AbstractArray{Float64,2},
        sub_labels::AbstractArray{Int64,1})

    splittable_cluster = cluster.cluster_params
    cpl = splittable_cluster.cluster_params_l
    cpr = splittable_cluster.cluster_params_r
    cp = splittable_cluster.cluster_params
    local_dim = cluster.local_dim
    gc = cluster.globalCluster
    gc_params = global_clusters_vector[gc].cluster_params


    # gl_suff_statistics = create_sufficient_statistics(gc_params.cluster_params_l.hyperparams,
    #     gc_params.cluster_params_l.posterior_hyperparams,
    #     (@view points[1:local_dim-1,(@view (sub_labels .<= 2)[:])]),
    #     ones(sum((@view (sub_labels .<= 2)[:]))))
    # gr_suff_statistics = create_sufficient_statistics(gc_params.cluster_params_r.hyperparams,
    #     gc_params.cluster_params_r.posterior_hyperparams,
    #     (@view points[1:local_dim-1,(@view (sub_labels .> 2)[:])]),
    #     nes(sum((@view (sub_labels .> 2)[:]))))

    cpl.suff_statistics = create_sufficient_statistics(cpl.hyperparams, cpl.posterior_hyperparams,@view points[local_dim: end,sub_labels .% 2 .== 1])
    cpr.suff_statistics = create_sufficient_statistics(cpr.hyperparams, cpr.posterior_hyperparams,@view points[local_dim: end,sub_labels .% 2 .== 0])
    l_count = sum(sub_labels .<= 2)
    r_count = sum(sub_labels .> 2)
    cp.suff_statistics = aggregate_suff_stats(cpl.suff_statistics, cpr.suff_statistics)
    cp.posterior_hyperparams = calc_posterior(cp.hyperparams, cp.suff_statistics)
    cpl.posterior_hyperparams = calc_posterior(cpl.hyperparams, cpl.suff_statistics)
    cpr.posterior_hyperparams = calc_posterior(cpr.hyperparams, cpr.suff_statistics)
    cluster.global_suff_stats = [l_count,r_count]
    cluster.cluster_params = splittable_cluster
    # cluster.global_suff_stats = [gl_suff_statistics, gr_suff_statistics]
end




function update_suff_stats_posterior!(group::local_group)
    local_dim = group.model_hyperparams.local_dim
    for (index,cluster) in enumerate(group.local_clusters)
        pts = @view group.points[:, (@view (group.labels .== index)[:])]
        sub_labels = @view group.labels_subcluster[group.labels .== index]
        update_local_cluster_params!(cluster,pts, sub_labels)
    end
end

function update_suff_stats_posterior!(group::local_group, clusters::AbstractArray{Int64,1})
    local_dim = group.model_hyperparams.local_dim
    for (index,cluster) in enumerate(group.local_clusters)
        if index in clusters
            pts = @view group.points[1 : end, (@view (group.labels .== index)[:])]
            sub_labels = @view group.labels_subcluster[(@view (group.labels .== index)[:]),:]
            cluster.points_count = size(pts,2)
            if cluster.points_count > 0
                update_splittable_cluster_params!(cluster.cluster_params,
                    pts[local_dim:end,:], (@view sub_labels[:]),false)
            end
        end
    end
end

# function split_cluster!(group::local_group, cluster::local_cluster, index::Int64, new_index::Int64)
#     labels = @view group.labels[group.labels .== index]
#     sub_labels = @view group.labels_subcluster[group.labels .== index]
#     labels[sub_labels .== 2] .= new_index
#     labels[sub_labels .== 3] .= new_index+1
#     labels[sub_labels .== 4] .= new_index+2
#     g_split = copy_local_cluster(cluster)
#     l_split = copy_local_cluster(cluster)
#     lg_split = copy_local_cluster(cluster)
#     l_split.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_r, group.model_hyperparams.α)
#     lg_split.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_r, group.model_hyperparams.α)
#     cluster.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_l, group.model_hyperparams.α)
#     g_split.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_l, group.model_hyperparams.α)
#     l_split.points_count = length(labels[sub_labels .== 2])
#     cluster.points_count = length(labels[sub_labels .== 1])
#     g_split.points_count = length(labels[sub_labels .== 3])
#     lg_split.points_count = length(labels[sub_labels .== 4])
#     sub_labels .= rand(1:4,length(sub_labels))
#     g_split.globalCluster = cluster.globalCluster + length(global_clusters_vector)
#     lg_split.globalCluster = g_split.globalCluster
#     cluster.globalCluster_subcluster = rand(1:2)
#     g_split.globalCluster_subcluster = rand(1:2)
#     lg_split.globalCluster_subcluster = rand(1:2)
#     l_split.globalCluster_subcluster = rand(1:2)
#     # new_sub_labels = @view sub_labels[sub_labels .==2]
#     # sub_labels = @view sub_labels[sub_labels .==1]
#     # if size(sub_labels,1) > 0
#     #     create_subclusters_labels!(reshape(sub_labels,:,1),(@view group.points[group.model_hyperparams.local_dim: end,(@view (group.labels .== index)[:])]), cluster.cluster_params)
#     # end
#     # if size(new_sub_labels,1) > 0
#     #     create_subclusters_labels!(reshape(new_sub_labels,:,1),(@view group.points[group.model_hyperparams.local_dim: end,(@view (group.labels .== new_index)[:])]), new_cluster.cluster_params)
#     # end
#     group.local_clusters[new_index] = l_split
#     group.local_clusters[new_index+1] = g_split
#     group.local_clusters[new_index+2] = lg_split
# end

function split_cluster_local!(group::local_group, cluster::local_cluster, index::Int64, new_index::Int64)
    labels = @view group.labels[group.labels .== index]
    sub_labels = @view group.labels_subcluster[group.labels .== index]
    labels[sub_labels .== 2] .= new_index

    labels[sub_labels .== 4] .= new_index
    l_split = copy_local_cluster(cluster)
    l_split.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_r, group.model_hyperparams.η)
    cluster.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_l, group.model_hyperparams.η)

    l_split.points_count = length(labels[sub_labels .== 2]) + length(labels[sub_labels .== 4])
    cluster.points_count = length(labels[sub_labels .== 1]) + length(labels[sub_labels .== 3])

    sub_labels[(x -> x ==1 || x==2).(sub_labels)] .= rand(1:2,length((@view sub_labels[(x -> x ==1 || x==2).(sub_labels)])))
    sub_labels[(x -> x ==3 || x==4).(sub_labels)] .= rand(3:4,length((@view sub_labels[(x -> x ==3 || x==4).(sub_labels)])))

    group.local_clusters[new_index] = l_split

end

function split_cluster_global!(group::local_group, cluster::local_cluster, index::Int64, new_index::Int64, new_global_index::Int64)
    labels = @view group.labels[group.labels .== index]
    sub_labels = @view group.labels_subcluster[group.labels .== index]
    labels[sub_labels .== 3] .= new_index
    labels[sub_labels .== 4] .= new_index
    g_split = copy_local_cluster(cluster)

    g_split.points_count = length(labels[sub_labels .== 3]) + length(labels[sub_labels .== 4])
    cluster.points_count = length(labels[sub_labels .== 1]) + length(labels[sub_labels .== 2])

    sub_labels[(x -> x ==1 || x==3).(sub_labels)] .= rand(1:2:3,length((@view sub_labels[(x -> x ==1 || x==3).(sub_labels)])))
    sub_labels[(x -> x ==2 || x==4).(sub_labels)] .= rand(2:2:4,length((@view sub_labels[(x -> x ==2 || x==4).(sub_labels)])))
    g_split.globalCluster = new_global_index
    group.local_clusters[new_index] = g_split
end


# function split_cluster!(group::local_group, cluster::local_cluster, index::Int64, new_index::Int64)
#     labels = @view group.labels[group.labels .== index]
#     sub_labels = @view group.labels_subcluster[group.labels .== index]
#     labels[sub_labels .== 2] .= new_index
#     labels[sub_labels .== 3] .= new_index+1
#     labels[sub_labels .== 2] .= new_index+2
#     new_cluster = copy_local_cluster(cluster)
#     new_cluster.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_r, group.model_hyperparams.η)
#     cluster.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_l, group.model_hyperparams.α)
#     new_cluster.points_count = new_cluster.cluster_params.cluster_params.suff_statistics.N
#     cluster.points_count = cluster.cluster_params.cluster_params.suff_statistics.N
#     new_sub_labels = @view sub_labels[sub_labels .==2]
#     sub_labels = @view sub_labels[sub_labels .==1]
#     if size(sub_labels,1) > 0
#         create_subclusters_labels!(reshape(sub_labels,:,1),(@view group.points[group.model_hyperparams.local_dim: end,(@view (group.labels .== index)[:])]), cluster.cluster_params)
#     end
#     if size(new_sub_labels,1) > 0
#         create_subclusters_labels!(reshape(new_sub_labels,:,1),(@view group.points[group.model_hyperparams.local_dim: end,(@view (group.labels .== new_index)[:])]), new_cluster.cluster_params)
#     end
#     group.local_clusters[new_index] = new_cluster
# end

function merge_clusters!(group::local_group,index_l::Int64, index_r::Int64)
    new_splittable_cluster = merge_clusters_to_splittable(group.local_clusters[index_l].cluster_params.cluster_params, group.local_clusters[index_r].cluster_params.cluster_params, group.model_hyperparams.η)
    group.local_clusters[index_l].cluster_params = new_splittable_cluster
    group.local_clusters[index_l].points_count += group.local_clusters[index_r].points_count
    group.local_clusters[index_r].points_count = 0
    group.local_clusters[index_r].cluster_params.cluster_params.suff_statistics.N = 0
    group.local_clusters[index_r].cluster_params.splittable = false
    # println("merging " * string(index_l) * " with " * string(index_r))
    for i=1:size(group.labels_subcluster,1)
        if group.labels[i] == index_l
            if group.labels_subcluster[i] <= 2
                group.labels_subcluster[i] = 1
            else
                group.labels_subcluster[i] = 3
            end
        elseif group.labels[i] == index_r
            if group.labels_subcluster[i] <= 2
                group.labels_subcluster[i] = 2
            else
                group.labels_subcluster[i] = 4
            end
        end
    end
    group.labels[@view (group.labels .== index_r)[:]] .= index_l
end

function should_split_local!(should_split::AbstractArray{Float64,1},
        cluster_params::splittable_cluster_params,global_params::splittable_cluster_params,
        global_subcluster_suff::Vector{sufficient_statistics}, α::Float64, final::Bool)
    cpl = cluster_params.cluster_params_l
    cpr = cluster_params.cluster_params_r
    cp = cluster_params.cluster_params
    cpgl = global_params.cluster_params_l
    cpgr = global_params.cluster_params_r
    cpg = global_params.cluster_params
    # println("bob2")
    if final || cpl.suff_statistics.N == 0 ||cpr.suff_statistics.N == 0
        should_split .= 0
        return
    end
    log_likihood_l = log_marginal_likelihood(cpl.hyperparams,cpl.posterior_hyperparams, cpl.suff_statistics)
    log_likihood_r = log_marginal_likelihood(cpr.hyperparams,cpr.posterior_hyperparams, cpr.suff_statistics)
    log_likihood_gl = log_marginal_likelihood(cpgl.hyperparams,cpgl.posterior_hyperparams, global_subcluster_suff[1])
    log_likihood_gr = log_marginal_likelihood(cpgr.hyperparams,cpgr.posterior_hyperparams, global_subcluster_suff[2])
    log_likihood = log_marginal_likelihood(cp.hyperparams, cp.posterior_hyperparams, cp.suff_statistics)
    log_likihood_g = log_marginal_likelihood(cpg.hyperparams, cpg.posterior_hyperparams, global_subcluster_suff[3])
    log_HR = (log(α) + logabsgamma(cpl.suff_statistics.N + cpgl.suff_statistics.N)[1] + log_likihood_l +
        logabsgamma(cpr.suff_statistics.N + cpgr.suff_statistics.N)[1] + log_likihood_r +
        log_likihood_gl +log_likihood_gr -
        (logabsgamma(cp.suff_statistics.N + cpg.suff_statistics.N)[1] + log_likihood +  log_likihood_g))
    println(log_likihood_l)
    if log_HR > log(rand())
        should_split .= 1
    end
end



function should_split_local!(should_split::AbstractArray{Float64,1},
        cluster_params::splittable_cluster_params, α::Float64, final::Bool, is_zero_dim = false)
    cpl = cluster_params.cluster_params_l
    cpr = cluster_params.cluster_params_r
    cp = cluster_params.cluster_params
    # println("bob")
    if (final || cpl.suff_statistics.N == 0 ||cpr.suff_statistics.N == 0) && is_zero_dim == false
        should_split .= 0
        return
    end
    log_likihood_l = log_marginal_likelihood(cpl.hyperparams,cpl.posterior_hyperparams, cpl.suff_statistics)
    log_likihood_r = log_marginal_likelihood(cpr.hyperparams,cpr.posterior_hyperparams, cpr.suff_statistics)
    log_likihood = log_marginal_likelihood(cp.hyperparams, cp.posterior_hyperparams, cp.suff_statistics)
    if is_zero_dim
        log_likihood_l = 0
        log_likihood_r = 0
        log_likihood = 0
    end
    log_HR = (log(α) + logabsgamma(cpl.suff_statistics.N)[1] + log_likihood_l +
        logabsgamma(cpr.suff_statistics.N)[1] + log_likihood_r -
        (logabsgamma(cp.suff_statistics.N)[1] + log_likihood))

    if log_HR > log(rand())
        should_split .= 1
    end
end

function should_merge!(should_merge::AbstractArray{Float64,1},lr_weights::AbstractArray{Float64, 1}, cpl::cluster_parameters,cpr::cluster_parameters, α::Float64, final::Bool)
    new_suff = aggregate_suff_stats(cpl.suff_statistics, cpr.suff_statistics)
    cp = cluster_parameters(cpl.hyperparams, cpl.distribution, new_suff,cpl.posterior_hyperparams)
    cp.posterior_hyperparams = calc_posterior(cp.hyperparams, cp.suff_statistics)
    log_likihood_l = log_marginal_likelihood(cpl.hyperparams, cpl.posterior_hyperparams, cpl.suff_statistics)
    log_likihood_r = log_marginal_likelihood(cpr.hyperparams, cpr.posterior_hyperparams, cpr.suff_statistics)
    log_likihood = log_marginal_likelihood(cp.hyperparams, cp.posterior_hyperparams, cp.suff_statistics)

    log_HR = -log(α) + logabsgamma(α)[1] -logabsgamma(lr_weights[1]*α)[1] -logabsgamma(lr_weights[2]*α)[1] + logabsgamma(cp.suff_statistics.N)[1] -logabsgamma(cp.suff_statistics.N + α)[1] + logabsgamma(cpl.suff_statistics.N + lr_weights[1]*α)[1]-logabsgamma(cpl.suff_statistics.N)[1]  -
        logabsgamma(cpr.suff_statistics.N)[1] + logabsgamma(cpr.suff_statistics.N + lr_weights[2]*α)[1]+
         log_likihood- log_likihood_l- log_likihood_r
    if (log_HR > log(rand())) || (final && log_HR > log(0.5))
        should_merge .= 1
    end
end

function check_and_split!(group::local_group, final::Bool)
    split_arr= zeros(length(group.local_clusters))

    for (index,cluster) in enumerate(group.local_clusters)
        if cluster.cluster_params.splittable == true

            should_split_local!((@view split_arr[index,:]), cluster.cluster_params,
                group.model_hyperparams.η,final, false)
        end
    end
    new_index = length(group.local_clusters) + 1
    bobob = new_index
    indices = Vector{Int64}()
    resize!(group.local_clusters,Int64(length(group.local_clusters) + sum(split_arr)))
    for i=1:length(split_arr)
        if split_arr[i] == 1
            # push!(indices, new_index)
            # push!(indices, new_index+1)
            # push!(indices, new_index+2)
            # split_cluster!(group, group.local_clusters[i],i,new_index)
            # new_index += 3
            push!(indices, new_index)
            split_cluster_local!(group, group.local_clusters[i],i,new_index)
            new_index += 1
        end
    end
    return indices
end

function check_and_merge!(group::local_group, final::Bool)
    clusters_dict = create_mergable_dict(group)
    indices = Vector{Int64}()

    for (k,v) in clusters_dict
        indices = vcat(indices,check_and_merge!(group, v, final))
    end
    return indices
end

function check_and_merge!(group::local_group, indices::Vector{Int64}, final::Bool)
    mergable = zeros(1)
    ret_indices = Vector{Int64}()
    for i=1:length(indices)
        for j=i+1:length(indices)
            if  (group.local_clusters[indices[i]].cluster_params.splittable == true && group.local_clusters[indices[j]].cluster_params.splittable == true)
                should_merge!(mergable,group.local_clusters[indices[j]].cluster_params.lr_weights, group.local_clusters[indices[i]].cluster_params.cluster_params,
                    group.local_clusters[indices[j]].cluster_params.cluster_params, group.model_hyperparams.η, final)
            end
            if mergable[1] == 1
                merge_clusters!(group, indices[i], indices[j])
                push!(ret_indices, indices[i])
            end
            mergable[1] = 0
        end
    end
    return ret_indices
end


function should_split_global!(should_split::AbstractArray{Float64,1},
            cluster_params::splittable_cluster_params,
            points::AbstractArray{Float64,2},
            sublabels::AbstractArray{Int64,2},
            α::Float64,
            final::Bool,
            group_num::Int64)
    parr = zeros(size(points,2), 1)
    sleft = @view (sublabels .<=2)[:]
    sright = @view (sublabels .>2)[:]
    lcount = sum(sleft)
    rcount = sum(sright)
    if lcount == 0 || rcount == 0
        should_split .= 0
        return
    end
    parr_left = zeros(lcount, 1)
    parr_right = zeros(rcount, 1)
    log_likelihood!(parr[:,1],points,cluster_params.cluster_params.distribution,group_num)
    log_likelihood!(parr_left[:,1],points[:,sleft],cluster_params.cluster_params_l.distribution,group_num)
    log_likelihood!(parr_right[:,1],points[:,sright],cluster_params.cluster_params_r.distribution,group_num)
    # println("sublabels lr_parr: "* string(size(lr_arr)))

    sum_all = sum(parr,dims = 1)[1]
    sum_left = sum(parr_left,dims = 1)[1]
    sum_right = sum(parr_right,dims = 1)[1]

    h_ratio = sum_left + logabsgamma(lcount)[1] + sum_right + logabsgamma(rcount)[1] - sum_all - logabsgamma(lcount + rcount)[1]

    if h_ratio > log(rand())
        # println("hratio: " * string(h_ratio))
        should_split .= 1
    end
end


function check_and_split_global!(group::local_group, final::Bool)
    split_arr= zeros(length(group.local_clusters))
    for (index,cluster) in enumerate(group.local_clusters)
        if cluster.cluster_params.splittable == true
            # should_split_local!((@view split_arr[index,:]), cluster.cluster_params,
            #     global_clusters_vector[cluster.globalCluster].cluster_params,
            #     cluster.global_subcluster_suff, group.model_hyperparams.α,final)
            should_split_global!((@view split_arr[index,:]),
                global_clusters_vector[cluster.globalCluster].cluster_params,
                group.points[1:cluster.local_dim-1,(@view (group.labels .== index)[:])],
                (@view group.labels_subcluster[(@view (group.labels .== index)[:]),:]),
                group.model_hyperparams.α,final, group.group_num)
            # split_arr[index,:] .= 1
            #break #This ensures 1 split per iteration
        end
    end
    new_index = length(group.local_clusters) + 1
    global_count = length(global_clusters_vector)
    indices = Vector{Int64}()
    resize!(group.local_clusters,Int64(length(group.local_clusters) + sum(split_arr)))

    for i=1:length(split_arr)
        if split_arr[i] == 1
            # push!(indices, new_index)
            # push!(indices, new_index+1)
            # push!(indices, new_index+2)
            # split_cluster!(group, group.local_clusters[i],i,new_index)
            # new_index += 3
            push!(indices, new_index)
            # println("new g cluster:" * string(group.local_clusters[i].globalCluster + global_count))
            split_cluster_global!(group, group.local_clusters[i],i,new_index,group.local_clusters[i].globalCluster + global_count)
            new_index += 1
            println("Global single split")
        end
    end
    return indices
end


function create_mergable_dict(group::local_group)
    clusters_dict = Dict()
    for (index,cluster) in enumerate(group.local_clusters)
        if haskey(clusters_dict,cluster.globalCluster)# && cluster.cluster_params.splittable == true
            push!(clusters_dict[cluster.globalCluster], index)
        else
            clusters_dict[cluster.globalCluster] = [index]
        end
    end
    return clusters_dict
end



function sample_clusters!(group::local_group)
    points_count = Vector{Float64}()
    for cluster in group.local_clusters
        push!(points_count, sample_cluster_params!(cluster.cluster_params, group.model_hyperparams.α, true))
    end
    push!(points_count, group.model_hyperparams.α)
    # println(points_count)
    group.weights = rand(Dirichlet(points_count))[1:end-1]
end

function remove_empty_clusters!(group::local_group)
    new_vec = Vector{local_cluster}()
    removed = 0
    for (index,cluster) in enumerate(group.local_clusters)
        if cluster.cluster_params.cluster_params.suff_statistics.N == 0
            # println("test" * string(index))
            group.labels[group.labels .> index - removed] .-= 1
            removed += 1
        else
            push!(new_vec,cluster)
        end
    end
    group.local_clusters = new_vec
end

function split_global_cluster!(group::local_group,global_cluster::Int64, new_global_index::Int64)
    clusters_count = 0
    for cluster in group.local_clusters
        if cluster.globalCluster == global_cluster
            clusters_count += 1
        end
    end
    new_index = length(group.local_clusters) + 1
    resize!(group.local_clusters,Int64(length(group.local_clusters) + clusters_count))
    for (i,cluster) in enumerate(group.local_clusters)
        if cluster.globalCluster == global_cluster
            split_cluster_global!(group, cluster, i,new_index,new_global_index)
            new_index += 1
        end
    end
end


function merge_global_cluster!(group::local_group,global_cluster::Int64, merged_index::Int64)
    clusters_count = 0
    for cluster in group.local_clusters
        if cluster.globalCluster == global_cluster
            clusters_count += 1
        end
    end
    for (i,cluster) in enumerate(group.local_clusters)
        if cluster.globalCluster == global_cluster
            sublabels = @view group.labels_subcluster[group.labels .== i]
            sublabels[(x -> x ==3 || x==4).(sublabels)] .-= 2
        elseif cluster.globalCluster == merged_index
            sublabels = @view group.labels_subcluster[group.labels .== i]
            sublabels[(x -> x ==1 || x==2).(sublabels)] .+= 2
            cluster.globalCluster = global_cluster
        end
    end
end


function group_step(group_num::Number, local_clusters::Vector{local_cluster}, final::Bool)
    group = groups_dict[group_num]
    group.local_clusters = local_clusters

    sample_clusters!(group)
    sample_labels!(group, (hard_clustering ? true : final))
    # sample_sub_clusters!(group, (hard_clustering ? true : final))
    sample_sub_clusters!(group, false)
    update_suff_stats_posterior!(group)
    remove_empty_clusters!(group)
    if final == false && ignore_local == false
        check_and_split!(group, final)
        indices = check_and_merge!(group, final)
        # check_and_split_global!(group, final)
        # if length(indices) > 0
        #     println(indices)
        # end
    end
    remove_empty_clusters!(group)
    return local_group_stats(group.labels, group.labels_subcluster, group.local_clusters)
end
