function create_first_global_cluster(hyperparams::model_hyper_params, groups_dict::Dict, cluster_index::Int64)
    pts = Vector{AbstractArray{Float64,2}}()
    sub_labels = Vector{AbstractArray{Int64,2}}()
    pts_to_groups = Vector{AbstractArray{Int64,1}}()
    count = 0
    for (k,v) in groups_dict
        push!(pts,(@view v.points[1:hyperparams.local_dim-1,:]))
        push!(sub_labels,v.labels_subcluster)
        push!(pts_to_groups,ones(Int64,(size(pts[end],2)))*k)
        for c in v.local_clusters
            if c.globalCluster == cluster_index
                count +=1
            end
        end
    end
    suff = create_sufficient_statistics(hyperparams.global_hyper_params, [])
    # suff = nothing
    post = hyperparams.global_hyper_params
    dist = sample_distribution(post)
    cp = cluster_parameters(hyperparams.global_hyper_params, dist, suff, post)
    cpl = deepcopy(cp)
    cpr = deepcopy(cp)
    splittable = splittable_cluster_params(cp,cpl,cpr,[0.5,0.5], false, ones(20).*-Inf)
    all_pts = reduce(hcat,pts)
    all_sub_labels = reduce(vcat,sub_labels)[:]
    all_pts_to_group = reduce(vcat,pts_to_groups)[:]
    update_splittable_cluster_params!(splittable,all_pts[1:hyperparams.local_dim-1,:],
        all_sub_labels,
         true, all_pts_to_group)
    # println(splittable)
    # println(count)
    cluster = global_cluster(splittable, hyperparams.total_dim, hyperparams.local_dim,splittable.cluster_params.suff_statistics.N,count,[1,1])
    return cluster
end

function get_p_for_point(x)
    x = log.(exp.(x .- maximum(x)) ./ exp(sum(x .- maximum(x))))
    x ./= sum(x)
    return log.(x)
end


function sample_group_cluster_labels(group_num::Int64, weights::AbstractArray{Float64, 1},final::Bool)
    group = groups_dict[group_num]
    points = group.points
    parr = zeros(length(group.labels), length(global_clusters_vector))
    for (k,v) in enumerate(global_clusters_vector)
        local_dim = v.local_dim
        log_likelihood!((@view parr[:,k]), points[1 : local_dim-1,:],v.cluster_params.cluster_params.distribution, group_num)
    end
    clusters_parr = zeros(length(group.local_clusters),length(global_clusters_vector))
    labels = zeros(Int64,length(group.local_clusters),1)
    for (i,c) in enumerate(group.local_clusters)
        relevant_arr = parr[(@view group.labels[:]) .== i,:]
        clusters_parr[i,:] .= sum(relevant_arr, dims = 1)[:] + log.(weights)
        sum_arr_trick = exp.(clusters_parr[i,:])
        testarr = clusters_parr[i,:] .- maximum(clusters_parr[i,:])
        testarr = exp.(testarr)
        if final
            labels[i,1] = argmax(clusters_parr[i,:][:])
        else
            labels[i,1] = sample(1:length(global_clusters_vector), ProbabilityWeights(testarr[:]))
        end
    end
    return labels
end



function sample_cluster_label(group::local_group, cluster::local_cluster ,i, weights::AbstractArray{Float64, 1},final::Bool)
    points = group.points[1 : cluster.local_dim - 1, @view (group.labels .== i)[:]]
    parr = zeros(size(points,2), length(global_clusters_vector))
    for (k,v) in enumerate(global_clusters_vector)
        log_likelihood!((@view parr[:,k]),points,v.cluster_params.cluster_params.distribution,group.group_num)
    end
    weights = reshape(weights,1,:)
    sum_arr = sum(parr,dims = 1)
    sum_arr .+= log.(weights)
    sum_arr_trick = exp.(sum_arr)
    testarr = sum_arr .- maximum(sum_arr)
    testarr = exp.(testarr)


    if final
        cluster.globalCluster = argmax(sum_arr[:])
    else
        cluster.globalCluster = sample(1:length(global_clusters_vector), ProbabilityWeights(testarr[:]))
    end
    # println("sum arr: " *string(sum_arr) * " parr:" * string(sum_arr_trick) * " testarr:" * string(testarr) * " choosen: " * string(cluster.globalCluster))
    return cluster.globalCluster
end

function sample_clusters_labels!(model::hdp_shared_features, final::Bool)
    labels_dict = Dict()
    groups_count = zeros(length(global_clusters_vector))
    wvector = model.weights
    @sync for (k,v) in model.groups_dict
        @async labels_dict[k] = @spawnat ((k % nworkers())+2) sample_group_cluster_labels(k, wvector, final)
        # labels_dict[k] = Dict()
        # for (i,c) in enumerate(v.local_clusters)
        #     # labels_dict[k][i] = @spawn sample_cluster_label(c, v.points[1 : v.model_hyperparams.local_dim - 1, (@view (v.labels .== i)[:])], model.weights, final)
        #     labels_dict[k][i] = sample_cluster_label(v,c, i, model.weights, final)
        # end
    end
    #at ((k % num_of_workers)+1)
    for (k,v) in model.groups_dict
        fetched_labels = fetch(labels_dict[k])
        for (i,c) in enumerate(v.local_clusters)
            c.globalCluster = fetched_labels[i,1]
            groups_count[c.globalCluster] += 1
        end
    end
    for (i,v) in enumerate(groups_count)
        global_clusters_vector[i].clusters_count = v
    end
end

function sample_sub_clusters!(model::hdp_shared_features)
    labels_dict = Dict()
    for v in global_clusters_vector
        v.clusters_sub_counts = [0,0]
    end
    for (k,v) in model.groups_dict
        labels_dict[k] = Dict()
        for (i,c) in enumerate(v.local_clusters)
            labels_dict[k][i] = @spawnat ((k % num_of_workers)+1) sample_cluster_sub_label(c, v.points[1 : v.model_hyperparams.local_dim - 1, (@view (v.labels .== i)[:])])
        end
    end
    for (k,v) in model.groups_dict
        for (i,c) in enumerate(v.local_clusters)
            c.globalCluster_subcluster = fetch(labels_dict[k][i])
            global_clusters_vector[c.globalCluster].clusters_sub_counts[c.globalCluster_subcluster] += 1
        end
    end
end


function split_cluster!(model::hdp_shared_features, index::Int64, new_index::Int64)
    cluster = global_clusters_vector[index]
    new_cluster = deepcopy(cluster)
    new_cluster.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_r, model.model_hyperparams.γ)
    cluster.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_l, model.model_hyperparams.γ)
    new_cluster.points_count = new_cluster.cluster_params.cluster_params.suff_statistics.N
    cluster.points_count = cluster.cluster_params.cluster_params.suff_statistics.N
    global_clusters_vector[new_index] = new_cluster
end


function merge_clusters!(model::hdp_shared_features,index_l::Int64, index_r::Int64)
    new_splittable_cluster = merge_clusters_to_splittable(global_clusters_vector[index_l].cluster_params.cluster_params, global_clusters_vector[index_r].cluster_params.cluster_params, model.model_hyperparams.α)
    global_clusters_vector[index_l].cluster_params = new_splittable_cluster
    global_clusters_vector[index_l].clusters_count += global_clusters_vector[index_r].clusters_count
    global_clusters_vector[index_l].cluster_params.splittable = true
    global_clusters_vector[index_r].cluster_params.cluster_params.suff_statistics.N = 0
    global_clusters_vector[index_r].cluster_params.splittable = true
    global_clusters_vector[index_r].clusters_count = 0
end

function should_split!(should_split::AbstractArray{Float64,1},
        cluster_params::splittable_cluster_params,
        groups_dict::Dict,
        α::Float64,
        γ::Float64,
        c_count::Int64,
        lr_count::AbstractArray{Int64,1},
        index::Int64,
        glob_weight::Float64,
        final::Bool)
    cpl = cluster_params.cluster_params_l
    cpr = cluster_params.cluster_params_r
    cp = cluster_params.cluster_params
    if final || cpl.suff_statistics.N == 0 ||cpr.suff_statistics.N == 0 #||lr_count[1] == 0 || lr_count[2] == 0
        should_split .= 0
        return
    end
    log_likihood_l = log_marginal_likelihood(cpl.hyperparams, cpl. posterior_hyperparams, cpl.suff_statistics)
    log_likihood_r = log_marginal_likelihood(cpr.hyperparams, cpr. posterior_hyperparams, cpr.suff_statistics)
    log_likihood = log_marginal_likelihood(cp.hyperparams, cp. posterior_hyperparams, cp.suff_statistics)


    log_HR = log(γ) + logabsgamma(cpl.suff_statistics.N)[1] + log_likihood_l + logabsgamma(cpr.suff_statistics.N)[1] + log_likihood_r -(logabsgamma(cp.suff_statistics.N)[1] + log_likihood) +
        cp.suff_statistics.N*log(glob_weight)-cpl.suff_statistics.N*log(glob_weight*cluster_params.lr_weights[1])-cpr.suff_statistics.N*log(glob_weight*cluster_params.lr_weights[2])
    log_HR += get_groups_split_log_likelihood(groups_dict,
        index,
        cluster_params.lr_weights[1],
        cluster_params.lr_weights[2],
        glob_weight,
        α)

    if log_HR > log(rand())
        should_split .= 1
    end
end

function should_merge!(should_merge::AbstractArray{Float64,1},
        cpl::cluster_parameters,
        cpr::cluster_parameters,
        groups_dict::Dict,
        α::Float64,
        c1_count::Int64,
        c2_count::Int64,
        i::Int64,
        j::Int64,
        wi::Float64,
        wj::Float64,
        final::Bool)
    new_suff = aggregate_suff_stats(cpl.suff_statistics, cpr.suff_statistics)
    cp = cluster_parameters(cpl.hyperparams, cpl.distribution, new_suff,cpl.posterior_hyperparams)
    cp.posterior_hyperparams = calc_posterior(cp.hyperparams, cp.suff_statistics)
    log_likihood_l = log_marginal_likelihood(cpl.hyperparams, cpl.posterior_hyperparams, cpl.suff_statistics)
    log_likihood_r = log_marginal_likelihood(cpr.hyperparams, cpr.posterior_hyperparams, cpr.suff_statistics)
    log_likihood = log_marginal_likelihood(cp.hyperparams, cp.posterior_hyperparams, cp.suff_statistics)
    log_HR = -log(α) + logabsgamma(α)[1] -logabsgamma(wi*α)[1] -logabsgamma(wj*α)[1] + logabsgamma(cp.suff_statistics.N)[1] -logabsgamma(cp.suff_statistics.N + α)[1] + logabsgamma(cpl.suff_statistics.N + wi*α)[1]-logabsgamma(cpl.suff_statistics.N)[1]  -
        logabsgamma(cpr.suff_statistics.N)[1] + logabsgamma(cpr.suff_statistics.N + wj*α)[1]+
         log_likihood- log_likihood_l- log_likihood_r

    if (log_HR > log(rand())) || (final && log_HR > log(0.5))
        should_merge .= 1
    end
end


function get_groups_split_log_likelihood(groups_dict::Dict,
        global_cluster_index::Int64,
        lweight::Float64,
        rweight::Float64,
        glob_weight::Float64,
        γ::Float64)
    total_likelihood = 0.0
    for (k,group) in groups_dict
        lcount = 0.0
        rcount = 0.0
        for c in group.local_clusters
            if c.globalCluster == global_cluster_index
                lcount = c.global_suff_stats[1]
                rcount = c.global_suff_stats[2]
                total_likelihood += (logabsgamma(γ * glob_weight)[1] - logabsgamma(γ * glob_weight + lcount+ rcount)[1] +
                    logabsgamma(γ * glob_weight * lweight + lcount)[1] - logabsgamma(γ * glob_weight * lweight)[1] +
                    logabsgamma(γ * glob_weight * rweight + rcount)[1] - logabsgamma(γ * glob_weight * rweight)[1])
            end
        end

    end
    # println(total_likelihood)
    return total_likelihood
end


function get_groups_merge_log_likelihood(groups_dict::Dict,
        global_cluster_i::Int64,
        global_cluster_j::Int64,
        weight_i::Float64,
        weight_j::Float64,
        γ::Float64)
    total_likelihood = 0.0
    for (k,group) in groups_dict
        lcount = 0.0
        rcount = 0.0
        for c in group.local_clusters
            if c.globalCluster == global_cluster_i
                lcount += c.cluster_params.cluster_params_l.suff_statistics.N
            end
            if c.globalCluster == global_cluster_j
                rcount += c.cluster_params.cluster_params_r.suff_statistics.N
            end
        end
        total_likelihood += logabsgamma(γ * (weight_i+weight_j))[1] - logabsgamma(γ * (weight_i+weight_j)[1] + lcount+ rcount)[1] +
            logabsgamma(γ * weight_i + lcount)[1] - logabsgamma(γ * weight_i)[1] +
            logabsgamma(γ * weight_j + rcount)[1] - logabsgamma(γ * weight_j)[1]
    end
    return -total_likelihood
end


function check_and_split!(model::hdp_shared_features, final::Bool)
    split_arr= zeros(length(global_clusters_vector))
    for (index,cluster) in enumerate(global_clusters_vector)
        # println("index: " * string(index) * " splittable: " * string(cluster.cluster_params.splittable))
        if cluster.cluster_params.splittable == true
            should_split!((@view split_arr[index,:]),
                cluster.cluster_params,
                model.groups_dict,
                model.model_hyperparams.α,
                model.model_hyperparams.γ,
                cluster.clusters_count,
                cluster.clusters_sub_counts,
                index,
                model.weights[index],
                final)
            if  split_arr[index,:] == 1
                break
            end
        end
    end
    new_index = length(global_clusters_vector) + 1
    resize!(global_clusters_vector,Int64(length(global_clusters_vector) + sum(split_arr)))
    indices = Vector{Int64}()
    for i=1:length(split_arr)
        if split_arr[i,1] == 1
            split_cluster!(model,i, new_index)
            push!(indices,i)
            push!(indices,new_index)
            new_index += 1
        end
    end
    return indices
end


function check_and_merge!(model::hdp_shared_features, final::Bool)
    mergable = zeros(1)
    indices = Vector{Int64}()
    distance_matrix = zeros(length(global_clusters_vector),length(global_clusters_vector))
    if (@isdefined use_mean_for_merge) && use_mean_for_merge == true
        for i=1:length(global_clusters_vector)-1
            for j=i+1:length(global_clusters_vector)
                distance_matrix[i,j] = norm(global_clusters_vector[i].cluster_params.cluster_params.distribution.μ - global_clusters_vector[j].cluster_params.cluster_params.distribution.μ)
            end
            indice_to_check = argmin(distance_matrix[i,i+1:end])
            if (global_clusters_vector[i].cluster_params.splittable == true && global_clusters_vector[indice_to_check].cluster_params.splittable == true) || final
                should_merge!(mergable, global_clusters_vector[i].cluster_params.cluster_params,
                    global_clusters_vector[indice_to_check].cluster_params.cluster_params,model.groups_dict,
                    model.model_hyperparams.γ,global_clusters_vector[i].clusters_count,
                    global_clusters_vector[indice_to_check].clusters_count,i,indice_to_check,model.weights[i], model.weights[indice_to_check], final)
            end
            if mergable[1] == 1
                merge_clusters!(model, i, indice_to_check)
                push!(indices,i)
                push!(indices,indice_to_check)
                break
            end
        end
    else
        for i=1:length(global_clusters_vector)
            for j=i+1:length(global_clusters_vector)
                if (global_clusters_vector[i].cluster_params.splittable == true && global_clusters_vector[j].cluster_params.splittable == true) || final
                    should_merge!(mergable, global_clusters_vector[i].cluster_params.cluster_params,
                        global_clusters_vector[j].cluster_params.cluster_params,model.groups_dict,
                        model.model_hyperparams.γ,global_clusters_vector[i].clusters_count,
                        global_clusters_vector[j].clusters_count,i,j,model.weights[i], model.weights[j], final)
                end
                if mergable[1] == 1
                    merge_clusters!(model, i, j)
                    push!(indices,i)
                    push!(indices,j)
                    break
                end
            end
            if mergable[1] == 1
                break
            end
        end
    end
    return indices
end


function update_suff_stats_posterior!(model::hdp_shared_features, indices::AbstractArray{Int64,1})
    local_dim = model.model_hyperparams.local_dim
    pts_vector_dict = Dict()
    sub_labels_vector_dict = Dict()
    clusters_count_dict = Dict()
    pts_to_groups = Dict()
    for i=1:length(global_clusters_vector)
        if i in indices
            pts_vector_dict[i] = Vector{AbstractArray{Float64,2}}()
            sub_labels_vector_dict[i] = Vector{AbstractArray{Int64,1}}()
            pts_to_groups[i] = Vector{AbstractArray{Int64,1}}()
            clusters_count_dict[i] = 0
        end
    end
    for (k,v) in model.groups_dict
        for (i,c) in enumerate(v.local_clusters)
            if c.globalCluster in indices
                push!(pts_vector_dict[c.globalCluster], (@view v.points[1:local_dim-1,(@view (v.labels .== i)[:])]))
                sub_labels = (@view v.labels_subcluster[(@view (v.labels .== i)[:])])
                push!(sub_labels_vector_dict[c.globalCluster], sub_labels)
                push!(pts_to_groups[c.globalCluster],ones(Int64,size(sub_labels,1))*k)
            end
        end
    end
    cluster_params_dict = Dict()
    begin
         @sync for (index,cluster) in enumerate(global_clusters_vector)
            if index in indices
                if size(pts_vector_dict[index],1) > 0
                    #pts = reshape(CatView(Tuple(pts_vector_dict[index])),local_dim - 1,:)
                    cluster.clusters_sub_counts = [0,0]
                    pts = reduce(hcat,pts_vector_dict[index])
                    for sublabel in sub_labels_vector_dict[index]
                        if sum(sublabel .<= 2) >= sum(sublabel .> 2)
                            cluster.clusters_sub_counts[1] += 1
                        else
                            cluster.clusters_sub_counts[2] += 1
                        end
                    end
                    sublabels = reduce(vcat,sub_labels_vector_dict[index])
                    pts_group= reduce(vcat,pts_to_groups[index])
                    #sublabels = CatView(Tuple(sub_labels_vector_dict[index]))
                    # println(cluster.clusters_sub_counts)
                    cluster_params_dict[index] =  update_splittable_cluster_params(cluster.cluster_params, pts , sublabels, true, pts_group)
                end
            end
        end
        for (index,cluster) in enumerate(global_clusters_vector)
            if index in indices
                if size(pts_vector_dict[index],1) > 0
                    cluster.cluster_params = fetch(cluster_params_dict[index])
                    #println(cluster.cluster_params)
                end
            end
        end
    end
end

function sample_global_clusters_params!(model::hdp_shared_features)
    points_count = Vector{Float64}()
    for cluster in global_clusters_vector
        #push!(points_count, cluster.clusters_count)
        push!(points_count, cluster.cluster_params.cluster_params.suff_statistics.N)
        sample_cluster_params!(cluster.cluster_params, model.model_hyperparams.γ, cluster.clusters_sub_counts)
    end
    push!(points_count, model.model_hyperparams.γ)
    model.weights = rand(Dirichlet(points_count))[1:end-1]
    # println("Weights:" * string(model.weights))
    # println("Samples: " * string([x.cluster_params.cluster_params.distribution for x in global_clusters_vector]))
end


function remove_empty_clusters!(model::hdp_shared_features)
    new_vec = Vector{global_cluster}()
    to_remove = []
    for (index,cluster) in enumerate(global_clusters_vector)
        if cluster.clusters_count == 0
            push!(to_remove, index)
        else
            push!(new_vec,cluster)
        end
    end
    if length(to_remove) > 0
        for (k,v) in model.groups_dict
            for (i,c) in enumerate(v.local_clusters)
                c.globalCluster -= sum(to_remove .< c.globalCluster)
            end
        end
        global global_clusters_vector = new_vec
    end
end

function local_split!(model::hdp_shared_features, index::Int64, new_index::Int64)
    cluster = global_clusters_vector[index]
    new_cluster = deepcopy(cluster)
    new_cluster.clusters_count = 0
    new_cluster.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_r, model.model_hyperparams.γ)
    cluster.cluster_params = create_splittable_from_params(cluster.cluster_params.cluster_params_l, model.model_hyperparams.γ)
    new_cluster.points_count = new_cluster.cluster_params.cluster_params.suff_statistics.N
    cluster.points_count = cluster.cluster_params.cluster_params.suff_statistics.N
    global_clusters_vector[new_index] = new_cluster
end


function split_new_clusters_from_local!(model::hdp_shared_features)
    cur_count = length(global_clusters_vector)
    new_clusters_to_global = Dict() #keys are global clusters to split
    for (i,group) in model.groups_dict
        for c in group.local_clusters
            new_global = c.globalCluster
            if new_global > cur_count #new cluster
                # println(new_global)
                if haskey(new_clusters_to_global, new_global - cur_count)
                    push!(new_clusters_to_global[new_global - cur_count], c)                     #The last2 clusters are in the new global
                else
                    new_clusters_to_global[new_global - cur_count] = [c]
                end
            end
        end
    end
    new_index =  Int64(length(global_clusters_vector)) +1
    # println(keys(new_clusters_to_global))
    resize!(global_clusters_vector,Int64(length(global_clusters_vector) + length(new_clusters_to_global)))
    indicies_to_re_evaluate = Vector{Int64}()
    for (k,v) in new_clusters_to_global
        local_split!(model,k,new_index)
        # global_clusters_vector[k].clusters_count += length(v) / 2
        # global_clusters_vector[new_index].clusters_count += length(v)
        for group in v
            group.globalCluster = new_index
        end
        push!(indicies_to_re_evaluate,k)
        push!(indicies_to_re_evaluate,new_index)
        new_index += 1
    end
    if length(indicies_to_re_evaluate) > 0
        update_pts_count!(model)
        remove_empty_clusters!(model)
        update_weights!(model)
    end
    return indicies_to_re_evaluate
end


function update_pts_count!(model::hdp_shared_features)
    counts = zeros(length(global_clusters_vector))
    for (k,v) in model.groups_dict
        for c in v.local_clusters
            counts[c.globalCluster] += 1
        end
    end
    # println(counts)
    for (k,v) in enumerate(global_clusters_vector)
        v.clusters_count = counts[k]
    end
end

function update_weights!(model::hdp_shared_features)
    counts = Vector{Float64}()
    for c in global_clusters_vector
        push!(counts,c.clusters_count)
    end
    push!(counts, model.model_hyperparams.γ)
    model.weights = rand(Dirichlet(counts))[1:end-1]
end

function update_partial_pts_count!(model::hdp_shared_features)
    counts = zeros(length(global_clusters_vector))
    for (k,v) in model.groups_dict
        for c in v.local_clusters
            if c.globalCluster < length(counts)
                counts[c.globalCluster] += 1
            end
        end
    end
    for (k,v) in enumerate(global_clusters_vector)
        v.clusters_count = counts[k]
    end
end
