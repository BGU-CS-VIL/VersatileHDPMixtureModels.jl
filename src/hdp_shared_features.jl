function init_model(swap_axes; data = nothing , model_params = nothing)
    if random_seed != nothing
        @eval @everywhere Random.seed!($random_seed)
    end
    global lm
    if data == nothing
        points_dict = load_data(data_path, groups_count, prefix = data_prefix)
        if swap_axes != nothing
            points_dict = axes_swapper(points_dict,swap_axes)
        end
        preprocessing!(points_dict,local_dim,global_preprocessing,local_preprocessing)
    else
        points_dict = data
    end

    if model_params == nothing
        model_hyperparams = model_hyper_params(global_hyper_params,local_hyper_params,α,γ,η,global_weight,local_weight,total_dim,local_dim)
    else
        model_hyperparams = model_params
    end
    blocal_hyper_params = model_hyperparams.local_hyper_params
    groups_dict = Dict()
    for (k,v) in points_dict
        labels = rand(1:initial_local_clusters,(size(v,2),1))
        labels_subcluster = rand(1:4,(size(v,2),1))
        weights = ones(initial_local_clusters) * (1/initial_local_clusters)
        local_clusters = local_cluster[]
        if isa(blocal_hyper_params,Array)
            localised_params = model_hyper_params(model_hyperparams.global_hyper_params
            ,blocal_hyper_params[k],model_hyperparams.α,
            model_hyperparams.γ,model_hyperparams.η,model_hyperparams.global_weight,
            model_hyperparams.local_weight,model_hyperparams.total_dim,model_hyperparams.local_dim)
        else
            localised_params = model_hyperparams
        end
        groups_dict[k] = local_group(localised_params,v,labels,labels_subcluster,local_clusters,Float64[],k)
    end
    if isa(model_hyperparams.global_hyper_params,topic_modeling_hyper)
        global is_tp = true
    else
        global is_tp = false
    end

    @eval global groups_dict = $groups_dict
    if mp
        num_of_workers = nworkers()
        for w in workers()
            @spawnat w global groups_dict = Dict()
        end

        @sync for (index,group) in groups_dict
            @spawnat ((index % num_of_workers)+2) set_group(index,group)
        end
    end
    return hdp_shared_features(model_hyperparams,groups_dict,global_cluster[],Float64[])
end


function init_first_clusters!(hdp_model::hdp_shared_features)
    for (k,v) in hdp_model.groups_dict
        v.local_clusters = []
        for i=1:initial_local_clusters
            push!(v.local_clusters, create_first_local_cluster(v, initial_global_clusters))
        end
    end
    global_c = []
    for i=1:initial_global_clusters
        push!(global_c,create_first_global_cluster(hdp_model.model_hyperparams, hdp_model.groups_dict, i))
    end
    global global_clusters_vector  = global_c
    for w in workers()
        @eval @spawnat $w global global_clusters_vector = $global_c
    end
    # @eval @everywhere global global_clusters_vector = $global_c
end


function hdp_shared_features(model_params, swap_axes = nothing;multiprocess = false)
    cur_dir = pwd()
    include(model_params)
    cd(cur_dir)
    hdp_model = init_model(swap_axes)
    init_first_clusters!(hdp_model)
    groups_stats = Dict()
    global num_of_workers = nworkers()
    if @isdefined split_delays
        @everywhere global split_delays = split_delays
    else
        @everywhere global split_delays = true
    end
    @everywhere global hard_clustering = hard_clustering
    global posterior_history = []
    global word_ll_history = []
    global topic_count = []
    global mp = multiprocess
    for i=1:iterations
        println("Iteration: " * string(i))
        println("Global Counts: " * string([x.clusters_count for x in global_clusters_vector]))
        final = false
        no_more_splits = false
        if i > iterations - argmax_sample_stop #We assume the cluters k has been setteled by now, and a low probability random split can do dmg
            final = true
        end
        if i >= iterations - split_stop
            no_more_splits = true
        end
        model_iteration(hdp_model,final,no_more_splits)

    end
    hdp_model.global_clusters = global_clusters_vector
    return hdp_model, posterior_history, word_ll_history, topic_count
end


function model_iteration(hdp_model,final, no_more_splits,burnout = 5)
    groups_stats = Vector{local_group_stats}(undef,length(groups_dict))
    @everywhere global burnout_period = 5
    sample_global_clusters_params!(hdp_model)
    global global_clusters_vector = global_clusters_vector
    refs= Dict()
    if mp
        for w in workers()
            refs[w] = remotecall(set_global_clusters_vector, w, global_clusters_vector)
        end
        for w in workers()
            fetch(refs[w])
        end
    end
    begin
        if mp
            @sync for (index,group) in hdp_model.groups_dict
                lc = group.local_clusters
                groups_stats[index] = @spawnat ((index % num_of_workers)+2) group_step(index,lc, final)
            end
        else
            Threads.@threads for index=1:length(groups_dict)
                group=groups_dict[index]
                lc = group.local_clusters
                groups_stats[index] = group_step(index,lc, final)
            end
        end
        for index=1:length(groups_dict)
            update_group_from_stats!(hdp_model.groups_dict[index], fetch(groups_stats[index]))
        end
    end
    sample_clusters_labels!(hdp_model, (hard_clustering ? true : final))
    remove_empty_clusters!(hdp_model)
    update_suff_stats_posterior!(hdp_model,collect(1:length(global_clusters_vector)))
    hdp_model.global_clusters = global_clusters_vector
    push!(posterior_history,calc_global_posterior(hdp_model))
    # if isa(hdp_model.model_hyperparams.global_hyper_params, topic_modeling_hyper)
    #     word_ll = calc_avg_word(hdp_model)
    #     println("Per Word LL:" * string(word_ll))
    #     push!(word_ll_history,word_ll)
    #     push!(topic_count,length(global_clusters_vector))
    # end
    if no_more_splits == false
        # println(length((global_clusters_vector)))
        indices = check_and_split!(hdp_model, final)
        i = 1
        while i < length(indices)
            for (index,group) in hdp_model.groups_dict
                split_global_cluster!(group,indices[i],indices[i+1])
            end
            i+= 2
        end
        # println(length((global_clusters_vector)))
        indices = check_and_merge!(hdp_model, final)
        if length(indices) > 0
            for (index,group) in hdp_model.groups_dict
                merge_global_cluster!(group,indices[1],indices[2])
            end
        end
        remove_empty_clusters!(hdp_model)
        if length(indices) > 0
            println("merged:" * string(indices))
        end
    end
end

function create_default_priors(gdim,ldim,prior_type::Symbol)
    if prior_type == :niw
        g_prior = niw_hyperparams(1.0,
            zeros(gdim),
            gdim+3,
            Matrix{Float64}(I, gdim, gdim)*1)
        l_prior = niw_hyperparams(1.0,
            zeros(ldim),
            ldim+3,
            Matrix{Float64}(I, ldim, ldim)*1)
    else
        g_prior = multinomial_hyper(ones(gdim)*500.0)
        l_prior = multinomial_hyper(ones(ldim)*500.0)
    end
    return g_prior, l_prior
end

@everywhere function swap_axes_worker(swap_vec)
    for (k,v) in groups_dict
        v.points = v.points[swap_vec,:]
    end
end


function create_swap_vec(dim,glob_mapping, index)
    swap_vec = zeros(dim)
    reverse_swap_vec = zeros(dim)
    front_index = 1
    back_index = dim
    for (k,v) in enumerate(glob_mapping)
        if k == index || v == 1
            swap_vec[front_index] = k
            reverse_swap_vec[k] = front_index
            front_index+=1
        else
            swap_vec[back_index] = k
            reverse_swap_vec[k] = back_index
            back_index -= 1
        end
    end
    return Int.(swap_vec), Int.(reverse_swap_vec)
end


function calc_global_posterior(hdp_model::hdp_shared_features, ismnm = false)
    pts_count = 0.0
    log_posterior = log(hdp_model.model_hyperparams.γ)
    for (k,group) in hdp_model.groups_dict
        pts_count += size(group.points,2)
        # if ismnm
            log_posterior+= calc_group_posterior(group)
        # end
    end
    log_posterior-= logabsgamma(pts_count)[1]
    for cluster in hdp_model.global_clusters
        if cluster.cluster_params.cluster_params.suff_statistics.N == 0
            continue
        end
        #posterior_param = update_posterior_evidence(model.hyper, model.clusters_params.sufficient_stats, index)
        log_posterior += log_marginal_likelihood(cluster.cluster_params.cluster_params.hyperparams,
            cluster.cluster_params.cluster_params.posterior_hyperparams,
            cluster.cluster_params.cluster_params.suff_statistics)
        log_posterior += log(hdp_model.model_hyperparams.γ) + logabsgamma(cluster.cluster_params.cluster_params.suff_statistics.N)[1]
        # println(cluster.cluster_params.cluster_params.suff_statistics)
    end
    return log_posterior
end



function calc_avg_word(hdp_model::hdp_shared_features)
    # return 0
    global_preds = get_model_global_pred(hdp_model)
    group_mixtures = Dict([k => [x/ sum(counts(v,length(hdp_model.global_clusters))) for x in counts(v,length(hdp_model.global_clusters))] for (k,v) in global_preds])
    # word_count = Dict([k => counts(Int.(v.points[:]), length(hdp_model.model_hyperparams.global_hyper_params.α)) for (k,v) in hdp_model.groups_dict])
    cluster_dists = [x.cluster_params.cluster_params.distribution.α for x in hdp_model.global_clusters]
    total_likelihood = 0.0
    total_points = 0
    for (k,v) in group_mixtures
        parr = zeros(length(hdp_model.model_hyperparams.global_hyper_params.α),length(v))
        for (index,part) in enumerate(v)
            parr[:,index] = cluster_dists[index] .+ log(part)
        end
        parr = exp.(parr)
        parr[isnan.(parr)] .= 0
        wordp = sum(parr,dims = 2)
        wordp = log.(wordp)
        wordp[isnan.(wordp)] .= 0
        rel_pts = hdp_model.groups_dict[k].points
        word_counts = counts(Int.(rel_pts), length(hdp_model.model_hyperparams.global_hyper_params.α))
        cluster_ll = wordp .* word_counts
        cluster_ll[isnan.(cluster_ll)] .= 0
        total_points += sum(word_counts)
        # println(any(isnan.(cluster_ll)))
        total_likelihood += sum(cluster_ll)
    end
    # println("ll: " * string(total_likelihood) * "   pts: " * string(total_points))
    return total_likelihood / total_points
end

function calc_group_posterior(group::local_group)
    log_posterior = log(group.model_hyperparams.α) - logabsgamma(size(group.points,2))[1]
    for cluster in group.local_clusters
        if cluster.cluster_params.cluster_params.suff_statistics.N == 0
            continue
        end
        log_posterior += log_marginal_likelihood(cluster.cluster_params.cluster_params.hyperparams,
            cluster.cluster_params.cluster_params.posterior_hyperparams,
            cluster.cluster_params.cluster_params.suff_statistics)
    end
    return log_posterior
end

function k_mean_likelihood(likehood_rating,k)
    min_likelihood = minimum(likehood_rating)
    max_likelihood = maximum(likehood_rating)
    k_means = zeros(k)
    k_means[1] = min_likelihood
    k_means[k] = 0
    k_interval = abs(max_likelihood - min_likelihood) / k
    for i=2:k-1
        k_means[i] = k_means[i-1] + k_interval
    end
    centers = reshape(k_means,1,k)
    R = kmeans!(reshape(likehood_rating,1,:), centers; maxiter=100)
    return R.centers, assignments(R)
end

function hdp_fit(data, α,γ,prior,iters, initial_custers = 1,burnout = 5;multiprocess=false)
    dim = size(data[1],1)
    gdim = dim
    gprior,lprior = create_default_priors(gdim,dim-gdim,:niw)
    return vhdp_fit(data,gdim, α,γ,α,prior,lprior,iters,initial_custers,burnout,multiprocess=multiprocess)
end

function vhdp_fit(data,gdim, α,γ,η,prior::Symbol,iters, initial_custers = 1,burnout = 5;multiprocess=false)
    dim = size(data[1],1)
    gprior,lprior = create_default_priors(gdim,dim-gdim,prior)
    return vhdp_fit(data,gdim, α,γ,η,gprior,lprior,iters, initial_custers,burnout,multiprocess=multiprocess)
end


function vhdp_fit(data,gdim, α,γ,η,gprior::distribution_hyper_params,lprior,iters, initial_custers = 1,burnout = 5;multiprocess=false)
    global random_seed = nothing
    global initial_local_clusters = initial_custers
    global initial_global_clusters = initial_custers
    global mp = multiprocess
    dim = size(data[1],1)
    model_hyperparams = model_hyper_params(gprior,lprior,α,γ,η,1.0,1.0,dim,gdim + 1)
    model = init_model(nothing; data = data , model_params = model_hyperparams)
    global posterior_history = []
    global word_ll_history = []
    global topic_count = []
    @everywhere global split_delays = true
    global burnout_period = burnout
    if mp
        for w in workers()
            @spawnat w set_burnout(burnout)
        end
    end
    global num_of_workers = nworkers()
    iter = 1
    total_time = 0

    init_first_clusters!(model)
    for i=1:iters
        tic = time()
        model_iteration(model,false,false,burnout)
        toc = time() -tic
        println("Iteration: " * string(i) * "|| Global Counts: " * string([x.clusters_count for x in global_clusters_vector]) * "|| iter time: " * string(toc))
        total_time+= toc

    end
    model.global_clusters = global_clusters_vector
    return model, total_time, posterior_history,word_ll_history,topic_count
end


function remove_all_zeros_from_data(data)
    concatenated_vals = reduce(hcat,values(data))
    all_zeros = []
    for i=1:size(concatenated_vals,1)
        if all(concatenated_vals[i,:] .== 0)
            push!(all_zeros,i)
        end
    end
    allvec = [i for i=1:size(concatenated_vals,1) if !(i in(all_zeros))]
    new_data = deepcopy(data)
    for (k,v) in data
        vlen = size(v,2)
        new_data[k] = v[allvec,:]
    end
    return new_data
end



function get_model_global_pred(model)
    return Dict([k=>create_global_labels(v) for (k,v) in model.groups_dict])
end

function results_stats(pred_dict, gt_dict)
    avg_nmi = 0
    for i=1:length(pred_dict)
        nmi = mutualinfo(pred_dict[i],gt_dict[i])
        avg_nmi += nmi
    end
    return avg_nmi / length(pred_dict)
end
