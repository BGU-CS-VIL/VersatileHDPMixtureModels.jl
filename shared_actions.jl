
function create_subclusters_labels!(labels::AbstractArray{Int64,2},points::AbstractArray{Float64,2},cluster_params::splittable_cluster_params)
    #lr_arr = create_array(zeros(length(labels), 2))
    if size(labels,1) == 0
        return
    end
    lr_arr = zeros(length(labels), 2)
    log_likelihood!(lr_arr[:,1],points,cluster_params.cluster_params_l.distribution)
    log_likelihood!(lr_arr[:,2],points,cluster_params.cluster_params_r.distribution)
    lr_arr[:,1] .+= log(cluster_params.lr_weights[1])
    lr_arr[:,2] .+= log(cluster_params.lr_weights[2])
    #labels .= mapslices(sample_log_cat,lr_arr, dims= [2])
    sample_log_cat_array!(labels,lr_arr)
end


function sample_labels!(labels::AbstractArray{Int64,2},points::AbstractArray{Float64,2},clusters_samples::Dict,final::Bool)
    #lr_arr = create_array(zeros(length(labels), 2))
    parr = zeros(length(labels), length(clusters_samples))
    for (k,v) in clusters_samples
        log_likelihood!((@view parr[:,k]),points,v)
    end
    #println(parr[1:10,:])
    sample_log_cat_array!(labels,parr,final)
end


function create_splittable_from_params(params::cluster_parameters, α::Float64)
    params_l = deepcopy(params)
    params_l.distribution = sample_distribution(params.posterior_hyperparams)
    params_r = deepcopy(params)
    params_r.distribution = sample_distribution(params.posterior_hyperparams)
    #params_l = cluster_parameters(params.distribution_hyper_params, sample_distribution(params.posterior_hyperparams), params.suff_stats, params.posterior_hyperparams)
    #params_r = cluster_parameters(params.distribution_hyper_params, sample_distribution(params.posterior_hyperparams), params.suff_stats, params.posterior_hyperparams)
    lr_weights = rand(Dirichlet([α / 2, α / 2]))
    return splittable_cluster_params(params,params_l,params_r,lr_weights, false, ones(20).*-Inf)
end


function merge_clusters_to_splittable(cpl::cluster_parameters,cpr::cluster_parameters, α::Float64)
    suff_stats = aggregate_suff_stats(cpl.suff_statistics,cpr.suff_statistics)
    posterior_hyperparams = calc_posterior(cpl.hyperparams, suff_stats)
    lr_weights = rand(Dirichlet([cpl.suff_statistics.N + (α / 2), cpr.suff_statistics.N + (α / 2)]))
    cp = cluster_parameters(cpl.hyperparams, cpl.distribution, suff_stats, posterior_hyperparams)
    return splittable_cluster_params(cp,cpl,cpr,lr_weights, false, ones(20).*-Inf)
end


function should_split!(should_split::AbstractArray{Float64,1}, cluster_params::splittable_cluster_params, α::Float64, final::Bool)

    cpl = cluster_params.cluster_params_l
    cpr = cluster_params.cluster_params_r
    cp = cluster_params.cluster_params
    if final || cpl.suff_statistics.N == 0 ||cpr.suff_statistics.N == 0
        should_split .= 0
        return
    end
    log_likihood_l = log_marginal_likelihood(cpl.hyperparams, cpl. posterior_hyperparams, cpl.suff_statistics)
    log_likihood_r = log_marginal_likelihood(cpr.hyperparams, cpr. posterior_hyperparams, cpr.suff_statistics)
    log_likihood = log_marginal_likelihood(cp.hyperparams, cp. posterior_hyperparams, cp.suff_statistics)
    log_HR = log(α) + logabsgamma(cpl.suff_statistics.N)[1] + log_likihood_l + logabsgamma(cpr.suff_statistics.N)[1] + log_likihood_r-(logabsgamma(cp.suff_statistics.N)[1] + log_likihood)
    if log_HR > log(rand())
        should_split .= 1
    end
end


function should_split!(cluster_params::splittable_cluster_params, α::Float64, final::Bool)

    cpl = cluster_params.cluster_params_l
    cpr = cluster_params.cluster_params_r
    cp = cluster_params.cluster_params
    if final || cpl.suff_statistics.N == 0 ||cpr.suff_statistics.N == 0
        should_split .= 0
        return
    end
    log_likihood_l = log_marginal_likelihood(cpl.hyperparams, cpl. posterior_hyperparams, cpl.suff_statistics)
    log_likihood_r = log_marginal_likelihood(cpr.hyperparams, cpr. posterior_hyperparams, cpr.suff_statistics)
    log_likihood = log_marginal_likelihood(cp.hyperparams, cp. posterior_hyperparams, cp.suff_statistics)
    log_HR = log(α) + logabsgamma(cpl.suff_statistics.N)[1] + log_likihood_l + logabsgamma(cpr.suff_statistics.N)[1] + log_likihood_r-(logabsgamma(cp.suff_statistics.N)[1] + log_likihood)
    # println(log_HR)
end




function update_splittable_cluster_params!(splittable_cluser::splittable_cluster_params,
        points::AbstractArray{Float64,2},
        sub_labels::AbstractArray{Int64,1},
        is_global::Bool,
        pts_to_groups = -1)
    update_splittable_cluster_params(splittable_cluser, points, reshape(sub_labels,:,1), is_global)
end


function update_splittable_cluster_params!(splittable_cluser::splittable_cluster_params,
        points::AbstractArray{Float64,2},
        sub_labels::AbstractArray{Int64,1},
        is_global::Bool,
        pts_to_groups = -1)
    cpl = splittable_cluser.cluster_params_l
    cpr = splittable_cluser.cluster_params_r
    cp = splittable_cluser.cluster_params
    # println("bob")
    # println(sum((@view (sub_labels .<= 2)[:])))
    # println(sum((@view (sub_labels .> 2)[:])))
    if is_global
        cp.suff_statistics = create_sufficient_statistics(cp.hyperparams,cp.posterior_hyperparams, points, pts_to_groups)
        pts_gl = @view pts_to_groups[(@view (sub_labels .<= 2)[:])]
        pts_gr = @view pts_to_groups[(@view (sub_labels .> 2)[:])]
        cpl.suff_statistics = create_sufficient_statistics(cpl.hyperparams,cpl.posterior_hyperparams, (@view points[:,(@view (sub_labels .<= 2)[:])]),pts_gl)
        cpr.suff_statistics = create_sufficient_statistics(cpr.hyperparams,cpr.posterior_hyperparams, (@view points[:,(@view (sub_labels .> 2)[:])]),pts_gr)
    else
        cp.suff_statistics = create_sufficient_statistics(cp.hyperparams,cp.posterior_hyperparams, points)
        cpl.suff_statistics = create_sufficient_statistics(cpl.hyperparams,cpl.posterior_hyperparams, @view points[:,(@view (sub_labels .% 2 .== 1)[:])])
        cpr.suff_statistics = create_sufficient_statistics(cpr.hyperparams,cpr.posterior_hyperparams, @view points[:,(@view (sub_labels .% 2 .== 0)[:])])
    end
    cp.posterior_hyperparams = calc_posterior(cp.hyperparams, cp.suff_statistics)
    cpl.posterior_hyperparams = calc_posterior(cpl.hyperparams, cpl.suff_statistics)
    cpr.posterior_hyperparams = calc_posterior(cpr.hyperparams, cpr.suff_statistics)
end

function update_splittable_cluster_params(splittable_cluser::splittable_cluster_params,
        points::AbstractArray{Float64,2},
        sub_labels::AbstractArray{Int64,1},
        is_global::Bool,
        pts_to_groups = -1)
    cpl = splittable_cluser.cluster_params_l
    cpr = splittable_cluser.cluster_params_r
    cp = splittable_cluser.cluster_params
    # println(sum((@view (sub_labels .> 2)[:])))
    if is_global
        cp.suff_statistics = create_sufficient_statistics(cp.hyperparams,cp.posterior_hyperparams, points, pts_to_groups)
        pts_gl = @view pts_to_groups[(@view (sub_labels .<= 2)[:])]
        pts_gr = @view pts_to_groups[(@view (sub_labels .> 2)[:])]
        cpl.suff_statistics = create_sufficient_statistics(cpl.hyperparams, cpl.posterior_hyperparams, (@view points[:,(@view (sub_labels .<= 2)[:])]),pts_gl)
        cpr.suff_statistics = create_sufficient_statistics(cpr.hyperparams, cpr.posterior_hyperparams,(@view points[:,(@view (sub_labels .> 2)[:])]),pts_gr)
    else
        cp.suff_statistics = create_sufficient_statistics(cp.hyperparams,cp.posterior_hyperparams,points)
        cpl.suff_statistics = create_sufficient_statistics(cpl.hyperparams, cpl.posterior_hyperparams,@view points[:,(@view (sub_labels .% 2 .== 1)[:])])
        cpr.suff_statistics = create_sufficient_statistics(cpr.hyperparams, cpr.posterior_hyperparams,@view points[:,(@view (sub_labels .% 2 .== 0)[:])])
    end
    begin
        cp.posterior_hyperparams = calc_posterior(cp.hyperparams, cp.suff_statistics)
        cpl.posterior_hyperparams = calc_posterior(cpl.hyperparams, cpl.suff_statistics)
        cpr.posterior_hyperparams = calc_posterior(cpr.hyperparams, cpr.suff_statistics)
    end
    return splittable_cluser
end

# function update_splittable_cluster_params!(splittable_cluser::splittable_cluster_params, points::AbstractArray{Float64,2}, sub_labels::AbstractArray{Int64,1})
#     cpl = splittable_cluser.cluster_params_l
#     cpr = splittable_cluser.cluster_params_r
#     cp = splittable_cluser.cluster_params
#     cp.suff_statistics = create_sufficient_statistics(cp.hyperparams, points)
#     cpl.suff_statistics = create_sufficient_statistics(cpl.hyperparams, @view points[:,(@view (sub_labels .== 1)[:])])
#     cpr.suff_statistics = create_sufficient_statistics(cpr.hyperparams, @view points[:,(@view (sub_labels .== 2)[:])])
#     cp.posterior_hyperparams = calc_posterior(cp.hyperparams, cp.suff_statistics)
#     cpl.posterior_hyperparams = calc_posterior(cpl.hyperparams, cpl.suff_statistics)
#     cpr.posterior_hyperparams = calc_posterior(cpr.hyperparams, cpr.suff_statistics)
# end
#
#
# function update_splittable_cluster_params(splittable_cluser::splittable_cluster_params, points::AbstractArray{Float64,2}, sub_labels::AbstractArray{Int64,1})
#     cpl = splittable_cluser.cluster_params_l
#     cpr = splittable_cluser.cluster_params_r
#     cp = splittable_cluser.cluster_params
#     cp.suff_statistics = create_sufficient_statistics(cp.hyperparams, points)
#     cpl.suff_statistics = create_sufficient_statistics(cpl.hyperparams, @view points[:,(@view (sub_labels .== 1)[:])])
#     cpr.suff_statistics = create_sufficient_statistics(cpr.hyperparams, @view points[:,(@view (sub_labels .== 2)[:])])
#     cp.posterior_hyperparams = calc_posterior(cp.hyperparams, cp.suff_statistics)
#     cpl.posterior_hyperparams = calc_posterior(cpl.hyperparams, cpl.suff_statistics)
#     cpr.posterior_hyperparams = calc_posterior(cpr.hyperparams, cpr.suff_statistics)
#     return splittable_cluser
# end

function sample_cluster_params!(params::splittable_cluster_params, α::Float64, is_zero_dim = false)
    points_count = Vector{Float64}()
    params.cluster_params.distribution = sample_distribution(params.cluster_params.posterior_hyperparams)
    params.cluster_params_l.distribution = sample_distribution(params.cluster_params_l.posterior_hyperparams)
    params.cluster_params_r.distribution = sample_distribution(params.cluster_params_r.posterior_hyperparams)
    push!(points_count, params.cluster_params_l.suff_statistics.N)
    push!(points_count, params.cluster_params_r.suff_statistics.N)
    points_count .+= α/2
    params.lr_weights = rand(Dirichlet(points_count))
    log_likihood_l = log_marginal_likelihood(params.cluster_params_l.hyperparams,params.cluster_params_l.posterior_hyperparams, params.cluster_params_l.suff_statistics)
    log_likihood_r = log_marginal_likelihood(params.cluster_params_r.hyperparams,params.cluster_params_r.posterior_hyperparams, params.cluster_params_r.suff_statistics)

    # params.logsublikelihood_hist[1:4] = params.logsublikelihood_hist[2:5]
    # params.logsublikelihood_hist[5] = log_likihood_l + log_likihood_r
    # logsublikelihood_now = 0.0
    # for i=1:5
    #     logsublikelihood_now += params.logsublikelihood_hist[i] *0.20
    # end
    # if logsublikelihood_now != -Inf && logsublikelihood_now - params.logsublikelihood_hist[5] < 1e-2 # propogate abs change to other versions?
    #     params.splittable = true
    # end
    # println(split_delays)
    # logsublikelihood_now = (log_likihood_l+log_likihood_r) /  (params.cluster_params_l.suff_statistics.N+ params.cluster_params_r.suff_statistics.N)
    # if logsublikelihood_now - params.logsublikelihood_hist[1] < 0
    #     params.splittable = true
    # end
    # params.logsublikelihood_hist[1] = logsublikelihood_now
    #
    # if split_delays == false
    #     params.splittable = true
    # end
    params.logsublikelihood_hist[1:burnout_period-1] = params.logsublikelihood_hist[2:burnout_period]
    params.logsublikelihood_hist[burnout_period] = log_likihood_l + log_likihood_r
    logsublikelihood_now = 0.0

    for i=1:burnout_period
        logsublikelihood_now += params.logsublikelihood_hist[i] *(1/(burnout_period-0.1))
    end

    if logsublikelihood_now != -Inf && logsublikelihood_now - params.logsublikelihood_hist[burnout_period] < 1e-2 # propogate abs change to other versions?
        # println(params.logsublikelihood_hist)
        params.splittable = true
    end
    if is_zero_dim == true
        params.splittable = true
    end
    return params.cluster_params.suff_statistics.N
end

function sample_cluster_params!(params::splittable_cluster_params, α::Float64, counts::AbstractArray{Int64,1})
    points_count = [params.cluster_params_l.suff_statistics.N, params.cluster_params_r.suff_statistics.N]
    params.cluster_params.distribution = sample_distribution(params.cluster_params.posterior_hyperparams)
    params.cluster_params_l.distribution = sample_distribution(params.cluster_params_l.posterior_hyperparams)
    params.cluster_params_r.distribution = sample_distribution(params.cluster_params_r.posterior_hyperparams)
    points_count .+= α/2
    # println(points_count)
    params.lr_weights = rand(Dirichlet(points_count))
    log_likihood_l = log_marginal_likelihood(params.cluster_params_l.hyperparams,params.cluster_params_l.posterior_hyperparams, params.cluster_params_l.suff_statistics)
    log_likihood_r = log_marginal_likelihood(params.cluster_params_r.hyperparams,params.cluster_params_r.posterior_hyperparams, params.cluster_params_r.suff_statistics)

    # params.logsublikelihood_hist[1:4] = params.logsublikelihood_hist[2:5]
    # params.logsublikelihood_hist[5] = log_likihood_l + log_likihood_r
    # logsublikelihood_now = 0.0
    # for i=1:5
    #     logsublikelihood_now += params.logsublikelihood_hist[i] *0.20
    # end
    # if logsublikelihood_now != -Inf && logsublikelihood_now - params.logsublikelihood_hist[5] < 1e-2 # propogate abs change to other versions?
    #     params.splittable = true
    # end
    params.logsublikelihood_hist[1:burnout_period-1] = params.logsublikelihood_hist[2:burnout_period]
    params.logsublikelihood_hist[burnout_period] = log_likihood_l + log_likihood_r
    logsublikelihood_now = 0.0
    for i=1:burnout_period
        logsublikelihood_now += params.logsublikelihood_hist[i] *(1/(burnout_period-0.1))
    end

    if logsublikelihood_now != -Inf && logsublikelihood_now - params.logsublikelihood_hist[burnout_period] < 1e-2 # propogate abs change to other versions?
        # println(params.logsublikelihood_hist)
        params.splittable = true
    end
    return params.cluster_params.suff_statistics.N
end
