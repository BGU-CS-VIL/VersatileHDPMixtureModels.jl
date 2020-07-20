include("../ds.jl")

struct topic_modeling_hyper <: distribution_hyper_params
    α::AbstractArray{Float64,1}
end

mutable struct topic_modeling_sufficient_statistics <: sufficient_statistics
    N::Float64
    points_sum::AbstractArray{Int64,1}
end


function calc_posterior(prior:: topic_modeling_hyper, suff_statistics::topic_modeling_sufficient_statistics)
    if suff_statistics.N == 0
        return prior
    end
    return topic_modeling_hyper(prior.α + suff_statistics.points_sum)
end

function sample_distribution(hyperparams::topic_modeling_hyper)
    cat_dist = rand(Dirichlet(hyperparams.α))
    return topic_modeling_dist(log.(cat_dist))
end

function create_sufficient_statistics(hyper::topic_modeling_hyper,posterior::topic_modeling_hyper,points::AbstractArray{Float64,2}, pts_to_group = 0)
    if length(points) == 0
        return topic_modeling_sufficient_statistics(size(points,2),zeros(Int64,length(hyper.α)))
    end
    pt_count = counts(Int.(points),length(hyper.α))
    return topic_modeling_sufficient_statistics(size(points,2),pt_count)
end

function log_multivariate_gamma(x::Number, D::Number)
    res::Float64 = D*(D-1)/4*log(pi)
    for d = 1:D
        res += logabsgamma(x+(1-d)/2)[1]
    end
    return res
end

function log_marginal_likelihood(hyper::topic_modeling_hyper, posterior_hyper::topic_modeling_hyper, suff_stats::topic_modeling_sufficient_statistics)
    D = length(suff_stats.points_sum)
    N = Int(ceil(sum(posterior_hyper.α - hyper.α)))
    logpi = log(pi)
    val = sum((x-> logabsgamma(x)[1]).(posterior_hyper.α) - suff_stats.points_sum .* (x-> logabsgamma(x)[1]).(hyper.α)) + logabsgamma(sum(hyper.α))[1] -logabsgamma(N + sum(hyper.α))[1]
    return val
end

function aggregate_suff_stats(suff_l::topic_modeling_sufficient_statistics, suff_r::topic_modeling_sufficient_statistics)
    return topic_modeling_sufficient_statistics(suff_l.N+suff_r.N, suff_l.points_sum + suff_r.points_sum)
end
