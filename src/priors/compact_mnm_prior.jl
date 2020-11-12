include("../ds.jl")

struct compact_mnm_hyper <: distribution_hyper_params
    α::AbstractArray{Float64,1}
end

mutable struct compact_mnm_sufficient_statistics <: sufficient_statistics
    N::Float64
    points_sum::AbstractArray{Int64,1}
end


function calc_posterior(prior:: compact_mnm_hyper, suff_statistics::compact_mnm_sufficient_statistics)
    if suff_statistics.N == 0
        return prior
    end
    return compact_mnm_hyper(prior.α + suff_statistics.points_sum)
end

function sample_distribution(hyperparams::compact_mnm_hyper)
    cat_dist = rand(Dirichlet(hyperparams.α))
    return compact_mnm_dist(log.(cat_dist))
end

function create_sufficient_statistics(hyper::compact_mnm_hyper,posterior::compact_mnm_hyper,points::AbstractArray{Float64,2}, pts_to_group = 0)
    if length(points) == 0
        return compact_mnm_sufficient_statistics(size(points,2),zeros(Int64,length(hyper.α)))
    end
    pt_count = counts(Int.(points),length(hyper.α))
    return compact_mnm_sufficient_statistics(size(points,2),pt_count)
end

function log_multivariate_gamma(x::Number, D::Number)
    res::Float64 = D*(D-1)/4*log(pi)
    for d = 1:D
        res += logabsgamma(x+(1-d)/2)[1]
    end
    return res
end

function log_marginal_likelihood(hyper::compact_mnm_hyper, posterior_hyper::compact_mnm_hyper, suff_stats::compact_mnm_sufficient_statistics)
    D = length(suff_stats.points_sum)
    logpi = log(pi)
    val = logabsgamma(sum(hyper.α))[1] -logabsgamma(sum(posterior_hyper.α))[1] + sum((x-> logabsgamma(x)[1]).(posterior_hyper.α) - (x-> logabsgamma(x)[1]).(hyper.α))
    return val
end

function aggregate_suff_stats(suff_l::compact_mnm_sufficient_statistics, suff_r::compact_mnm_sufficient_statistics)
    return compact_mnm_sufficient_statistics(suff_l.N+suff_r.N, suff_l.points_sum + suff_r.points_sum)
end
