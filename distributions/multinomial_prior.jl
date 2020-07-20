include("../ds.jl")

struct multinomial_hyper <: distribution_hyper_params
    α::AbstractArray{Float64,1}
end

mutable struct multinomial_sufficient_statistics <: sufficient_statistics
    N::Float64
    points_sum::AbstractArray{Float64,1}
    S
end


function calc_posterior(prior:: multinomial_hyper, suff_statistics::multinomial_sufficient_statistics)
    if suff_statistics.N == 0
        return prior
    end
    return multinomial_hyper(prior.α + suff_statistics.points_sum)
end

function sample_distribution(hyperparams::multinomial_hyper)
    return multinomial_dist(log.(rand(Dirichlet(hyperparams.α))))
end

function create_sufficient_statistics(hyper::multinomial_hyper,posterior::multinomial_hyper,points::AbstractArray{Float64,2}, pts_to_group = 0)
    # pts = copy(points)
    points_sum = sum(points, dims = 2)[:]
    #S = pts * pts'
    return multinomial_sufficient_statistics(size(points,2),points_sum, 0)
end

function log_multivariate_gamma(x::Number, D::Number)
    res::Float64 = D*(D-1)/4*log(pi)
    for d = 1:D
        res += logabsgamma(x+(1-d)/2)[1]
    end
    return res
end

function log_marginal_likelihood(hyper::multinomial_hyper, posterior_hyper::multinomial_hyper, suff_stats::multinomial_sufficient_statistics)
    D = length(suff_stats.points_sum)
    logpi = log(pi)
    val = logabsgamma(sum(hyper.α))[1] -logabsgamma(sum(posterior_hyper.α))[1] + sum((x-> logabsgamma(x)[1]).(posterior_hyper.α) - (x-> logabsgamma(x)[1]).(hyper.α))
    return val
end

function aggregate_suff_stats(suff_l::multinomial_sufficient_statistics, suff_r::multinomial_sufficient_statistics)
    return multinomial_sufficient_statistics(suff_l.N+suff_r.N, suff_l.points_sum + suff_r.points_sum, suff_l.S+suff_r.S)
end
