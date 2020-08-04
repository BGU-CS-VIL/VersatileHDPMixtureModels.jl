struct bayes_network_model <: distribution_hyper_params
    κ::Float64
    m::AbstractArray{Float64}
    ν::Float64
    ψ::AbstractArray{Float64}
    count::Int64
    ms::Vector{AbstractArray{Float64}}
    λ::Float64
end

mutable struct bayes_network_sufficient_statistics <: sufficient_statistics
    N::Float64
    Ngroups::Vector{Float64}
    points_sum::AbstractArray{Float64,1}
    mu_vector::Vector{AbstractArray{Float64,1}}
    S::AbstractArray{Float64,2}
end


function calc_posterior(prior::bayes_network_model, suff_statistics::bayes_network_sufficient_statistics)
    if suff_statistics.N == 0
        return prior
    end
    κ = prior.κ + suff_statistics.N
    ν = prior.ν + suff_statistics.N
    m = (prior.m.*prior.κ + suff_statistics.points_sum) / κ
    ψ = (prior.ν * prior.ψ + prior.κ*prior.m*prior.m' -κ*m*m'+ suff_statistics.S) / ν
    ψ = Matrix(Hermitian(ψ))
    #ψ = Hermitian((prior.ν * prior.ψ + prior.κ*prior.m*prior.m' -κ*m*m' + suff_statistics.S) ./ ν)
    if  isposdef(ψ) == false
        println(ψ)
        println(m)
        println(κ*m*m')
        println(suff_statistics)
    end
    return bayes_network_model(κ,m,ν,ψ,prior.count,suff_statistics.mu_vector,prior.λ)
end

function sample_distribution(hyperparams::bayes_network_model)
    Σ = rand(Distributions.InverseWishart(hyperparams.ν, hyperparams.ν* hyperparams.ψ))
    mu_vector = Vector{AbstractArray{Float64,1}}()
    μ = rand(Distributions.MvNormal(hyperparams.m, Σ/hyperparams.κ))
    for i=1:hyperparams.count
        push!(mu_vector, rand(Distributions.MvNormal(hyperparams.ms[i], Σ/hyperparams.κ)))
    end
    return mv_group_gaussian(mu_vector,Σ,inv(Σ),logdet(Σ))
end

function create_sufficient_statistics(hyper::bayes_network_model,
        posterior::bayes_network_model,
        points::AbstractArray{Float64,2},
        point_to_group = 0)
    if size(points,2) == 0
        return bayes_network_sufficient_statistics(0,
            [0 for i=1:hyper.count],
            zeros(length(hyper.m)),
            [hyper.m for i=1:hyper.count],
            zeros(length(hyper.m),length(hyper.m)))
    end
    pts_dict = Dict()
    for i=1:hyper.count
        pts_dict[i] = @view points[:,point_to_group .== i]
    end
    mu_vector = create_mu_vector(pts_dict, posterior.ψ,hyper.λ)
    dim = size(posterior.ψ,1)
    S = zeros(dim,dim)
    points_sum = zeros(dim)
    Ngroups = Vector{Float64}()
    N = 0
    for i=1:hyper.count
        pts = Array(pts_dict[i])
        if size(pts,2) == 0
            push!(Ngroups,0)
        else
            movedPts = pts .- mu_vector[i]
            S += movedPts * movedPts'
            points_sum += sum(movedPts, dims = 2)
            push!(Ngroups,size(pts,2))
            N += size(pts,2)
        end
    end
    return bayes_network_sufficient_statistics(N,Ngroups,points_sum[:],mu_vector,S)
end

function log_marginal_likelihood(hyper::bayes_network_model, posterior_hyper::bayes_network_model, suff_stats::bayes_network_sufficient_statistics)
    D = size(suff_stats.S,1)
    logpi = log(pi)
    return -suff_stats.N*D/2*logpi +
        log_multivariate_gamma(posterior_hyper.ν/2, D)-
        log_multivariate_gamma(hyper.ν/2, D) +
         (hyper.ν/2)*logdet(hyper.ψ*hyper.ν)-
         (posterior_hyper.ν/2)*logdet(posterior_hyper.ψ*posterior_hyper.ν) +
         (D/2)*(log(hyper.κ)-(D/2)*log(posterior_hyper.κ))
end

function aggregate_suff_stats(suff_l::bayes_network_sufficient_statistics, suff_r::bayes_network_sufficient_statistics)
    new_suff = deepcopy(suff_l)
    N = 0
    for i=1:length(new_suff.Ngroups)
        new_suff.mu_vector[i] = (new_suff.mu_vector[i] .* (new_suff.Ngroups[i] / (new_suff.Ngroups[i] + suff_r.Ngroups[i]))) +(suff_r.mu_vector[i] .* (suff_r.Ngroups[i] / (new_suff.Ngroups[i] + suff_r.Ngroups[i])))
        new_suff.Ngroups[i] += suff_r.Ngroups[i]
        new_suff.N += suff_r.Ngroups[i]
    end
    new_suff.S += suff_r.S
    new_suff.points_sum += suff_r.points_sum
    return new_suff
end

function create_mu_vector(points::Dict, Σ::AbstractArray{Float64}, λ::Float64)
    group_count = length(keys(points))
    A, b = create_matrix_for_least_squares(points, Σ, λ, group_count)
    xhat = inv(A'*A)*(A'*b)
    dim = size(Σ,1)
    mu_vector = reshape(xhat, dim, group_count)
    return [mu_vector[:,i] for i=1:group_count]
end

function create_matrix_for_least_squares(points::Dict, Σ::AbstractArray{Float64}, λ::Float64, group_count::Int64)
    points_count = zeros(group_count)
    points_sum::Int64 = 0
    dim = size(Σ,1)
    for i=1:length(points)
        points_count[i] = (points[i] == -1 ? 0 : size(points[i],2))
        points_sum += points_count[i]
    end
    points = [x for x in values(points) if x != -1]
    points_arr = vcat(points, [zeros(dim) for x in 1:group_count])

    points_arr = reduce(hcat, points_arr)

    points_arr = points_arr[:]

    A = zeros(length(points_arr),groups_count * dim)
    L = cholesky(Σ).U
    Linv = Matrix(inv(L))

    for i=1:points_sum
        b =  points_arr[(i-1)*dim+1:i*dim]
        tmp = Linv * b

        points_arr[(i-1)*dim+1:i*dim] = Linv * points_arr[(i-1)*dim+1:i*dim]
    end
    count = 1


    for i=1:group_count
        for j=1:points_count[i]
            A[count:count+dim-1,dim*(i-1)+1:dim*i] = Linv
            count+= dim
        end
    end

    λsqrt = λ^2
    count += dim #Skip the first μ

    for i=1:(size(A,2) - dim)
        A[count,i] = -λsqrt
        A[count,i+dim] = λsqrt
        count+=1
    end
    return A, points_arr
end


function create_bayes_model_hyper_from_niw(niw::niw_hyperparams, λ, count)
    ms = [niw.m for i=1:count]
    return bayes_network_model(niw.κ,niw.m,niw.ν,niw.ψ,count,ms,λ)
end
