include("../ds.jl")
using LinearAlgebra
using Distributions
using PDMats


struct mv_group_gaussian <: distibution_sample
    μ::Vector{AbstractArray{Float64,1}}
    Σ::AbstractArray{Float64,2}
    invΣ::AbstractArray{Float64,2}
    logdetΣ::Float64
end


function dinvquad!(r,a,x)
    dcolwise_dot!(r,x, a \ x)
end



#MV Gaussian, Using distributions for now, will need to fix it later in order to use cuda
function log_likelihood!(r::AbstractArray,x::AbstractArray, distibution_sample::mv_group_gaussian ,group::Int64 = 1)
    # mvn = MvNormal(distibution_sample.μ,distibution_sample.Σ)
    # logpdf!((@view r[:]),mvn,x)
    z = x .- distibution_sample.μ[group]
    dcolwise_dot!(r,z, distibution_sample.invΣ * z)
    r .= -((length(distibution_sample.Σ) * Float64(log(2π)) + logdet(distibution_sample.Σ))/2) .-r
end
