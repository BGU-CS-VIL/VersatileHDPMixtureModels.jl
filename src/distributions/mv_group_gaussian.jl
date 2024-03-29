struct mv_group_gaussian <: distibution_sample
    μ::Vector{AbstractArray{Float64,1}}
    Σ::AbstractArray{Float64,2}
    invΣ::AbstractArray{Float64,2}
    logdetΣ::Float64
end


function log_likelihood!(r::AbstractArray,x::AbstractArray, distibution_sample::mv_group_gaussian ,group::Int64 = 1)
    z = x .- distibution_sample.μ[group]
    dcolwise_dot!(r,z, distibution_sample.invΣ * z)
    r .= -((length(distibution_sample.Σ) * Float64(log(2π)) + logdet(distibution_sample.Σ))/2) .-r
end
