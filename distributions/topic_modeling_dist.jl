include("../ds.jl")
using LinearAlgebra
using Distributions
# using PDMats


struct topic_modeling_dist <: distibution_sample
    α::AbstractArray{Float64,1}
end



#topic_modeling_dist
function log_likelihood!(r::AbstractArray,x::AbstractArray, distibution_sample::topic_modeling_dist , group::Int64 = -1)
    if length(distibution_sample.α) == 0
        return
    end
    @inbounds for i in eachindex(r)
        r[i] = distibution_sample.α[Int64.(x[i])]
    end
end
