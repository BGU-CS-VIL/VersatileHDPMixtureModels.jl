include("../ds.jl")
using LinearAlgebra
using Distributions
# using PDMats


struct compact_mnm_dist <: distibution_sample
    α::AbstractArray{Float64,1}
end



#topic_modeling_dist
function log_likelihood!(r::AbstractArray,x::AbstractArray, distibution_sample::compact_mnm_dist , group::Int64 = -1)
    if length(distibution_sample.α) == 0
        return
    end
    @inbounds for i in eachindex(r)
        r[i] = 0.0
        @inbounds for j=1:size(x,1)
            if x[j,i] > 0
                r[i] += distibution_sample.α[Int64.(x[j,i])]
            end
        end
    end
end
