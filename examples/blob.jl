using Distributed
addprocs(1)
@everywhere using Revise
@everywhere using VersatileHDPMixtureModels

function generate_data(dim, groups_count,sample_count,var,α = 10, γ = 1)
    crf_prior = hdp_prior_crf_draws(sample_count,groups_count,α,γ)
    pts,labels = generate_grouped_gaussian_from_hdp_group_counts(crf_prior[2],dim,var)
    return pts, labels
end

function results_stats(pred_dict, gt_dict)
    avg_nmi = 0
    for i=1:length(pred_dict)
        nmi = mutualinfo(pred_dict[i],gt_dict[i])
        avg_nmi += nmi
    end
    return avg_nmi / length(pred_dict)
end


function run_and_compare(pts,labels,gdim,iters = 100)     
     gprior, lprior = create_default_priors(gdim,0,:niw)
     model = hdp_fit(pts,10,1,gprior,iters)
     model_results = get_model_global_pred(model[1])
     return results_stats(labels,model_results)
 end

gdim = 3
pts,labels = generate_data(gdim,4,100,100.0)
run_and_compare(pts,labels,gdim)