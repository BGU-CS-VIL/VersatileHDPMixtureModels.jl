include("ds.jl")
include("utils.jl")
include("distributions/niw.jl")
include("distributions/niw_stable_var.jl")
include("distributions/bayes_network_model.jl")
include("distributions/mv_gaussian.jl")
include("distributions/mv_group_gaussian.jl")
include("distributions/multinomial_dist.jl")
include("distributions/multinomial_prior.jl")


mutable struct Dish
    count::Int64
    cluster_params::cluster_parameters
end

mutable struct Table
    count::Int64
    dish_index::Int64
end

mutable struct Restaurant
    tables::Vector{Table}
    points::AbstractArray{Float64,2}
    labels::AbstractArray{Int64,2}
end


mutable struct Crf_model
    dishes::Vector{Dish}
    restaurants::Vector{Restaurant}
    prior::distribution_hyper_params
end


function create_new_dish(prior)
    #pts = point[:,:]
    suff = create_sufficient_statistics(prior,[])
    post = calc_posterior(prior,suff)
    dist = sample_distribution(prior)
    cluster_params = cluster_parameters(prior,dist,suff,post)
    dish = Dish(1,cluster_params)
    return dish
end


function resample_dish!(pts_vec,dish)
    if length(pts_vec) == 0
        suff = create_sufficient_statistics(dish.cluster_params.hyperparams,[])
        post = calc_posterior(dish.cluster_params.hyperparams,suff)
        dist = sample_distribution(dish.cluster_params.hyperparams)
        cluster_params = cluster_parameters(dish.cluster_params.hyperparams,dist,suff,post)
        dish.cluster_params = cluster_params
    else
        all_pts = reduce(hcat,pts_vec)
        suff = create_sufficient_statistics(dish.cluster_params.hyperparams,dish.cluster_params.hyperparams,all_pts)
        post = calc_posterior(dish.cluster_params.hyperparams,suff)
        dist = sample_distribution(post)
        cluster_params = cluster_parameters(dish.cluster_params.hyperparams,dist,suff,post)
        dish.cluster_params = cluster_params
    end
end

function init_crf_model(data,prior)
    restaurants = []
    dishes = []
    for i=1:length(data)
        v = data[i]
        table = Table(size(v,2),1)
        restaurant = Restaurant([table],v,ones(size(v,2),1))
        push!(restaurants,restaurant)
    end
    first_dish = create_new_dish(prior)
    first_dish.count = length(data) #Amount of tables at init
    model = Crf_model([first_dish],restaurants,prior)
    return model
end


function get_tables_probability_vec(tables,α)
    counts = [x.count for x in tables]
    push!(counts,α)
    weights = [x/sum(counts) for x in counts]
    return weights
end

function sample_dish(dishes, pts, γ,new_dish,cur_dish = -1)
    counts = [x.count for x in dishes]
    if cur_dish > 0
        counts[cur_dish] -= 1
    end
    push!(counts,γ)
    weights = [x/sum(counts) for x in counts]
    dists = [x.cluster_params.distribution for x in dishes]
    push!(dists, new_dish.cluster_params.distribution)
    parr = zeros(length(size(pts,2)), length(dists))
    for (k,v) in enumerate(dists)
        log_likelihood!(reshape((@view parr[:,k]),:,1), pts,v)
    end
    parr = sum(parr, dims = 1)[:,:] + reshape(log.(weights),1,:)
    labels = [0][:,:]
    sample_log_cat_array!(labels,parr)
    return labels[1,1]
end

function sample_table(dists,weights,pts)
    parr = zeros(length(size(pts,2)), length(dists))
    for (k,v) in enumerate(dists)
        log_likelihood!(reshape((@view parr[:,k]),:,1), pts,v)
    end
    parr = sum(parr, dims = 1) + reshape(log.(weights),1,:)

    labels = [0][:,:]
    sample_log_cat_array!(labels,parr)
    return labels[1,1]
end



function crf_hdp_fit(data,α,γ,prior,iters)
    model = init_crf_model(data,prior)
    #Each iteration is a single pass on all points
    for iter=1:iters
        println("Iter " * string(iter) * "  Dish count: " * string(length(model.dishes)))
        #Resample dishes distribution
        cur_new_dish = create_new_dish(prior)
        for (dish_index,dish) in enumerate(model.dishes)
            pts = []
            for restaurant in model.restaurants
                for (table_index,table) in enumerate(restaurant.tables)
                    if table.dish_index == dish_index
                        push!(pts, restaurant.points[:,restaurant.labels[:] .== table_index])
                    end
                end
            end
            resample_dish!(pts,dish)
        end
        #Samples points
        for restaurant in model.restaurants
            # println([x.count for x in restaurant.tables])
            for i=1:size(restaurant.points,2)
                point = restaurant.points[:,i][:,:]
                prev_table = restaurant.labels[i,1]
                restaurant.tables[prev_table].count -= 1
                weights = get_tables_probability_vec(restaurant.tables,α)
                new_table_dish = sample_dish(model.dishes, point,γ,cur_new_dish)
                dists = [model.dishes[x.dish_index].cluster_params.distribution for x in restaurant.tables]
                if new_table_dish > length(model.dishes)
                    push!(dists, cur_new_dish.cluster_params.distribution)
                else
                    push!(dists,model.dishes[new_table_dish].cluster_params.distribution)
                end
                try
                    log.(weights)
                catch e
                    println("error print:")
                    println([x.count for x in restaurant.tables])
                    rethrow([e])
                end
                new_table = sample_table(dists,weights,point)
                restaurant.labels[i,1] = new_table
                if new_table > length(restaurant.tables)
                    push!(restaurant.tables,Table(1,new_table_dish))
                    if new_table_dish > length(model.dishes)
                        cur_new_dish.count = 1
                        push!(model.dishes, cur_new_dish)
                        cur_new_dish = create_new_dish(prior)
                    else
                        model.dishes[new_table_dish].count +=1
                    end
                else
                    restaurant.tables[new_table].count += 1
                end
                if restaurant.tables[prev_table].count == 0
                    model.dishes[restaurant.tables[prev_table].dish_index].count -= 1
                end
            end
        end
        #Samples tables
        for restaurant in model.restaurants
            for (index,table) in enumerate(restaurant.tables)
                if table.count == 0
                    continue
                end
                point = restaurant.points[:,(restaurant.labels .== index)[:]]
                prev_dish = table.dish_index
                new_table_dish = sample_dish(model.dishes, point,γ,cur_new_dish,prev_dish)
                model.dishes[prev_dish].count -=1
                if new_table_dish > length(model.dishes)
                    cur_new_dish.count = 1
                    push!(model.dishes, cur_new_dish)
                    cur_new_dish = create_new_dish(prior)
                else
                    model.dishes[new_table_dish].count +=1
                end
                # println("old dish:" * string(prev_dish) * " new dish:" * string(new_table_dish))
                table.dish_index = new_table_dish
            end
        end
    end
    return model
end



function get_dish_labels_from_model(model::Crf_model)
    labels_dict = Dict()
    for (k,v) in enumerate(model.restaurants)
        labels = [v.tables[v.labels[i,1]].dish_index for i=1:size(v.points,2)]
        labels_dict[k] = labels
    end
    return labels_dict
end
