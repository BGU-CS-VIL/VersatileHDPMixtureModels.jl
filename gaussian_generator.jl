using NPZ
using Distributions
using LinearAlgebra
using Distributed
using DistributedArrays
using StatsBase
using Distributions
using SpecialFunctions
using CatViews
using LinearAlgebra
using Random

function generate_sph_gaussian_data(N::Int64, D::Int64, K::Int64)
	x = randn(D,N)
	tpi = rand(Dirichlet(ones(K)))
	tzn = rand(Multinomial(N,tpi))
	tz = zeros(N)

	tmean = zeros(D,K)
	tcov = zeros(D,D,K)

	ind = 1
	println(tzn)
	for i=1:length(tzn)
		indices = ind:ind+tzn[i]-1
		tz[indices] .= i
		tmean[:,i] .= rand(MvNormal(zeros(D), 100*Matrix{Float64}(I, D, D)))
		tcov[:,:,i] .= rand(InverseGamma((D+2)/2,1))*Matrix{Float64}(I, D, D)
		# T = chol(slice(tcov,:,:,i))
		# x[:,indices] = broadcast(+, T*x[:,indices], tmean[:,i]);
		d = MvNormal(tmean[:,i], tcov[:,:,i])
		for j=indices
			x[:,j] = rand(d)
		end
		ind += tzn[i]
	end
	x, tz, tmean, tcov
end


function generate_gaussian_data(N::Int64, D::Int64, K::Int64)
	x = randn(D,N)
	tpi = rand(Dirichlet(ones(K)))
	tzn = rand(Multinomial(N,tpi))
	tz = zeros(N)

	tmean = zeros(D,K)
	tcov = zeros(D,D,K)

	ind = 1
	println(tzn)
	for i=1:length(tzn)
		indices = ind:ind+tzn[i]-1
		tz[indices] .= i
		tmean[:,i] .= rand(MvNormal(zeros(D), 100*Matrix{Float64}(I, D, D)))
		tcov[:,:,i] .= rand(InverseWishart(D+2, Matrix{Float64}(I, D, D)))
		# T = chol(slice(tcov,:,:,i))
		# x[:,indices] = broadcast(+, T'*x[:,indices], tmean[:,i]);
		d = MvNormal(tmean[:,i], tcov[:,:,i])
		for j=indices
			x[:,j] = rand(d)
		end
		ind += tzn[i]
	end
	x, tz, tmean, tcov
end


function generate_mnmm_data(N::Int64, D::Int64, K::Int64, trials::Int64)
	clusters = zeros(D,K)
	x = zeros(D,N)
	labels = rand(1:K,(N,))
	for i=1:K
		alphas = rand(1:20,(D,))
		alphas[i] = rand(30:100)
		clusters[:,i] = rand(Dirichlet(alphas))
	end
	for i=1:N
		x[:,i] = rand(Multinomial(trials,clusters[:,labels[i]]))
	end
	return x, labels, clusters
end



function generate_grouped_mnm_data(N::Int64, D_global::Int64, D_local, K_global::Int64, K_local::Int64, groups_count::Int64, rand_local::Bool, trials::Int64)
	global_weights = rand(Dirichlet(ones(K_global)))
	pts_dict = Dict()
	labels_dict = Dict()
	clusters_g = zeros(D_global,K_global)
	for i=1:length(global_weights)
		alphas = ones(D_global)*2
		alphas[rand(1:D_global,Int(floor(D_global/40)))] .= rand(trials/2:trials)
		clusters_g[:,i] = rand(Dirichlet(alphas))
	end
	for j=1:groups_count
		x = zeros(D_global + D_local,N)
		x_labels = zeros(size(x,2),2)
		if rand_local
			local_k = rand(1:K_local*2)
		else
			local_k = K_local
		end
		clusters_tzn = sample(1:length(global_weights),ProbabilityWeights(global_weights),local_k)
		local_weights = rand(Dirichlet(ones(local_k)*100))
		clusters_l = zeros(D_global+D_local,local_k)
		ind = 1
		group_ind = 1
		group_tzn = rand(Multinomial(N,local_weights))
		# group_tzn = rand(Multinomial(N,ones(local_k)))
		for i=1:length(local_weights)
			indices = ind:ind+group_tzn[i]-1
			# println(indices)

			alphas = ones(D_local)*2
			alphas[rand(1:D_local,Int(floor(D_local/1.25)))] .= rand(trials/2:trials)
			# local_part = rand(Dirichlet(alphas))
			# clusters_l[:,i] .= cat(clusters_g[:,clusters_tzn[i]], alphas, dims = [1])

			pvector = rand(Dirichlet(alphas))
			for a=indices
				x[1:D_global,a] = rand(Multinomial(trials,clusters_g[:,clusters_tzn[i]]))
				x[D_global+1:end,a] = rand(Multinomial(trials,pvector))
				x_labels[a,1] = clusters_tzn[i]
				x_labels[a,2] = i
			end
			ind += group_tzn[i]
		end
		pts_dict[j] = x
		labels_dict[j] = x_labels
	end
	return pts_dict, labels_dict
end





function generate_grouped_gaussian_data(N::Int64, D_global::Int64, D_local, K_global::Int64, K_local::Int64, groups_count::Int64, rand_local::Bool, var_size, hdp=false)
	global_weights = rand(Dirichlet(ones(K_global)))
	pts_dict = Dict()
	labels_dict = Dict()
	tmean = zeros(D_global,K_global)
	tcov = zeros(D_global,D_global,K_global)
	for i=1:length(global_weights)
		tmean[:,i] .= rand(MvNormal(zeros(D_global), var_size*Matrix{Float64}(I, D_global, D_global)))
		tcov[:,:,i] .= rand(InverseWishart(D_global+2, Matrix{Float64}(I, D_global, D_global)))
	end
	for j=1:groups_count
		x = randn(D_global + D_local,N)
		x_labels = zeros(size(x,2),2)
		if rand_local
			local_k = rand(K_local-2:K_local+2)
		else
			local_k = K_local
		end
		clusters_tzn = sample(1:length(global_weights),ProbabilityWeights(global_weights),local_k)
		local_weights = rand(Dirichlet(ones(local_k)*100))
		group_mean = zeros(D_global+D_local,local_k)
		group_cov = zeros(D_global+D_local,D_global+D_local,local_k)
		ind = 1
		group_ind = 1
		group_tzn = rand(Multinomial(N,local_weights))
		# group_tzn = rand(Multinomial(N,ones(local_k)))
		for i=1:length(local_weights)
			indices = ind:ind+group_tzn[i]-1
			g_mean = rand(MvNormal(zeros(D_local), var_size*Matrix{Float64}(I, D_local, D_local)))
			g_cov = rand(InverseWishart(D_local+2, Matrix{Float64}(I, D_local, D_local)))
			group_mean[:,i] .= cat(tmean[:,clusters_tzn[i]], g_mean, dims = [1])
			group_cov[:,:,i] .= cat(tcov[:,:,clusters_tzn[i]], g_cov, dims = [1,2])
			d = MvNormal(group_mean[:,i], group_cov[:,:,i])
			for j=indices
				x[:,j] = rand(d)
				x_labels[j,1] = clusters_tzn[i]
				x_labels[j,2] = i
			end
			ind += group_tzn[i]
		end
		if hdp
			x[D_global+1:end,:] .= 0
		end
		pts_dict[j] = x
		labels_dict[j] = x_labels
	end
	return pts_dict, labels_dict
end



function create_mnmm_data()
	 x,labels,clusters = generate_mnmm_data(10^6,100,6,1000)
	 npzwrite("data/mnmm/1milD100K6.npy",x')
	 x,labels,clusters = generate_mnmm_data(10^6,100,60,100)
	 npzwrite("data/mnmm/1milD100K60.npy",x')
 end

function create_gaussian_data()
	x,labels,clusters = generate_gaussian_data(10^5,2,20)
	npzwrite("data/2d-1mil/samples_100k1.npy",x')
	# x,labels,clusters = generate_gaussian_data(10^5,100,60,100)
	# npzwrite("data/30d-1mil/samples.npy",x')
end

function create_grouped_gaussian_data()
	samples_count = 10000
	global_dim = 2
	local_dim = 1
	global_clusters_count = 4
	local_clusters_count = 5
	groups_count = 5
	random_local_cluster_count = true
	var_size = 60
	hdp = true

	path_to_save = "data/3d-gaussians/"
	prefix = "gg"

	pts, labels = generate_grouped_gaussian_data(samples_count,
		global_dim,
		local_dim,
		global_clusters_count,
		local_clusters_count,
		groups_count,
		random_local_cluster_count,
		var_size,
		hdp)
	for (k,v) in pts
		npzwrite(path_to_save * prefix * string(k) * ".npy",v')
	end
	for (k,v) in labels
		npzwrite(path_to_save * prefix*"_labels" * string(k) * ".npy",v)
	end
end



function create_grouped_mnmm_data()
	samples_count = 10000
	global_dim = 100
	local_dim = 100
	global_clusters_count = 10
	local_clusters_count = 20
	groups_count = 4
	random_local_cluster_count = false
	trials = 1000

	path_to_save = "data/200d-mnm/"
	prefix = "g"

	pts, labels = generate_grouped_mnm_data(samples_count,
		global_dim,
		local_dim,
		global_clusters_count,
		local_clusters_count,
		groups_count,
		random_local_cluster_count,
		trials)
	for (k,v) in pts
		npzwrite(path_to_save * prefix * string(k) * ".npy",v')
	end
	for (k,v) in labels
		npzwrite(path_to_save * prefix*"_labels" * string(k) * ".npy",v)
	end
end


function single_crp_draw(tables,α)
	tables = push!(tables,α)
	sum_tables = sum(tables)
	table_probs = [x / sum_tables for x in tables]
	return sample(ProbabilityWeights(table_probs))
end



function hdp_prior_crf_draws(N,J,α,γ)
	groups_tables_counts = Dict()
	for j=1:J
		table_counts = []
		for i=1:N
			point_table = single_crp_draw(table_counts[:],α)
			if point_table == length(table_counts)+1
				push!(table_counts,1)
			else
				table_counts[point_table] += 1
			end
		end
		groups_tables_counts[j] = table_counts
	end
	global_tables_count = []
	groups_tables_assignments = Dict([i=>[] for i=1:J])
	cur_group_table = Dict([i=>1 for i=1:J])
	is_done = [false for i=1:J]
	while any(is_done .== false)
		for j=1:J
			group_assignments = groups_tables_assignments[j]
			i = cur_group_table[j]
			if i > length(groups_tables_counts[j])
				is_done[j] = true
				continue
			end
			table_assignment = single_crp_draw(global_tables_count,γ)
			if table_assignment == length(global_tables_count)+1
				push!(global_tables_count,1)
			else
				global_tables_count[table_assignment] += 1
			end
			push!(group_assignments,table_assignment)
			groups_tables_assignments[j] = group_assignments
			cur_group_table[j]+=1
		end
	end
	global_mixture = [x / sum(global_tables_count) for x in global_tables_count]
	groups_mixtures = Dict()
	for j=1:J
		local_mixture = zeros(length(global_mixture))
		for i = 1:length(global_mixture)
			if i in groups_tables_assignments[j]
				local_mixture[i] = sum(groups_tables_counts[j][groups_tables_assignments[j] .== i])
			end
		end
		groups_mixtures[j] = local_mixture
	end
	return global_mixture,groups_mixtures,groups_tables_counts,groups_tables_assignments
end




function generate_grouped_gaussian_from_hdp_group_counts(group_counts, dim, var_size)
	pts_dict = Dict()
	labels_dict = Dict()
	K = length(group_counts[1])
	tmean = zeros(dim,K)
	tcov = zeros(dim,dim,K)
	J = length(group_counts)
	components = []
	for i=1:length(group_counts[1])
		tmean[:,i] .= rand(MvNormal(zeros(dim), var_size*Matrix{Float64}(I, dim, dim)))
		tcov[:,:,i] .= rand(InverseWishart(dim+2, Matrix{Float64}(I, dim, dim)))
		push!(components,MvNormal(tmean[:,i], tcov[:,:,i]))
	end
	for j=1:J
		points = reduce(hcat,[rand(components[i],Int(group_counts[j][i])) for i=1:length(group_counts[j])])
		labels = reduce(vcat,[Int.(ones(Int(group_counts[j][i])))*i for i=1:length(group_counts[j])])
		pts_dict[j] = points
		labels_dict[j] = labels
	end
	return pts_dict, labels_dict
end
