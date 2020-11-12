using VersatileHDPMixtureModels

nips_path = "docword.nips.txt"


function text_file_to_dict(path)
	data_dict = Dict()
	cur_doc = 0
	open(path) do file
		cur_vec =[]
		for ln in eachline(file)
			items = [parse(Int64,x) for x in split(ln)]
			if length(items) < 3
				continue
			end
			if cur_doc != items[1]
				data_dict[cur_doc] = Float32.(cur_vec)[:,:]'
				cur_vec = []
				cur_doc = items[1]
			end
			for i=1:items[3]
				push!(cur_vec,items[2])
			end
		end
		data_dict[cur_doc] = Float32.(cur_vec)[:,:]'
	end

	delete!(data_dict,0)
	return data_dict
end



nips_data = text_file_to_dict(nips_path)
nips_gprior = topic_modeling_hyper(Float64.(ones(12419))*0.1)
model = hdp_fit(nips_data,1.0,1.0,1.0,nips_gprior,100,10,15)
