"""
    DictVectorizer

A vectorizer that converts dictionaries of features into a matrix representation.

# Fields
- `feature_names::Vector{String}`: The list of feature names extracted from the input dictionaries.

# Examples
```julia
vectorizer = DictVectorizer()
```
"""
mutable struct DictVectorizer <: BaseRawExtractor
    feature_names::Vector{String}

    DictVectorizer() = new([])
end

"""
    fit!(dv::DictVectorizer, dicts::Vector{Dict{String, Any}})

Fits the DictVectorizer to extract feature names from the given dictionaries.

# Arguments
- `dv::DictVectorizer`: The DictVectorizer instance.
- `dicts::Vector{Dict{String, Any}}`: A list of dictionaries representing data samples.

# Returns
- `DictVectorizer`: The updated instance with extracted feature names.
"""
function fit!(dv::DictVectorizer, dicts::Vector{Dict{String, Any}})
    feature_set = Set{String}()
    categorical_features = Dict{String, Set{String}}()

    # Iterate through each dictionary in the dataset
    for dict in dicts
        for (key, value) in dict
            push!(feature_set, key)

            if value isa String
                if !haskey(categorical_features, key)
                    categorical_features[key] = Set{String}()
                end
                push!(categorical_features[key], value)
            end
        end
    end

    feature_list = String[]

    # Add categorical features first
    for key in sort(collect(keys(categorical_features)))
        categories = categorical_features[key]
        for category in sort(collect(categorical_features[key]))
            push!(feature_list, "$(key)=$(category)")
        end
    end

    # Add numeric features
    for key in sort(collect(setdiff(feature_set, keys(categorical_features))))
        push!(feature_list, key)
    end

    dv.feature_names = feature_list
    return dv
end


"""
    transform(dv::DictVectorizer, dicts::Vector{Dict{String, Any}})

Transforms the given dictionaries into a numerical feature matrix.

# Arguments
- `dv::DictVectorizer`: The DictVectorizer instance.
- `dicts::Vector{Dict{String, Any}}`: A list of dictionaries representing data samples.

# Returns
- `Matrix{Float64}`: The transformed feature matrix.
"""
function transform(dv::DictVectorizer, dicts::AbstractArray) # dicts::Vector{Dict{String, Any}})
    n_samples = length(dicts)
    n_features = length(dv.feature_names)

    X = zeros(Float64, n_samples, n_features)

    for (i, dict) in enumerate(dicts)
        for (key, value) in dict
            feature_key = value isa String ? "$(key)=$(value)" : key

            feature_index = findfirst(==(feature_key), dv.feature_names)
            if feature_index !== nothing
                X[i, feature_index] = value isa Number ? value : 1.0
            end
        end
    end

    return X  # Stellt sicher, dass X eine (n_samples, n_features) Matrix bleibt
end




"""
    fit_transform!(dv::DictVectorizer, dicts::Vector{Dict{String, Any}})

Fits the DictVectorizer and transforms the given dictionaries into a feature matrix.

# Arguments
- `dv::DictVectorizer`: The DictVectorizer instance.
- `dicts::Vector{Dict{String, Any}}`: A list of dictionaries representing data samples.

# Returns
- `Matrix{Float64}`: The transformed feature matrix.
"""
function fit_transform!(dv::DictVectorizer, dicts::Vector{Dict{String, Any}})
    fit!(dv, dicts)
    return transform(dv, dicts)
end
