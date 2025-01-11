module FeatureExtraction

using DataFrames
using Statistics
using TextAnalysis
using LinearAlgebra


export FeatureExtractionTransformer, extract_feature, tokenize, get_vocabulary, bag_of_words, pca

import ..TransformerModule: Transformer, fit!, transform

mutable struct FeatureExtractionTransformer <: Transformer
    strategy::String
    transformed_data::DataFrame

    function FeatureExtractionTransformer(strategy::String)
        if !(strategy in ["bow", "pca", "basic"])
            throw(ArgumentError("Unknown strategy: $strategy. Supported strategies are: 'bow', 'pca', 'basic'"))
        end
        if strategy == "constant" && constant_value === nothing
            throw(ArgumentError("constant_value must be provided when using 'constant' strategy"))
        end
        new(strategy, Float64[], constant_value)
    end
end

function fit!(transformer::FeatureExtractionTransformer, data::Any)

end

function transform(transformer::FeatureExtractionTransformer, data::Matrix{Any})
    if typeof(data) == Matrix{Any} || typeof(data) == Matrix{String}
        data = string.(data)  
        data = vcat(data...) 
    end
end

function transform(transformer::FeatureExtractionTransformer, data::Any)
    if transformer.strategy == "bow"
        transformer.transformed_data = DataFrame(extract_feature(data))
    elseif transformer.strategy == "pca"
        transformer.transformed_data = DataFrame(extract_feature(data))
    elseif transformer.strategy == "basic"
        transformer.transformed_data = DataFrame(extract_feature(data))
    end

end

# extract the feature
function extract_feature(text_data::Vector{String})
    
    # Implementation of Bag-of-Words (without stop words) in Julia for extraction of text data (c.f. Scikit Learn data transforms section 6.2)
    text_tokens = tokenize(text_data)
    text_vocab = get_vocabulary(text_tokens)
    bow_vector = [bag_of_words(token, text_vocab) for token in text_tokens]
    return bow_vector
    # text_bow = bag_of_words(text_data, text_vocab)
end

# Helpers for Bag-of-Words feature extraction 
function tokenize(text_data::Vector{String})
    
    # TODO: Docstring Documentation
    text_data = lowercase.(text_data)
    text_data = replace.(text_data, r"[^\w\s]" => "")
    text_tokens = [split(item) for item in text_data]
    return text_tokens
end

function get_vocabulary(tokenized_data::Vector{String})
    # TODO: Docstring Documentation
    vocabulary = unique(vcat(tokenized_data...))
    return vocabulary
end

function bag_of_words(text_data::Vector{String}, vocabulary::Vector{String})
    # TODO: Docstring Documentation
    word_count = Dict(word => count(==(word), text_data) for word in vocabulary)
    return [word_count[word] for word in vocabulary]
end

function extract_feature(tabular_data)
    # Implementation of PCA in Julia for extraction of tabular data (c.f. Scikit Learn data transforms section 6.2)
    # TODO: Docstring Documentation
    principal_components = pca(tabular_data)
    return principal_components

end 

function pca(X, k=2)  
    # PCA Implementation from HW2
    # TODO: Docstring Documentation
	μ = mean(X, dims = 2)
	X_centered = X .-μ
	C = X_centered * X_centered' / (size(X, 2) - 1)
	eigenvals, eigenvecs = eigen(C)
	sorted = sortperm(eigenvals, rev=true)

    Wk = eigenvecs[:, sorted[1:k]]

    H = Wk' * X_centered
    return H 
end

function extract_feature(data::DataFrame; target_column::Union{Symbol, Nothing}=nothing, variance_threshold::Float64=0.01,cardinality_threshold::Int=2,correlation_threshold::Float64=0.2)
    # TODO: Docstring Documentation
    filtered_data = variance_filter(data, variance_threshold)
    filtered_data = low_cardinality_filter(filtered_data, cardinality_threshold)
    if target_column !== nothing
        filtered_data = correlation_filter(filtered_data, target_column, correlation_threshold)
    end
    return filtered_data
end

function filter_variance(data::DataFrame, threshold::Float64)
    # TODO: Docstring Documentation
    variances = map(col -> var(data[:, col]), names(data))
    selected_columns = names(data)[variances .> threshold]
    return select(data, selected_columns)
end

function filter_correlation(data::DataFrame, target_column::Symbol, threshold::Float64)
    # TODO: Docstring Documentation
    target = data[:, target_column]
    correlations = Dict(col => cor(data[:, col], target) for col in names(data) if col != target_column)
    selected_columns = filter(col -> abs(correlations[col]) > threshold, keys(correlations))
    return select(data, [target_column; selected_columns])
end

function filter_low_cardinality(data::DataFrame)
    # TODO: Docstring Documentation
    cardinalities = map(col -> length(unique(data[:, col])), names(data))
    selected_columns = names(data)[cardinalities .> threshold]
    return select(data, selected_columns)
end

end