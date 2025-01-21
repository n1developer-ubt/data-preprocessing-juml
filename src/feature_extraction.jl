module FeatureExtraction

using Statistics
using LinearAlgebra
using TextAnalysis

export FeatureExtractionTransformer, extract_feature, tokenize, get_vocabulary, bag_of_words, pca, generate_ngrams

import ..TransformerModule: Transformer, fit!, transform

mutable struct FeatureExtractionTransformer <: Transformer
    strategy::String
    transformed_data::Matrix{Float64}

    """
        FeatureExtractionTransformer(strategy::String)

    Initialize the feature extraction transformer with a given strategy.

    # Arguments
    - `strategy::String`: The transformation strategy to apply. Supported strategies: 'bow', 'pca', 'basic'.

    # Example
    ```julia
    transformer = FeatureExtractionTransformer("bow")
    ```
    """
    function FeatureExtractionTransformer(strategy::String)
        if !(strategy in ["bow", "pca", "basic", "tfidf", "ngrams", "flatten_image"])
            throw(ArgumentError("Unknown strategy: $strategy. Supported strategies are: 'bow', 'pca', 'basic', 'tfidf', 'ngrams', 'flatten_image'"))
        end
        new(strategy, zeros(Float64, 0, 0))  # Initialize with an empty matrix
    end
end

"""
    fit!(transformer::FeatureExtractionTransformer, data::Any)

Placeholder for fitting the transformer. Currently not implemented.

# Arguments
- `transformer::FeatureExtractionTransformer`: The transformer instance.
- `data::Any`: The input data to fit the transformer.
"""
function fit!(transformer::FeatureExtractionTransformer, data::Any)
    
end

"""
transform(transformer::FeatureExtractionTransformer, data::Any)

Transform the input data based on the transformer's strategy.

# Arguments
- `transformer::FeatureExtractionTransformer`: The transformer instance.
- `data::Any`: Input data to be transformed.

# Returns
- `Matrix{Float64}`: The transformed data.
"""
function transform(transformer::FeatureExtractionTransformer, data::Any)
    if transformer.strategy == "bow"
        transformer.transformed_data = extract_feature_bow(data)
    elseif transformer.strategy == "pca"
        transformer.transformed_data = extract_feature_pca(data)
    elseif transformer.strategy == "basic"
        transformer.transformed_data = data  # Basic logic: no transformation
    elseif transformer.strategy == "tfidf"
        bow_matrix = extract_feature_bow(data)
        transformer.transformed_data = compute_tfidf(bow_matrix)
    elseif transformer.strategy == "ngrams"
        ngrams_matrix = generate_ngrams(data, 2)  # Example with bigrams
        transformer.transformed_data = ngrams_matrix  # Conversion to matrix not required here
    elseif transformer.strategy == "flatten_image"
        transformer.transformed_data = flatten_image(data) |> reshape(:, 1)  # Ensure matrix form
    else
        throw(ArgumentError("Unknown strategy: $(transformer.strategy)"))
    end
end


# Text data feature extraction
"""
    extract_feature_bow(text_data::Vector{String})

Extract Bag-of-Words features from a vector of strings.

# Arguments
- `text_data::Vector{String}`: A vector of strings to process.

# Returns
- `Matrix{Float64}`: Bag-of-Words feature matrix.
"""
function extract_feature_bow(text_data::Vector{String})
    text_tokens = tokenize(text_data)
    text_vocab = get_vocabulary(text_tokens)
    bow_matrix = hcat([bag_of_words(token, text_vocab) for token in text_tokens]...)'  # Convert to matrix
    return bow_matrix
end

"""
    tokenize(text_data::Vector{String})

Tokenize a vector of strings into lowercase words without special characters.

# Arguments
- `text_data::Vector{String}`: A vector of strings to tokenize.

# Returns
- `Vector{Vector{String}}`: Tokenized text data.
"""
function tokenize(text_data::Vector{String})
    text_data = lowercase.(text_data)
    text_data = replace.(text_data, r"[^\w\s]" => "")
    text_tokens = [split(item) for item in text_data]
    return text_tokens
end

"""
    get_vocabulary(tokenized_data::Vector{Vector{String}})

Get the unique vocabulary from tokenized data.

# Arguments
- `tokenized_data::Vector{Vector{String}}`: Tokenized text data.

# Returns
- `Vector{String}`: Unique vocabulary from the tokenized data.
"""
function get_vocabulary(tokenized_data::Vector{Vector{String}})
    vocabulary = unique(vcat(tokenized_data...))
    return vocabulary
end

"""
    bag_of_words(text_data::Vector{String}, vocabulary::Vector{String})

Generate a Bag-of-Words vector for a given text based on the vocabulary.

# Arguments
- `text_data::Vector{String}`: Tokenized text data.
- `vocabulary::Vector{String}`: Vocabulary to use for the Bag-of-Words.

# Returns
- `Vector{Float64}`: Bag-of-Words feature vector.
"""
function bag_of_words(text_data::Vector{String}, vocabulary::Vector{String})
    word_count = Dict(word => count(==(word), text_data) for word in vocabulary)
    return [word_count[word] for word in vocabulary]
end


"""
generate_ngrams(text_data::Vector{String}, n::Int)

Generate n-grams from a vector of strings.

# Arguments
- `text_data::Vector{String}`: A vector of strings to process.
- `n::Int`: The size of the n-grams to generate.

# Returns
- `Vector{Vector{String}}`: A vector of n-grams for each string.
"""
function generate_ngrams(text_data::Vector{String}, n::Int)
    tokenized = tokenize(text_data)
    ngram_list = [join(token[i:i+n-1], " ") for token in tokenized for i in 1:(length(token) - n + 1) if length(token) >= n]
    return ngram_list
end


"""
generate_ngrams(text_data::Vector{String}, n::Int)

Generate n-grams from a vector of strings.

# Arguments
- `text_data::Vector{String}`: A vector of strings to process.
- `n::Int`: The size of the n-grams to generate.

# Returns
- `Vector{Vector{String}}`: A vector of n-grams for each string.
"""
function generate_ngrams(text_data::Vector{String}, n::Int)
    tokenized = tokenize(text_data)
    ngram_list = [join(token[i:i+n-1], " ") for token in tokenized for i in 1:(length(token) - n + 1) if length(token) >= n]
    return ngram_list
end

# Numerical data feature extraction
"""
    extract_feature_pca(X::Matrix{Float64}, k::Int=2)

Perform PCA to reduce the dimensionality of the input matrix.

# Arguments
- `X::Matrix{Float64}`: Input matrix for dimensionality reduction.
- `k::Int=2`: Number of principal components to retain.

# Returns
- `Matrix{Float64}`: The transformed data in reduced dimensions.
"""
function extract_feature_pca(X::Matrix{Float64}, k::Int=2)
    μ = mean(X, dims=2)
    X_centered = X .- μ
    C = X_centered * X_centered' / (size(X, 2) - 1)
    eigenvals, eigenvecs = eigen(C)
    sorted = sortperm(eigenvals, rev=true)
    Wk = eigenvecs[:, sorted[1:k]]
    H = Wk' * X_centered
    return H
end

# Picture Data
"""
flatten_image(image::AbstractArray)

Flatten an image into a 1D array of pixel values.

# Arguments
- `image::AbstractArray`: The input image array.

# Returns
- `Vector{Float64}`: Flattened 1D array of pixel values.
"""
function flatten_image(image::AbstractArray)
    return Float64.(reshape(image, :))
end

# Other feature extraction methods

"""
    filter_variance(data::Matrix{Float64}, threshold::Float64)

Filter columns of the matrix based on variance threshold.

# Arguments
- `data::Matrix{Float64}`: Input matrix to filter.
- `threshold::Float64`: Variance threshold for column selection.

# Returns
- `Matrix{Float64}`: Filtered matrix with selected columns.
"""
function filter_variance(data::Matrix{Float64}, threshold::Float64)
    variances = var(data, dims=1)
    selected_columns = findall(variances .> threshold)
    return data[:, selected_columns]
end

"""
    filter_correlation(data::Matrix{Float64}, target::Vector{Float64}, threshold::Float64)

Filter columns based on their correlation with a target vector.

# Arguments
- `data::Matrix{Float64}`: Input matrix to filter.
- `target::Vector{Float64}`: Target vector for correlation.
- `threshold::Float64`: Correlation threshold for column selection.

# Returns
- `Matrix{Float64}`: Filtered matrix with selected columns.
"""
function filter_correlation(data::Matrix{Float64}, target::Vector{Float64}, threshold::Float64)
    correlations = [cor(data[:, col], target) for col in 1:size(data, 2)]
    selected_columns = findall(abs.(correlations) .> threshold)
    return data[:, selected_columns]
end

"""
    filter_low_cardinality(data::Matrix{Int}, threshold::Int)

Filter columns with low cardinality based on a given threshold.

# Arguments
- `data::Matrix{Int}`: Input matrix with integer values.
- `threshold::Int`: Minimum cardinality for column selection.

# Returns
- `Matrix{Int}`: Filtered matrix with selected columns.
"""
function filter_low_cardinality(data::Matrix{Int}, threshold::Int)
    cardinalities = [length(unique(data[:, col])) for col in 1:size(data, 2)]
    selected_columns = findall(cardinalities .> threshold)
    return data[:, selected_columns]
end

end



