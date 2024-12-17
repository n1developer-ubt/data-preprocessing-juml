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
    """
    Extract features from a vector of strings using the Bag-of-Words (BoW) model.

    The Bag-of-Words model tokenizes the input text, constructs a vocabulary, and creates
    a vector representation for each input string based on the vocabulary. Stop word removal
    is not performed in this implementation.

    # Arguments
    - `text_data::Vector{String}`: A vector of strings containing the text data to be processed.

    # Returns
    A vector of BoW vectors, where each inner vector represents the BoW representation
    of the corresponding input string.
    """
    
    # Implementation of Bag-of-Words (without stop words) in Julia for extraction of text data (c.f. Scikit Learn data transforms section 6.2)
    text_tokens = tokenize(text_data)
    text_vocab = get_vocabulary(text_tokens)
    bow_vector = [bag_of_words(token, text_vocab) for token in text_tokens]
    return bow_vector
    # text_bow = bag_of_words(text_data, text_vocab)
end

# Helpers for Bag-of-Words feature extraction 
function tokenize(text_data::Vector{String})
    """
    Tokenize a vector of strings into a vector of word lists.

    This function converts all input text to lowercase, removes non-alphanumeric
    characters (except spaces), and splits each string into individual words.

    # Arguments
    - `text_data::Vector{String}`: A vector of strings containing the text data to be tokenized.

    # Returns
    A vector of vectors, where each inner vector contains the tokens (words) of the corresponding
    input string.
    """
    # TODO: Docstring Documentation
    text_data = lowercase.(text_data)
    text_data = replace.(text_data, r"[^\w\s]" => "")
    text_tokens = [split(item) for item in text_data]
    return text_tokens
end


function get_vocabulary(tokenized_data::Vector{String})
    """
        get_vocabulary(tokenized_data::Vector{String})

    Generate a list of unique tokens from the input tokenized data. The function flattens
    the tokenized data and returns a vocabulary containing each unique token.

    # Arguments
    - tokenized_data::Vector{String}: A vector of strings, where each string represents
    tokenized data (e.g., a word or sequence of tokens).

    # Returns
    A vector of unique tokens found in the tokenized data.
    """
    # TODO: Docstring Documentation
    vocabulary = unique(vcat(tokenized_data...))
    return vocabulary
end

function bag_of_words(text_data::Vector{String}, vocabulary::Vector{String})
    """
        bag_of_words(text_data::Vector{String}, vocabulary::Vector{String})

    Generate a bag-of-words representation of the input text data based on a given vocabulary.
    For each word in the vocabulary, the function counts its occurrences in the `text_data` and
    returns a vector with the corresponding word counts.

    # Arguments
    - text_data::Vector{String}: A vector of strings where each string represents a document
    or text entry, and the function will count word occurrences in this data.
    - vocabulary::Vector{String}: A vector of unique words representing the vocabulary for which
    the word counts will be calculated.

    # Returns
    A vector of word counts corresponding to the words in the vocabulary.
    """
    # TODO: Docstring Documentation
    word_count = Dict(word => count(==(word), text_data) for word in vocabulary)
    return [word_count[word] for word in vocabulary]
end

function extract_feature(tabular_data)
    # Implementation of PCA in Julia for extraction of tabular data (c.f. Scikit Learn data transforms section 6.2)
    # TODO: Docstring Documentation
    """
        extract_feature(tabular_data)

    Extract features from tabular data using Principal Component Analysis (PCA).
    This function applies PCA to reduce the dimensionality of the input data and returns
    the principal components.

    # Arguments
    - tabular_data: A matrix or DataFrame representing the input tabular data. Each row
    corresponds to an observation, and each column represents a feature.

    # Returns
    A matrix representing the principal components of the input data after applying PCA.
    """
    principal_components = pca(tabular_data)
    return principal_components

end 

function pca(X, k=2)  
    # PCA Implementation from HW2
    # TODO: Docstring Documentation
    """
        pca(X, k=2)

    Perform Principal Component Analysis (PCA) on the input data `X` and return the
    top `k` principal components. This function centers the data, computes the covariance
    matrix, and extracts the eigenvectors corresponding to the largest eigenvalues.

    # Arguments
    - X: A matrix of size `(m, n)`, where `m` is the number of features (rows) and `n` is
    the number of observations (columns). Each column represents an observation, and each
    row represents a feature.
    - k: The number of principal components to return. Default is 2.

    # Returns
    A matrix `H` of size `(k, n)` representing the projections of the input data `X` onto
    the top `k` principal components. Each row of `H` corresponds to a principal component,
    and each column corresponds to a data point.
    """
	μ = mean(X, dims = 2)
	X_centered = X .-μ
	C = X_centered * X_centered' / (size(X, 2) - 1)
	eigenvals, eigenvecs = eigen(C)
	sorted = sortperm(eigenvals, rev=true)

    Wk = eigenvecs[:, sorted[1:k]]

    H = Wk' * X_centered
    return H 
end

function extract_feature(data::DataFrames; target_column::Union{Symbol, Nothing}=nothing, variance_threshold::Float64=0.01,cardinality_threshold::Int=2,correlation_threshold::Float64=0.2)
    """
        extract_feature(data::DataFrame; target_column::Union{Symbol, Nothing}=nothing, 
                        variance_threshold::Float64=0.01, cardinality_threshold::Int=2, 
                        correlation_threshold::Float64=0.2)

    Extract relevant features from the input `DataFrame` by applying multiple filters:
    1. Variance thresholding to remove features with low variance.
    2. Low cardinality filtering to remove categorical features with too few distinct values.
    3. Correlation filtering (if `target_column` is provided) to remove features highly correlated with the target variable.

    # Arguments
    - data::DataFrame: The input data, which should be a `DataFrame` containing the features (columns) to be filtered.
    - target_column::Union{Symbol, Nothing}: The name of the target column (optional). If provided, the function will remove features highly correlated with this target. Default is `nothing`.
    - variance_threshold::Float64: The minimum variance required for a feature to be retained. Features with variance below this threshold will be removed. Default is `0.01`.
    - cardinality_threshold::Int: The maximum number of unique categories allowed for a categorical feature to be retained. Features with cardinality above this threshold will be removed. Default is `2`.
    - correlation_threshold::Float64: The maximum correlation allowed between features and the target variable. Features with correlation above this threshold will be removed. Default is `0.2`.

    # Returns
    A `DataFrame` with the features filtered according to the specified thresholds.
    """ 
    # TODO: Docstring Documentation
    filtered_data = variance_filter(data, variance_threshold)
    filtered_data = low_cardinality_filter(filtered_data, cardinality_threshold)
    if target_column !== nothing
        filtered_data = correlation_filter(filtered_data, target_column, correlation_threshold)
    end
    return filtered_data
end

function filter_variance(data::DataFrames, threshold::Float64)
    # TODO: Docstring Documentation
    """
        filter_variance(data::DataFrame, threshold::Float64)

    Filters the columns of the input `DataFrame` based on the variance. Only columns with a variance 
    greater than the specified `threshold` are retained.

    # Arguments
    - data::DataFrame: The input `DataFrame` containing the features to be filtered.
    - threshold::Float64: The minimum variance required for a feature to be retained. Columns with 
    variance below this threshold will be removed.

    # Returns
    A `DataFrame` containing only the columns with variance greater than the specified `threshold`.
    """
    variances = map(col -> var(data[:, col]), names(data))
    selected_columns = names(data)[variances .> threshold]
    return select(data, selected_columns)
end

function filter_correlation(data::DataFrames, target_column::Symbol, threshold::Float64)
    """
        filter_correlation(data::DataFrame, target_column::Symbol, threshold::Float64)

    Filters the columns of the input `DataFrame` based on their correlation with the specified 
    target column. Only columns with an absolute correlation greater than the specified 
    `threshold` are retained.

    # Arguments
    - data::DataFrame: The input `DataFrame` containing the features and the target column.
    - target_column::Symbol: The name of the target column. This column will be excluded from 
    correlation calculation but retained in the final result.
    - threshold::Float64: The minimum absolute correlation required for a feature to be retained. 
    Columns with an absolute correlation below this threshold will be removed.

    # Returns
    A `DataFrame` containing the target column and only those columns that have an absolute 
    correlation greater than the specified `threshold` with the target column.
    """ 
    # TODO: Docstring Documentation
    target = data[:, target_column]
    correlations = Dict(col => cor(data[:, col], target) for col in names(data) if col != target_column)
    selected_columns = filter(col -> abs(correlations[col]) > threshold, keys(correlations))
    return select(data, [target_column; selected_columns])
end

function filter_low_cardinality(data::DataFrames)
    """
    filter_low_cardinality(data::DataFrame)

        Filters the columns of the input `DataFrame` based on the cardinality (the number of unique 
        values) in each column. Columns with cardinality below a certain threshold are removed.

        # Arguments
        - data::DataFrame: The input `DataFrame` containing the features to be filtered.

        # Returns
        A `DataFrame` containing only the columns with cardinality greater than the specified threshold.
    """
    # TODO: Docstring Documentation
    cardinalities = map(col -> length(unique(data[:, col])), names(data))
    selected_columns = names(data)[cardinalities .> threshold]
    return select(data, selected_columns)
end

end