import ...TransformerModule: fit!, transform, fit_transform!, inverse_transform

"""
    OneHotEncoder

    One Hot Encoder is an encoder that encodes the data using the one hot encoding technique.

    # Arguments
    - `categories::Union{Vector{String}, Vector{Vector{String}}, Nothing}`: The categories to encode. It can be one of the following:
        - `Vector{String}`: The categories to encode.
        - `Vector{Vector{String}}`: The categories to encode for each feature (if the data has multiple features).
        - `Nothing`: The categories will be inferred from the data.
    - `handle_unknown::String`: The strategy to handle unknown categories. It can be one of the following:
        - "error": Raise an error.
        - "ignore": Ignore the unknown categories.
"""
mutable struct OneHotEncoder <: BaseEncoder
    categories::Union{Vector{String}, Nothing}
    handle_unknown::String

    OneHotEncoder() = new(nothing, "error")
    OneHotEncoder(categories::Union{Vector{String}, Vector{Vector{String}}, Nothing}, handle_unknown::String) = new(categories, handle_unknown)
end

"""
    fit!(encoder::OneHotEncoder, X::Vector{<:String})

    Fit the encoder to the data.

    # Arguments
    - `encoder::OneHotEncoder`: An instance of `OneHotEncoder`.
    - `X::Vector{<:String}`: The data to fit the encoder.

    # Returns
    The fitted `OneHotEncoder`.
"""
function fit!(encoder::OneHotEncoder, X::Vector{<:String}, y::Vector{Any} = [])
    if encoder.categories === nothing
        encoder.categories = unique(X)
    end
    return encoder
end

"""
    transform(encoder::OneHotEncoder, X::Vector{<:String})

    Transform the data using the encoder.

    # Arguments
    - `encoder::OneHotEncoder`: An instance of `OneHotEncoder`.
    - `X::Vector{<:String}`: The data to transform.

    # Returns
    The transformed data.
"""
function transform(encoder::OneHotEncoder, X::Vector{<:String})
    cats = encoder.categories

    if encoder.handle_unknown == "error"
        unseen_cats = setdiff(unique(X), cats)

        if length(unseen_cats) > 0
            throw(ArgumentError("Unseen categories: $unseen_cats"))
        end
    end

    one_hot_matrix = zeros(Int, length(X), length(cats))

    for (i, category) in enumerate(cats)
        one_hot_matrix[:, i] = X .== category
    end

    return one_hot_matrix
end

"""
    inverse_transform!(encoder::OneHotEncoder, X::Matrix{<:Real})

    Inverse transform the data using the encoder.

    # Arguments
    - `encoder::OneHotEncoder`: An instance of `OneHotEncoder`.
    - `X::Matrix{<:Real}`: The data to inverse transform.

    # Returns
    The inverse transformed data.
"""
function inverse_transform(encoder::OneHotEncoder, X::Matrix{<:Real})
    if size(X, 2) != length(encoder.categories)
        throw(ArgumentError("Number of columns in X does not match the number of categories in the encoder"))
    end

    data = []

    # for loop on each row, 
    for i in 1:size(X, 1)
        for j in 1:size(X, 2)
            if X[i, j] == 1
                cat = encoder.categories[j]
                push!(data, cat)
            end
        end
    end

    return data
end

"""
    fit_transform!(encoder::OneHotEncoder, X::Vector{<:String}, y::Vector{Any})

    Fit the encoder to the data and transform the data.

    # Arguments
    - `encoder::OneHotEncoder`: An instance of `OneHotEncoder`.
    - `X::Vector{<:String}`: The data to fit and transform.
    - `y::Vector{Any}`: The target data.

    # Returns
    The transformed data.
"""
function fit_transform!(encoder::OneHotEncoder, X::Vector{<:String}, y::Vector{Any})
    fit!(encoder, X, y)
    return transform!(encoder, X)
end