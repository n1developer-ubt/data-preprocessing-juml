include("base_normalizer.jl")
using LinearAlgebra

import ...TransformerModule: fit!, transform, inverse_transform

mutable struct StandardNormalizer <: BaseNormalizer
    type::String
    norm::Union{Vector{Float64}, Float64, Nothing}

    StandardNormalizer() = new("l2", nothing)
    StandardNormalizer(type::String) = new(type, nothing)
end

"""
    fit!(normalizer::StandardNormalizer, X::Vector{<:Real})

    Fit the normalizer to the data.

    # Arguments
    - `normalizer::StandardNormalizer`: An instance of `StandardNormalizer`.
    - `X::Vector{<:Real}`: The data to fit the normalizer.

    # Returns
    The fitted `StandardNormalizer`.
"""
function fit!(normalizer::StandardNormalizer, X::Vector{<:Real}, y::Vector{Any} = [])
    if normalizer.type == "l1"
        normalizer.norm = norm(X, 1)
    elseif normalizer.type == "l2"
        normalizer.norm = norm(X, 2)
    elseif normalizer.type == "max"
        normalizer.norm = norm(X, Inf)
    else
        throw(ArgumentError("Invalid norm type."))
    end
    return normalizer
end

"""
    fit!(normalizer::StandardNormalizer, X::Matrix{<:Real})

    Fit the normalizer to the data.

    # Arguments
    - `normalizer::StandardNormalizer`: An instance of `StandardNormalizer`.
    - `X::Matrix{<:Real}`: The data to fit the normalizer.

    # Returns
    The fitted `StandardNormalizer`.
"""
function fit!(normalizer::StandardNormalizer, X::Matrix{<:Real}, y::Vector{Any} = [])
    if normalizer.type == "l1"
        normalizer.norm = [norm(row, 1) for row in eachrow(X)]
    elseif normalizer.type == "l2"
        normalizer.norm = [norm(row, 2) for row in eachrow(X)]
    elseif normalizer.type == "max"
        normalizer.norm = [norm(row, Inf) for row in eachrow(X)]
    else
        throw(ArgumentError("Invalid norm type."))
    end
    return normalizer
end

"""
    transform(normalizer::StandardNormalizer, X::Vector{<:Real})

    Transform the data using the fitted normalizer.

    # Arguments
    - `normalizer::StandardNormalizer`: An instance of `StandardNormalizer`.
    - `X::Vector{<:Real}`: The data to transform.

    # Returns
    The transformed data.
"""
function transform(normalizer::StandardNormalizer, X::Vector{<:Real}, y::Vector{Any} = [])
    if(isnothing(normalizer.norm))
        throw(ArgumentError("StandardNormalizer not fitted yet"))
    end

    return X ./ normalizer.norm
end

"""
    transform(normalizer::StandardNormalizer, X::Matrix{<:Real})

    Transform the data using the fitted normalizer.

    # Arguments
    - `normalizer::StandardNormalizer`: An instance of `StandardNormalizer`.
    - `X::Matrix{<:Real}`: The data to transform.

    # Returns
    The transformed data.
"""
function transform(normalizer::StandardNormalizer, X::Matrix{<:Real}, y::Vector{Any} = [])
    if(isnothing(normalizer.norm))
        throw(ArgumentError("StandardNormalizer not fitted yet"))
    end

    return X ./ normalizer.norm
end

"""
    inverse_transform(normalizer::StandardNormalizer, X::Vector{<:Real})

    Inverse transform the data using the fitted normalizer.

    # Arguments
    - `normalizer::StandardNormalizer`: An instance of `StandardNormalizer`.
    - `X::Vector{<:Real}`: The data to inverse transform.

    # Returns
    The inverse transformed data.
"""
function inverse_transform(normalizer::StandardNormalizer, X::Vector{<:Real}, y::Vector{Any} = [])
    if(isnothing(normalizer.norm))
        throw(ArgumentError("StandardNormalizer not fitted yet"))
    end

    return X .* normalizer.norm
end

"""
    inverse_transform(normalizer::StandardNormalizer, X::Matrix{<:Real})

    Inverse transform the data using the fitted normalizer.

    # Arguments
    - `normalizer::StandardNormalizer`: An instance of `StandardNormalizer`.
    - `X::Matrix{<:Real}`: The data to inverse transform.

    # Returns
    The inverse transformed data.
"""
function inverse_transform(normalizer::StandardNormalizer, X::Matrix{<:Real}, y::Vector{Any} = [])
    if(isnothing(normalizer.norm))
        throw(ArgumentError("StandardNormalizer not fitted yet"))
    end

    return X .* normalizer.norm
end

"""
    fit_transform!(normalizer::StandardNormalizer, X::Vector{<:Real})

    Fit and transform the data using the normalizer.

    # Arguments
    - `normalizer::StandardNormalizer`: An instance of `StandardNormalizer`.
    - `X::Vector{<:Real}`: The data to fit and transform.

    # Returns
    The transformed data.
"""
function fit_transform!(normalizer::StandardNormalizer, X::Vector{<:Real}, y::Vector{Any} = [])
    fit!(normalizer, X)
    return transform(normalizer, X)
end

"""
    fit_transform!(normalizer::StandardNormalizer, X::Matrix{<:Real})

    Fit and transform the data using the normalizer.

    # Arguments
    - `normalizer::StandardNormalizer`: An instance of `StandardNormalizer`.
    - `X::Matrix{<:Real}`: The data to fit and transform.

    # Returns
    The transformed data.
"""
function fit_transform!(normalizer::StandardNormalizer, X::Matrix{<:Real}, y::Vector{Any} = [])
    fit!(normalizer, X)
    return transform(normalizer, X)
end