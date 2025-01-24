using Statistics
import ...TransformerModule: fit!, transform, inverse_transform

"""
    struct MinMaxScaler

A structure representing a min-max scaler.
The min-max scaler scales features to a given range.
The range is defined by the `feature_range` parameter.

# Fields
- `feature_range::Tuple{<:Real, <:Real}`: The range to scale the features to.
- `min::Union{Vector{<:Real}, <:Real, Nothing}`: Minimum value of the features.
- `max::Union{Vector{<:Real}, <:Real, Nothing}`: Maximum value of the features.

"""
mutable struct MinMaxScaler <: BaseScaler
    feature_range::Tuple{<:Real, <:Real}
    min::Union{Vector{<:Real}, <:Real, Nothing}
    max::Union{Vector{<:Real}, <:Real, Nothing}

    MinMaxScaler(feature_range::Tuple{<:Real, <:Real}) = new(feature_range, nothing, nothing)
end

"""
    fit!(scaler::MinMaxScaler, X::Vector{<:Real})

Fit the min-max scaler to the data.

# Arguments
- `scaler::MinMaxScaler`: An instance of `MinMaxScaler`.
- `X::Vector{<:Real}`: The data to fit the scaler.

# Returns
The fitted `MinMaxScaler`.
"""
function fit!(scaler::MinMaxScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    scaler.min = minimum(X)
    scaler.max = maximum(X)
    return scaler
end

"""
    fit!(scaler::MinMaxScaler, X::Matrix{<:Real})

Fit the min-max scaler to the data.

# Arguments
- `scaler::MinMaxScaler`: An instance of `MinMaxScaler`.
- `X::Matrix{<:Real}`: The data to fit the scaler.

# Returns
The fitted `MinMaxScaler`.
"""
function fit!(scaler::MinMaxScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    # Calculate min and max for each feature, and convert matrix to vector
    scaler.min = minimum(X, dims=1)[:]
    scaler.max = maximum(X, dims=1)[:]
    return scaler
end

"""
    transform(scaler::MinMaxScaler, X::Vector{<:Real})

Transform the data using the min-max scaler.

# Arguments
- `scaler::MinMaxScaler`: An instance of `MinMaxScaler`.
- `X::Vector{<:Real}`: The data to transform.

# Returns
The transformed data.
"""
function transform(scaler::MinMaxScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    feature_min, feature_max = scaler.feature_range
    range = scaler.max - scaler.min

    X_scaled = ((X .- scaler.min) ./ (range) ) .* (feature_max - feature_min) .+ feature_min

    return X_scaled 
end

"""
    transform(scaler::MinMaxScaler, X::Matrix{<:Real})

Transform the data using the min-max scaler.

# Arguments
- `scaler::MinMaxScaler`: An instance of `MinMaxScaler`.
- `X::Matrix{<:Real}`: The data to transform.

# Returns
The transformed data.
"""
function transform(scaler::MinMaxScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    feature_min, feature_max = scaler.feature_range
    X_scaled = similar(X, Float64)
    for j in 1:size(X, 2)
        min_val = scaler.min[j]
        max_val = scaler.max[j]
        range = max_val - min_val
        if range == 0
            X_scaled[:, j] .= feature_min
        else
            X_scaled[:, j] .= ((X[:, j] .- min_val) ./ range) .* (feature_max - feature_min) .+ feature_min
        end
    end
    return X_scaled
end

"""
    inverse_transform(scaler::MinMaxScaler, X::Vector{<:Real})

Inverse transform the data using the min-max scaler.

# Arguments
- `scaler::MinMaxScaler`: An instance of `MinMaxScaler`.
- `X::Vector{<:Real}`: The data to inverse transform.

# Returns
The inverse transformed data.
"""
function inverse_transform(scaler::MinMaxScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    feature_min, feature_max = scaler.feature_range
    min, max = scaler.min, scaler.max

    range = max - min

    X_original = ((X .- feature_min) ./ (feature_max - feature_min)) .* range .+ min

    return X_original
end

"""
    inverse_transform(scaler::MinMaxScaler, X::Matrix{<:Real})

Inverse transform the data using the min-max scaler.

# Arguments
- `scaler::MinMaxScaler`: An instance of `MinMaxScaler`.
- `X::Matrix{<:Real}`: The data to inverse transform.

# Returns
The inverse transformed data.
"""
function inverse_transform(scaler::MinMaxScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    feature_min, feature_max = scaler.feature_range
    min, max = scaler.min, scaler.max

    X_original = similar(X, Float64)

    for j in 1:size(X, 2)
        min_val = min[j]
        max_val = max[j]
        range_val = max_val - min_val
        if range_val == 0
            X_original[:, j] .= min_val
        else
            X_original[:, j] .= ((X[:, j] .- feature_min) ./ (feature_max - feature_min)) .* range_val .+ min_val
        end
    end

    return X_original
end

"""
    fit_transform!(scaler::MinMaxScaler, X::Vector{<:Real})

Fit and transform the data using the min-max scaler.

# Arguments
- `scaler::MinMaxScaler`: An instance of `MinMaxScaler`.
- `X::Vector{<:Real}`: The data to fit and transform.

# Returns
The transformed data.
"""
function fit_transform!(scaler::MinMaxScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    fit!(scaler, X)
    return transform(scaler, X)
end

"""
    fit_transform!(scaler::MinMaxScaler, X::Matrix{<:Real})

Fit and transform the data using the min-max scaler.

# Arguments
- `scaler::MinMaxScaler`: An instance of `MinMaxScaler`.
- `X::Matrix{<:Real}`: The data to fit and transform.

# Returns
The transformed data.
"""
function fit_transform!(scaler::MinMaxScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    fit!(scaler, X)
    return transform(scaler, X)
end