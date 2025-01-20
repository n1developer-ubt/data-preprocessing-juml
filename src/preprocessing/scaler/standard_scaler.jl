# import required modules

using Statistics

import ...TransformerModule: fit!, transform, inverse_transform

"""
    struct StandardScaler
    
    A structure representing a standard scaler.
    The standard scaler standardizes features by removing the mean.

    # Fields
    - `mean::Union{Vector{Float64}, Float64, Nothing}`: Mean of the features.
    - `std::Union{Vector{Float64}, Float64, Nothing}`: Standard deviation of the features.

"""
mutable struct StandardScaler <: BaseScaler
    mean::Union{Vector{Float64}, Float64, Nothing}
    std::Union{Vector{Float64}, Float64, Nothing}

    StandardScaler() = new(nothing, nothing)
end


"""
    fit!(scaler::StandardScaler, X::Vector{<:Real})

    Fit the standard scaler to the data.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Vector{<:Real}`: The data to fit the scaler.

    # Returns
    The fitted `StandardScaler`.
"""
function fit!(scaler::StandardScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    scaler.mean = mean(X)
    scaler.std = std(X, corrected=false)
    return scaler
end

"""
    fit!(scaler::StandardScaler, X::Matrix{<:Real})

    Fit the standard scaler to the data.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Matrix{<:Real}`: The data to fit the scaler.

    # Returns
    The fitted `StandardScaler`.
"""
function fit!(scaler::StandardScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    # Calculate mean and std for each feature, and convert matrix to vector
    scaler.mean = mean(X, dims=1)[:]
    scaler.std = std(X, dims=1, corrected=false)[:]
    return scaler
end

"""
    transform(scaler::StandardScaler, X::Vector{<:Real})

    Transform the data using the fitted standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Vector{<:Real}`: The data to transform.

    # Returns
    The transformed data.
"""
function transform(scaler::StandardScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    if(isnothing(scaler.mean) || isnothing(scaler.std))
        throw(ArgumentError("Scaler not fitted yet"))
    end

    return (X .- scaler.mean) ./ scaler.std
end

"""
    transform(scaler::StandardScaler, X::Matrix{<:Real})

    Transform the data using the fitted standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Matrix{<:Real}`: The data to transform.

    # Returns
    The transformed data.
"""
function transform(scaler::StandardScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    if(isnothing(scaler.mean) || isnothing(scaler.std))
        throw(ArgumentError("Scaler not fitted yet"))
    end
    if length(scaler.mean) != size(X, 2) || length(scaler.std) != size(X, 2)
        throw(ArgumentError("Number of features in scaler and X should be same"))
    end
    # using transpose function to transpose the mean vector to align with the matrix on column axis
    return (X .- transpose(scaler.mean)) ./ transpose(scaler.std)
end


"""
    inverse_transform(scaler::StandardScaler, X::Vector{<:Real})

    Inverse transform the data using the fitted standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Vector{<:Real}`: The data to inverse transform.

    # Returns
    The inverse transformed data.
"""
function inverse_transform(scaler::StandardScaler, X::Vector{<:Real})
    if(isnothing(scaler.mean) || isnothing(scaler.std))
        throw(ArgumentError("Scaler not fitted yet"))
    end

    return X .* scaler.std .+ scaler.mean
end

"""
    inverse_transform(scaler::StandardScaler, X::Matrix{<:Real})

    Inverse transform the data using the fitted standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Matrix{<:Real}`: The data to inverse transform.

    # Returns
    The inverse transformed data.
"""
function inverse_transform(scaler::StandardScaler, X::Matrix{<:Real})
    if(isnothing(scaler.mean) || isnothing(scaler.std))
        throw(ArgumentError("Scaler not fitted yet"))
    end
    return X .* transpose(scaler.std) .+ transpose(scaler.mean)
end

"""
    fit_transform!(scaler::StandardScaler, X::Vector{<:Real})

    Fit and transform the data using the standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Vector{<:Real}`: The data to fit and transform.

    # Returns
    The transformed data.
"""
function fit_transform!(scaler::StandardScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    fit!(scaler, X)
    return transform(scaler, X)
end

"""
    fit_transform!(scaler::StandardScaler, X::Matrix{<:Real})

    Fit and transform the data using the standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Matrix{<:Real}`: The data to fit and transform.

    # Returns
    The transformed data.
"""
function fit_transform!(scaler::StandardScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    fit!(scaler, X)
    return transform(scaler, X)
end