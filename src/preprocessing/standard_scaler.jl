# import required modules
include("scaler.jl")

using Statistics

"""
    struct StandardScaler
    
    A structure representing a standard scaler.
    The standard scaler standardizes features by removing the mean.

    # Fields
    - `mean::Union{Vector{Float64}, Float64, Nothing}`: Mean of the features.
    - `std::Union{Vector{Float64}, Float64, Nothing}`: Standard deviation of the features.

"""
mutable struct StandardScaler <: Scaler
    mean::Union{Vector{Float64}, Float64, Nothing}
    std::Union{Vector{Float64}, Float64, Nothing}

    StandardScaler() = new(nothing, nothing)
end


"""
    fit!(scaler::StandardScaler, X::Matrix{<:Number})

    Fit the standard scaler to the data.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Matrix{<:Number}`: The data to fit the scaler.

    # Returns
    The fitted `StandardScaler`.
"""
function fit!(scaler::StandardScaler, X::Matrix{<:Number}, y::Vector{Any} = [])
    # Calculate mean and std for each feature, and convert matrix to vector
    scaler.mean = mean(X, dims=1)[:]
    scaler.std = std(X, dims=1, corrected=false)[:]
    return scaler
end

"""
    fit!(scaler::StandardScaler, X::Vector{<:Number})

    Fit the standard scaler to the data.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Vector{<:Number}`: The data to fit the scaler.

    # Returns
    The fitted `StandardScaler`.
"""
function fit!(scaler::StandardScaler, X::Vector{<:Number}, y::Vector{Any} = [])
    scaler.mean = mean(X)
    scaler.std = std(X, corrected=false)
    return scaler
end

"""
    transform(scaler::StandardScaler, X::Matrix{<:Number})

    Transform the data using the fitted standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Matrix{<:Number}`: The data to transform.

    # Returns
    The transformed data.
"""
function transform(scaler::StandardScaler, X::Matrix{<:Number}, y::Vector{Any} = [])
    if(isnothing(scaler.mean) || isnothing(scaler.std))
        throw(ArgumentError("Scaler not fitted yet"))
    end
    if length(scaler.mean) != size(X, 2) || length(scaler.std) != size(X, 2)
        throw(ArgumentError("Number of features in scaler and X should be same"))
    end
    # using ' to transpose the mean vector to align with the matrix on column axis
    return (X .- scaler.mean') ./ scaler.std'
end

"""
    transform(scaler::StandardScaler, X::Vector{<:Number})

    Transform the data using the fitted standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Vector{<:Number}`: The data to transform.

    # Returns
    The transformed data.
"""
function transform(scaler::StandardScaler, X::Vector{<:Number}, y::Vector{Any} = [])
    if(isnothing(scaler.mean) || isnothing(scaler.std))
        throw(ArgumentError("Scaler not fitted yet"))
    end

    return (X .- scaler.mean) ./ scaler.std
end


"""
    inverse_transform(scaler::StandardScaler, X::Matrix{<:Number})

    Inverse transform the data using the fitted standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Matrix{<:Number}`: The data to inverse transform.

    # Returns
    The inverse transformed data.
"""
function inverse_transform(scaler::StandardScaler, X::Matrix{<:Number})
    if(isnothing(scaler.mean) || isnothing(scaler.std))
        throw(ArgumentError("Scaler not fitted yet"))
    end
    if size(scaler.mean) != size(X, 2)
        throw(ArgumentError("Number of features in scaler and X should be same"))
    end
    return X .* scaler.std .+ scaler.mean
end

"""
    inverse_transform(scaler::StandardScaler, X::Vector{<:Number})

    Inverse transform the data using the fitted standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Vector{<:Number}`: The data to inverse transform.

    # Returns
    The inverse transformed data.
"""
function inverse_transform(scaler::StandardScaler, X::Vector{<:Number})
    if(isnothing(scaler.mean) || isnothing(scaler.std))
        throw(ArgumentError("Scaler not fitted yet"))
    end

    return X .* scaler.std .+ scaler.mean
end


"""
    fit_transform!(scaler::StandardScaler, X::Matrix{<:Number})

    Fit and transform the data using the standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Matrix{<:Number}`: The data to fit and transform.

    # Returns
    The transformed data.
"""
function fit_transform!(scaler::StandardScaler, X::Matrix{<:Number}, y::Vector{Any} = [])
    fit!(scaler, X)
    return transform(scaler, X)
end

"""
    fit_transform!(scaler::StandardScaler, X::Vector{<:Number})

    Fit and transform the data using the standard scaler.

    # Arguments
    - `scaler::StandardScaler`: An instance of `StandardScaler`.
    - `X::Vector{<:Number}`: The data to fit and transform.

    # Returns
    The transformed data.
"""
function fit_transform!(scaler::StandardScaler, X::Vector{<:Number}, y::Vector{Any} = [])
    fit!(scaler, X)
    return transform(scaler, X)
end