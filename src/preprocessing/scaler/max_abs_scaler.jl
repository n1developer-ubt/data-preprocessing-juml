
"""
    MaxAbsScaler

Max Abs Scaler is a scaler that scales the data using the maximum absolute value of the data.

# Examples
```julia
scaler = MaxAbsScaler()
```
"""
mutable struct MaxAbsScaler <: BaseScaler
    max_abs::Union{Matrix{<:Real}, <:Real, Nothing}

    MaxAbsScaler() = new(nothing)
end

"""
    fit!(scaler::MaxAbsScaler, X::Vector{<:Real})

Fit the max-abs scaler to the data.

# Arguments
- `scaler::MaxAbsScaler`: An instance of `MaxAbsScaler`.
- `X::Vector{<:Real}`: The data to fit the scaler.

# Returns
The fitted `MaxAbsScaler`.
"""
function fit!(scaler::MaxAbsScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    scaler.max_abs = maximum(abs.(X))
    return scaler
end

"""
    fit!(scaler::MaxAbsScaler, X::Matrix{<:Real})

Fit the max-abs scaler to the data.

# Arguments
- `scaler::MaxAbsScaler`: An instance of `MaxAbsScaler`.
- `X::Matrix{<:Real}`: The data to fit the scaler.

# Returns
The fitted `MaxAbsScaler`.
"""
function fit!(scaler::MaxAbsScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    # Calculate max for each feature, and convert matrix to vector
    scaler.max_abs = maximum(abs.(X), dims=1)
    return scaler
end

"""
    transform(scaler::MaxAbsScaler, X::Vector{<:Real})

Transform the data using the max-abs scaler.

# Arguments
- `scaler::MaxAbsScaler`: An instance of `MaxAbsScaler`.
- `X::Vector{<:Real}`: The data to transform.

# Returns
The transformed data.
"""
function transform(scaler::MaxAbsScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    return X ./ scaler.max_abs
end

"""
    transform(scaler::MaxAbsScaler, X::Matrix{<:Real})

Transform the data using the max-abs scaler.

# Arguments
- `scaler::MaxAbsScaler`: An instance of `MaxAbsScaler`.
- `X::Matrix{<:Real}`: The data to transform.

# Returns
The transformed data.
"""
function transform(scaler::MaxAbsScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    return X ./ scaler.max_abs
end

"""
    inverse_transform(scaler::MaxAbsScaler, X::Vector{<:Real})

Inverse transform the data using the max-abs scaler.

# Arguments
- `scaler::MaxAbsScaler`: An instance of `MaxAbsScaler`.
- `X::Vector{<:Real}`: The data to inverse transform.

# Returns
The inverse transformed data.
"""
function inverse_transform(scaler::MaxAbsScaler, X::Vector{<:Real}, y::Vector{Any} = [])
    return X .* scaler.max_abs
end

"""
    inverse_transform(scaler::MaxAbsScaler, X::Matrix{<:Real})

Inverse transform the data using the max-abs scaler.

# Arguments
- `scaler::MaxAbsScaler`: An instance of `MaxAbsScaler`.
- `X::Matrix{<:Real}`: The data to inverse transform.

# Returns
The inverse transformed data.
"""
function inverse_transform(scaler::MaxAbsScaler, X::Matrix{<:Real}, y::Vector{Any} = [])
    return X .* scaler.max_abs
end