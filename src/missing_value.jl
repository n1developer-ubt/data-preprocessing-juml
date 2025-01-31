module MissingValue

export MissingValueTransformer, fit!, transform

import ..TransformerModule: Transformer, fit!, transform

"""
    MissingValueTransformer(strategy::String="drop", constant_value::Union{Number, String}=0)

Transformer for handling missing values in data matrices.

# Arguments
- `strategy::String`: Strategy to handle missing values:
    - "drop": Remove rows containing any missing values
    - "mean": Replace missing values with column means (numeric data only)
    - "constant": Replace missing values with a specified constant value
- `constant_value::Union{Number, String}=0`: Value to use when strategy is "constant"

# Returns
- `MissingValueTransformer`: A transformer object for handling missing values

# Throws
- `ArgumentError`: If strategy is not one of ["drop", "mean", "constant"]
- `ArgumentError`: If strategy is "constant" but no valid constant_value is provided
"""
mutable struct MissingValueTransformer <: Transformer
    strategy::String
    mean_values::Vector{Float64}
    constant_value::Union{Number, String}
    
    function MissingValueTransformer(strategy::String="drop", constant_value::Union{Number, String}=0)
        if !(strategy in ["drop", "mean", "constant"])
            throw(ArgumentError("Unknown strategy: $strategy. Supported strategies are: 'drop', 'mean', 'constant'"))
        end
        if strategy == "constant" && constant_value === 0 && !isa(constant_value, String)
            throw(ArgumentError("constant_value must be provided when using 'constant' strategy"))
        end
        new(strategy, Float64[], constant_value)
    end
end


"""
    fit!(transformer::MissingValueTransformer, X::Matrix{Union{Missing, T}}) where T <: Number

Fit the transformer to the input data by computing necessary statistics (e.g., column means).

# Arguments
- `transformer::MissingValueTransformer`: The transformer to fit
- `X::Matrix{Union{Missing, T}} where T <: Number`: Input data matrix

# Returns
- `MissingValueTransformer`: The fitted transformer

# Notes
- For "mean" strategy: Computes and stores column means, ignoring missing values
- For "drop" and "constant" strategies: No fitting required
"""
function fit!(transformer::MissingValueTransformer, X::Matrix{Union{Missing, T}}) where T <: Number
    if transformer.strategy == "mean"
        transformer.mean_values = vec([calculate_mean(col) for col in eachcol(X)])
    end
    return transformer
end


function fit!(transformer::MissingValueTransformer, X::Matrix{Union{Missing, String}})
    if transformer.strategy == "mean"
        throw(ArgumentError("Mean strategy is not supported for string data"))
    end
    return transformer
end


"""
    transform(transformer::MissingValueTransformer, X::Matrix{Union{Missing, T}}) where T <: Number

Transform the input data by handling missing values according to the fitted strategy.

# Arguments
- `transformer::MissingValueTransformer`: The fitted transformer
- `X::Matrix{Union{Missing, T}} where T <: Number`: Input data matrix with potential missing values

# Returns
- `Matrix{Number}`: Transformed matrix with missing values handled according to the strategy
    - Empty input returns empty output
    - Output is converted to Number for numeric input
    - Output is converted to String for string input

# Notes
- "drop" strategy: Returns matrix with fewer rows if missing values are present
- "mean" strategy: Uses column means computed during fit
- "constant" strategy: Uses the specified constant_value
"""
function transform(transformer::MissingValueTransformer, X::Matrix{Union{Missing, T}}) where T <: Number
    if isempty(X)
        return X
    end

    # Drop missing values
    transformed = if transformer.strategy == "drop"
        valid_rows = .!vec(any(ismissing, X, dims=2))
        X[valid_rows, :]

    # Replace missing values with mean of the column
    elseif transformer.strategy == "mean"
        result = copy(X)
        for (col_idx, col) in enumerate(eachcol(result))
            missing_mask = ismissing.(col)
            if any(missing_mask)
                col[missing_mask] .= transformer.mean_values[col_idx]
            end
        end
        result

    # Replace missing values with a constant value
    elseif transformer.strategy == "constant"
        result = copy(X)
        for col in eachcol(result)
            missing_mask = ismissing.(col)
            if any(missing_mask)
                col[missing_mask] .= transformer.constant_value
            end
        end
        result
    end

    # === CAST STEP ===
    if all(x -> x isa Number, transformed)
        return Matrix{Float64}(transformed)
    else
        return Matrix{String}(map(string, transformed))
    end

end

function transform(transformer::MissingValueTransformer, X::Matrix{Union{Missing, String}})
    if isempty(X)
        return X
    end
    
    transformed = if transformer.strategy == "drop"
        valid_rows = .!vec(any(ismissing, X, dims=2))
        X[valid_rows, :]
    elseif transformer.strategy == "constant"
        result = copy(X)
        for col in eachcol(result)
            missing_mask = ismissing.(col)
            if any(missing_mask)
                col[missing_mask] .= transformer.constant_value
            end
        end
        result
    end
    
    return Matrix{String}(map(string, transformed))
end


"""
    calculate_mean(col)

Calculate the mean of a column, ignoring missing values.

# Arguments
- `col`: Column of data that may contain missing values

# Returns
- `Float64`: Mean value of non-missing elements
    - Returns 0.0 if all elements are missing
    - Returns 0.0 if column is empty

# Notes
- Missing values are filtered out before calculation
- Uses standard arithmetic mean for computation
"""
function calculate_mean(col)
    values = filter(!ismissing, col)
    isempty(values) ? 0.0 : sum(values) / length(values)
end

end
