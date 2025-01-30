module MissingValue

export MissingValueTransformer, fit!, transform

import ..TransformerModule: Transformer, fit!, transform

"""
    MissingValueTransformer(strategy::String="drop", constant_value::Any=nothing)

Transformer for handling missing values using different strategies:
- "drop": Remove rows with missing values
- "mean": Replace with column means
- "constant": Replace with specified value

# Arguments
- `strategy::String`: Strategy to handle missing values ("drop", "mean", or "constant")
- `constant_value::Any`: Value to use when strategy is "constant"

# Returns
- `MissingValueTransformer`: A transformer object with the specified strategy and constant value
"""
mutable struct MissingValueTransformer <: Transformer
    strategy::String
    mean_values::Vector{Float64}
    constant_value::Any
    
    function MissingValueTransformer(strategy::String="drop", constant_value::Any=nothing)
        if !(strategy in ["drop", "mean", "constant"])
            throw(ArgumentError("Unknown strategy: $strategy. Supported strategies are: 'drop', 'mean', 'constant'"))
        end
        if strategy == "constant" && constant_value === nothing
            throw(ArgumentError("constant_value must be provided when using 'constant' strategy"))
        end
        new(strategy, Float64[], constant_value)
    end
end


"""
    fit!(transformer::MissingValueTransformer, X::Matrix{Any})

Fit the transformer to the input data. Updates the transformer's internal state.

# Arguments
- `transformer::MissingValueTransformer`: The transformer to fit.
- `X::Matrix{Any}`: Input data matrix.

# Returns
The updated transformer.
"""
function fit!(transformer::MissingValueTransformer, X::Matrix{Any})
    if transformer.strategy == "mean"
        transformer.mean_values = vec([calculate_mean(col) for col in eachcol(X)])
    end
    return transformer
end


"""
    transform(transformer::MissingValueTransformer, X::Matrix{Any})

Transform the input data by handling missing values according to the chosen strategy.

# Arguments
- `transformer::MissingValueTransformer`: The fitted transformer
- `X::Matrix{Any}`: Input data matrix with potential missing values

# Returns
- Transformed matrix with missing values handled according to the strategy
"""
function transform(transformer::MissingValueTransformer, X::Matrix{<:Any})
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
    # Attempt to cast the entire matrix to Float64 if all elements are numbers;
    # otherwise cast to String.
    if all(x -> x isa Number, transformed)
        return Matrix{Float64}(map(Float64, transformed))
    else
        return Matrix{String}(map(string, transformed))
    end
end


"""
    calculate_mean(col)

Calculate the mean of a column, ignoring missing values.

# Arguments
- `col`: Column of data that may contain missing values

# Returns
- Mean value of non-missing elements, or 0.0 if all elements are missing
"""
function calculate_mean(col)
    values = filter(!ismissing, col)
    isempty(values) ? 0.0 : sum(values) / length(values)
end

end
