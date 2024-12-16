module MissingValue

export MissingValueTransformer

import ..TransformerModule: Transformer, fit!, transform

"""
    MissingValueTransformer(strategy::String="drop", constant_value::Any=nothing)

Transformer for handling missing values using different strategies:
- "drop": Remove rows with missing values
- "mean": Replace with column means
- "constant": Replace with specified value
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


function fit!(transformer::MissingValueTransformer, X::Matrix{Any})
    if transformer.strategy == "mean"
        transformer.mean_values = vec([calculate_mean(col) for col in eachcol(X)])
    end
    return transformer
end


function transform(transformer::MissingValueTransformer, X::Matrix{Any})
    if isempty(X)
        return X
    end

    # Drop missing values
    if transformer.strategy == "drop"
        valid_rows = .!vec(any(ismissing, X, dims=2))
        return X[valid_rows, :]

    # Replace missing values with mean of the column
    elseif transformer.strategy == "mean"
        result = copy(X)
        for (col_idx, col) in enumerate(eachcol(result))
            missing_mask = ismissing.(col)
            if any(missing_mask)
                col[missing_mask] .= transformer.mean_values[col_idx]
            end
        end
        return result

    # Replace missing values with a constant value
    elseif transformer.strategy == "constant"
        result = copy(X)
        for col in eachcol(result)
            missing_mask = ismissing.(col)
            if any(missing_mask)
                col[missing_mask] .= transformer.constant_value
            end
        end
        return result
    end
end


function calculate_mean(col)
    values = filter(!ismissing, col)
    isempty(values) ? 0.0 : sum(values) / length(values)
end


end