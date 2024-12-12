module MissingValue

using ..TransformerModule: Transformer

export MissingValueTransformer


mutable struct MissingValueTransformer <: Transformer
    strategy::String
    mean_values::Vector{Float64}
    
    function MissingValueTransformer(strategy::String="drop")
        if !(strategy in ["drop", "mean"])
            throw(ArgumentError("Unknown strategy: $strategy. Supported strategies are: 'drop', 'mean'"))
        end
        new(strategy, Float64[])
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

    if transformer.strategy == "drop"
        valid_rows = .!vec(any(ismissing, X, dims=2))
        return X[valid_rows, :]

    elseif transformer.strategy == "mean"
        result = copy(X)
        for (col_idx, col) in enumerate(eachcol(result))
            missing_mask = ismissing.(col)
            if any(missing_mask)
                col[missing_mask] .= transformer.mean_values[col_idx]
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