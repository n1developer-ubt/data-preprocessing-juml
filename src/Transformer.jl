module TransformerModule

export Transformer, AddTransformer, MissingValueTransformer

abstract type Transformer end

"""
    fit!(transformer::Transformer, X::Matrix{Any})

Fits the transformer to the given data `X`. This method should be implemented
for any concrete subtype of `Transformer`.

# Arguments
- `transformer::Transformer`: The transformer instance.
- `X::Matrix{Any}`: The input data to fit the transformer.

# Returns
The fitted transformer.
"""
function fit!(transformer::Transformer, X::Matrix{Any})
    throw(MethodError(fit!, (transformer, X)))
end

"""
    transform(transformer::Transformer, X::Matrix{Any})

Applies the transformation defined by the transformer to the input data `X`.
This method should be implemented for any concrete subtype of `Transformer`.

# Arguments
- `transformer::Transformer`: The transformer instance.
- `X::Matrix{Any}`: The input data to transform.

# Returns
A transformed version of `X`.
"""
function transform(transformer::Transformer, X::Matrix{Any})
    throw(MethodError(transform, (transformer, X)))
end

# EXAMPLE ##### TODO remove

mutable struct AddTransformer <: Transformer
    value::Int

    AddTransformer() = new(0)
end

function fit!(addTrans::AddTransformer, X::Matrix{Any})
    addTrans.value = 2 # Set fixed value for example
    return addTrans
end

function transform(addTrans::AddTransformer, X::Matrix{Any})
    return X .+ addTrans.value
end

function fit!(addTrans::AddTransformer, X::Matrix{Int64})
    addTrans.value = 2 # Set fixed value for example
    return addTrans
end

function transform(addTrans::AddTransformer, X::Matrix{Int64})
    return X .+ addTrans.value
end



### MISSING VALUE TRANSFORMER ##### TODO remove



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



end # module