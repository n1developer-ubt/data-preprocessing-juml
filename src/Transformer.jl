module TransformerModule

export Transformer, AddTransformer

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

end # module