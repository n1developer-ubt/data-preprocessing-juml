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

end # module
