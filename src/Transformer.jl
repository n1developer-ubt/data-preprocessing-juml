module TransformerModule

export Transformer, fit!, transform, inverse_transform, fit_transform!

"""
    Transformer

Abstract type for a step in a pipeline. Defines the essential interface for all transformer types.

# Methods to be implemented by subtypes
- `fit!(::Transformer, X::AbstractArray)`: Fit the transformer to the data.
- `transform(::Transformer, X::AbstractArray)`: Transform the data.
- `inverse_transform(::Transformer, X::AbstractArray)`: Inverse transform the data.
- `fit_transform!(::Transformer, X::AbstractArray)`: Fit the transformer to the data and transform it in one step.
"""
abstract type Transformer end


"""
    fit!(transformer::Transformer, X::AbstractArray) -> Transformer

Fits the transformer to the given data `X`. This method should be implemented
for any concrete subtype of `Transformer`.

# Arguments
- `transformer::Transformer`: The transformer instance.
- `X::AbstractArray`: The input data to fit the transformer.

# Returns
The fitted transformer.
"""
function fit!(transformer::Transformer, X::AbstractArray)
    throw(MethodError(fit!, (transformer, X)))
end


"""
    transform(transformer::Transformer, X::AbstractArray) -> AbstractArray

Applies the transformation defined by the transformer to the input data `X`.
This method should be implemented for any concrete subtype of `Transformer`.

# Arguments
- `transformer::Transformer`: The transformer instance.
- `X::AbstractArray`: The input data to transform.

# Returns
A transformed version of `X`.
"""
function transform(transformer::Transformer, X::AbstractArray)
    throw(MethodError(transform, (transformer, X)))
end


"""
    inverse_transform(transformer::Transformer, X::AbstractArray) -> AbstractArray

Inverse transform the data using the transformer.

# Arguments
- `transformer::Transformer`: The transformer to use for inverse transformation.
- `X::AbstractArray`: Input data array.

# Returns
The inverse transformed data array.
"""
function inverse_transform(transformer::Transformer, X::AbstractArray)
    throw(MethodError(inverse_transform, (transformer, X)))
end


"""
    fit_transform!(transformer::Transformer, X::AbstractArray) -> AbstractArray

Fits the transformer to the given data `X` and then also applies the transformation.

# Arguments
- `transformer::Transformer`: The transformer instance.
- `X::AbstractArray`: The input data to fit the transformer.

# Returns
A transformed version of `X`.
"""
function fit_transform!(transformer::Transformer, X::AbstractArray)
    throw(MethodError(fit_transform!, (transformer, X)))
end

end # module
