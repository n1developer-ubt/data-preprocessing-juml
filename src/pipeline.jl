module PipelineModule

using OrderedCollections: OrderedDict

# Pipelines fit! and transform are also extensions to Transformer
import ..TransformerModule: Transformer, fit!, transform, inverse_transform, fit_transform!

export Pipeline, make_pipeline, fit!, transform, inverse_transform, fit_transform!, add_step!


"""
    struct Pipeline

A structure representing a machine learning pipeline. The pipeline consists of a
series of named steps (transformers) applied sequentially to the input data.

# Fields
- `named_steps::D`: Dictionary of named transformer steps.
- `n_features_in_::Int`: Number of input features in the dataset.
- `feature_names_in_::V`: Names of the input features.
"""
mutable struct Pipeline{D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}
    named_steps::D
    n_features_in_::Int
    feature_names_in_::V
end


"""
    Pipeline(steps::AbstractDict{String, <:Transformer})

Initialize a `Pipeline` with the given steps.

# Arguments
- `steps::AbstractDict{String, <:Transformer}`: A dictionary of named transformer steps.

# Returns
A new `Pipeline` instance.
"""
function Pipeline(steps::AbstractDict{String, <:Transformer})
    ordered_steps = OrderedDict(steps)
    feature_names = String[]
    return Pipeline{typeof(ordered_steps), typeof(feature_names)}(steps, 0, feature_names)
end


"""
    add_step!(pipeline::Pipeline{D, V}, name::String, step::T) where {T<:Transformer, D<:AbstractDict{String, T}, V<:AbstractVector{String}}

Add a transformer step to an existing pipeline.

# Arguments
- `pipeline::Pipeline{D, V}`: The pipeline to which the step will be added.
- `name::String`: Name of the transformer step.
- `step::T`: The transformer to be added.

# Returns
Nothing. Modifies the pipeline in place.
"""
function add_step!(pipeline::Pipeline{D, V}, name::String, step::T) where {T<:Transformer, D<:AbstractDict{String, T}, V<:AbstractVector{String}}
    pipeline.named_steps[name] = step
end


"""
    fit!(pipeline::Pipeline{D, V}, X::Matrix{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}

Fit the pipeline to the input data by fitting each step sequentially. Updates the
transformers in `pipeline.named_steps` with the results of their `fit!` method.

# Arguments
- `pipeline::Pipeline{D, V}`: The pipeline to fit.
- `X::Matrix{T}`: Input data matrix.

# Returns
The updated pipeline.
"""
function fit!(pipeline::Pipeline{D, V}, X::Matrix{T}) where {T, D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}
    pipeline.n_features_in_ = size(X, 2)
    pipeline.feature_names_in_ = ["feature_$i" for i in 1:size(X, 2)]
    X_copy = copy(X)  # Create a copy of X

    # Sequentially fit each step, updating the pipeline's references if necessary
    for (name, step) in pipeline.named_steps
        fitted_step = fit!(step, X_copy)
        X_copy = transform(step, X_copy)  # Transform X_copy for the next step
        pipeline.named_steps[name] = fitted_step  # Update the step in the pipeline
    end

    return pipeline
end


"""
    fit!(pipeline::Pipeline{D, V}, X::Vector{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}

Fit the pipeline to the input data by fitting each step sequentially. Updates the
transformers in `pipeline.named_steps` with the results of their `fit!` method.

# Arguments
- `pipeline::Pipeline{D, V}`: The pipeline to fit.
- `X::Vector{T}`: Input data vector.

# Returns
The updated pipeline.
"""
function fit!(pipeline::Pipeline{D, V}, X::Vector{T}) where {T, D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}
    pipeline.n_features_in_ = length(X)
    pipeline.feature_names_in_ = ["feature_$i" for i in 1:length(X)]
    X_copy = copy(X)  # Create a copy of X

    # Sequentially fit each step, updating the pipeline's references if necessary
    for (name, step) in pipeline.named_steps
        fitted_step = fit!(step, X_copy)
        X_copy = transform(step, X_copy)  # Transform X_copy for the next step
        pipeline.named_steps[name] = fitted_step  # Update the step in the pipeline
    end

    return pipeline
end


"""
    transform(pipeline::Pipeline{D, V}, X::Matrix{Any}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}

Transform the input data using the pipeline by applying each step sequentially.

# Arguments
- `pipeline::Pipeline{D, V}`: The pipeline to use for transformation.
- `X::Matrix{T}`: Input data matrix.

# Returns
Transformed data matrix.
"""
function transform(pipeline::Pipeline{D, V}, X::Matrix{T}) where {T, D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}
    X_transformed = copy(X)
    for (name, step) in pipeline.named_steps
        X_transformed = transform(step, X_transformed)
    end
    return X_transformed
end


"""
    transform(pipeline::Pipeline{D, V}, X::Vector{Any}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}

Transform the input data using the pipeline by applying each step sequentially.

# Arguments
- `pipeline::Pipeline{D, V}`: The pipeline to use for transformation.
- `X::Vector{T}`: Input data vector.

# Returns
Transformed data vector.
"""
function transform(pipeline::Pipeline{D, V}, X::Vector{T}) where {T, D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}
    X_transformed = copy(X)
    for (name, step) in pipeline.named_steps
        X_transformed = transform(step, X_transformed)
    end
    return X_transformed
end


"""
    inverse_transform(pipeline::Pipeline{D, V}, X::Matrix{Any}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}

Inverse transform the data using the pipeline by applying each step's inverse transform sequentially in reverse order.

# Arguments
- `pipeline::Pipeline{D, V}`: The pipeline to use for inverse transformation.
- `X::Matrix{T}`: Input data matrix.

# Returns
Inverse transformed data matrix.
"""
function inverse_transform(pipeline::Pipeline{D, V}, X::Matrix{T}) where {T, D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}
    X_inv_transformed = copy(X)
    for (name, step) in reverse(collect(pipeline.named_steps))
        X_inv_transformed = inverse_transform(step, X_inv_transformed)
    end
    return X_inv_transformed
end


"""
    inverse_transform(pipeline::Pipeline{D, V}, X::Vector{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}

Inverse transform the data using the pipeline by applying each step's inverse transform sequentially in reverse order.

# Arguments
- `pipeline::Pipeline{D, V}`: The pipeline to use for inverse transformation.
- `X::Vector{T}`: Input data vector.

# Returns
Inverse transformed data vector.
"""
function inverse_transform(pipeline::Pipeline{D, V}, X::Vector{T}) where {T, D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}
    X_inv_transformed = copy(X)
    for (name, step) in reverse(collect(pipeline.named_steps))
        X_inv_transformed = inverse_transform(step, X_inv_transformed)
    end
    return X_inv_transformed
end


"""
    fit_transform!(pipeline::Pipeline{D, V}, X::Matrix{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}

Fit the pipeline to the data and transform it in one step.

# Arguments
- `pipeline::Pipeline{D, V}`: The pipeline to fit and transform.
- `X::Matrix{T}`: Input data matrix.

# Returns
Transformed data matrix.
"""
function fit_transform!(pipeline::Pipeline{D, V}, X::Matrix{T}) where {T, D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}
    fit!(pipeline, X)
    return transform(pipeline, X)
end


"""
    fit_transform!(pipeline::Pipeline{D, V}, X::Vector{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}

Fit the pipeline to the data and transform it in one step.

# Arguments
- `pipeline::Pipeline{D, V}`: The pipeline to fit and transform.
- `X::Vector{T}`: Input data vector.

# Returns
Transformed data vector.
"""
function fit_transform!(pipeline::Pipeline{D, V}, X::Vector{T}) where {T, D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}
    fit!(pipeline, X)
    return transform(pipeline, X)
end


"""
    make_pipeline(steps::Pair{String, <:Transformer}...)

Create a pipeline from a sequence of named transformer steps.

# Arguments
- `steps::Pair{String, <:Transformer}...`: Pairs of step names and transformers.

# Returns
A new `Pipeline` instance.
"""
function make_pipeline(steps::Pair{String, <:Transformer}...)
    named_steps = OrderedDict(steps)
    return Pipeline(named_steps)
end

end # module
