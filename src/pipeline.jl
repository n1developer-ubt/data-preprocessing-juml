module PipelineModule

# Pipelines fit! and transform are also extensions to Transformer
import ..TransformerModule: Transformer, fit!, transform, inverse_transform, fit_transform!

export Pipeline, make_pipeline, fit!, transform, fit_transform!, add_step!


"""
    struct Pipeline

A structure representing a machine learning pipeline. The pipeline consists of a
series of named steps (transformers) applied sequentially to the input data.

# Fields
- `named_steps::Dict{String, Transformer}`: Dictionary of named transformer steps.
- `n_features_in_::Int`: Number of input features in the dataset.
- `feature_names_in_::Vector{String}`: Names of the input features.
"""
mutable struct Pipeline
    named_steps::Dict{String, Transformer}
    n_features_in_::Int
    feature_names_in_::Vector{String}
end


"""
    Pipeline(steps::Dict{String, Transformer})

Initialize a `Pipeline` with the given steps.

# Arguments
- `steps::Dict{String, Transformer}`: A dictionary of named transformer steps.

# Returns
A new `Pipeline` instance.
"""
function Pipeline(steps::Dict{String, <:Transformer})
    feature_names = String[]
    return Pipeline(steps, 0, feature_names)
end


"""
    add_step!(pipeline::Pipeline, name::String, step::Transformer)

Add a transformer step to an existing pipeline.

# Arguments
- `pipeline::Pipeline`: The pipeline to which the step will be added.
- `name::String`: Name of the transformer step.
- `step::Transformer`: The transformer to be added.

# Returns
Nothing. Modifies the pipeline in place.
"""
function add_step!(pipeline::Pipeline, name::String, step::Transformer)
    pipeline.named_steps[name] = step
end


"""
    fit!(pipeline::Pipeline, X::Matrix{Any})

Fit the pipeline to the input data by fitting each step sequentially. Updates the
transformers in `pipeline.named_steps` with the results of their `fit!` method.

# Arguments
- `pipeline::Pipeline`: The pipeline to fit.
- `X::Matrix{Any}`: Input data matrix.

# Returns
The updated pipeline.
"""
function fit!(pipeline::Pipeline, X::Matrix)
    pipeline.n_features_in_ = size(X, 2)
    pipeline.feature_names_in_ = ["feature_$i" for i in 1:size(X, 2)]

    # Sequentially fit each step, updating the pipeline's references if necessary
    for (name, step) in pipeline.named_steps
        fitted_step = fit!(step, X)
        pipeline.named_steps[name] = fitted_step  # Update the step in the pipeline
    end

    return pipeline
end

function fit!(pipeline::Pipeline, X::Vector)
    pipeline.n_features_in_ = size(X, 2)
    pipeline.feature_names_in_ = ["feature_$i" for i in 1:size(X, 2)]

    # Sequentially fit each step, updating the pipeline's references if necessary
    for (name, step) in pipeline.named_steps
        fitted_step = fit!(step, X)
        pipeline.named_steps[name] = fitted_step  # Update the step in the pipeline
    end

    return pipeline
end


"""
    transform(pipeline::Pipeline, X::Matrix{Any})

Transform the input data using the pipeline by applying each step sequentially.

# Arguments
- `pipeline::Pipeline`: The pipeline to use for transformation.
- `X::Matrix{Any}`: Input data matrix.

# Returns
Transformed data matrix.
"""
function transform(pipeline::Pipeline, X::Matrix)
    X_transformed = copy(X)
    for (name, step) in pipeline.named_steps
        X_transformed = transform(step, X_transformed)
    end
    return X_transformed
end

function transform(pipeline::Pipeline, X::Vector)
    X_transformed = copy(X)
    for (name, step) in pipeline.named_steps
        X_transformed = transform(step, X_transformed)
    end
    return X_transformed
end


"""
    fit_transform!(pipeline::Pipeline, X::Matrix)

Fit the pipeline to the data and transform it in one step.

# Arguments
- `pipeline::Pipeline`: The pipeline to fit and transform.
- `X::Matrix{Any}`: Input data matrix.

# Returns
Transformed data matrix.
"""
function fit_transform!(pipeline::Pipeline, X::Matrix)
    fit!(pipeline, X)
    return transform(pipeline, X)
end

function fit_transform!(pipeline::Pipeline, X::Vector)
    fit!(pipeline, X)
    return transform(pipeline, X)
end


"""
    make_pipeline(steps::Pair{String, Transformer}...)

Create a pipeline from a sequence of named transformer steps.

# Arguments
- `steps::Pair{String, Transformer}...`: Pairs of step names and transformers.

# Returns
A new `Pipeline` instance.
"""
function make_pipeline(steps::Pair{String, <:Transformer}...)
    named_steps = Dict(steps)
    return Pipeline(named_steps)
end

end # module
