
module PipelineModule

export Pipeline, PipelineStep, Transformer, NoScaler, make_pipeline, fit!, transform, predict, fit_transform!, fit_predict!, score, add_step!

# Abstract base class for all pipeline steps
abstract type PipelineStep end

# Example transformer with fit and transform methods #####
abstract type Transformer <: PipelineStep end

# TODO remove NoScaler

mutable struct NoScaler <: Transformer
    mean::Vector{Float64}
    std::Vector{Float64}

    NoScaler() = new(Float64[], Float64[])
end

function fit!(scaler::NoScaler, X::Matrix{Float64}, y::Vector{Any} = [])
    num_features = size(X, 2)
    scaler.mean = zeros(num_features)
    scaler.std = ones(num_features)
    return scaler
end

function transform(scaler::NoScaler, X::Matrix{Float64})
    return X
end

# PIPELINE #####

"""
    struct Pipeline

A structure representing a machine learning pipeline. The pipeline consists of a
series of named steps (transformers) applied sequentially to the input data.

# Fields
- `named_steps::Dict{String, Transformer}`: Dictionary of named transformer steps.
- `classes_::Vector{String}`: Classes handled by the pipeline (if applicable).
- `n_features_in_::Int`: Number of input features in the dataset.
- `feature_names_in_::Vector{String}`: Names of the input features.
"""
mutable struct Pipeline
    named_steps::Dict{String, Transformer}
    classes_::Vector{String}
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
    return Pipeline(steps, String[], 0, feature_names)
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
    fit!(pipeline::Pipeline, X::Matrix{Float64}, y::Vector{Any} = [])

Fit the pipeline to the input data by fitting each step sequentially. Updates the
transformers in `pipeline.named_steps` with the results of their `fit!` method.

# Arguments
- `pipeline::Pipeline`: The pipeline to fit.
- `X::Matrix{Float64}`: Input data matrix.
- `y::Vector{Any}`: Optional target values (default: empty vector).

# Returns
The updated pipeline.
"""
function fit!(pipeline::Pipeline, X::Matrix{Float64}, y::Vector{Any} = [])
    pipeline.n_features_in_ = size(X, 2)
    pipeline.feature_names_in_ = ["feature_$i" for i in 1:size(X, 2)]

    # Sequentially fit each step, updating the pipeline's references if necessary
    for (name, step) in pipeline.named_steps
        fitted_step = fit!(step, X, y)
        pipeline.named_steps[name] = fitted_step  # Update the step in the pipeline
    end

    return pipeline
end


"""
    transform(pipeline::Pipeline, X::Matrix{Float64})

Transform the input data using the pipeline by applying each step sequentially.

# Arguments
- `pipeline::Pipeline`: The pipeline to use for transformation.
- `X::Matrix{Float64}`: Input data matrix.

# Returns
Transformed data matrix.
"""
function transform(pipeline::Pipeline, X::Matrix{Float64})
    @info X
    X_transformed = copy(X)
    for (name, step) in pipeline.named_steps
        X_transformed = transform(step, X_transformed)
        #@info name, X_transformed
    end
    return X_transformed
end

"""
    fit_transform!(pipeline::Pipeline, X::Matrix{Float64}, y::Vector{Any} = [])

Fit the pipeline to the data and transform it in one step.

# Arguments
- `pipeline::Pipeline`: The pipeline to fit and transform.
- `X::Matrix{Float64}`: Input data matrix.
- `y::Vector{Any}`: Optional target values (default: empty vector).

# Returns
Transformed data matrix.
"""
function fit_transform!(pipeline::Pipeline, X::Matrix{Float64}, y::Vector{Any} = [])
    fit!(pipeline, X, y)
    return transform(pipeline, X)
end

"""
    predict(pipeline::Pipeline, X::Matrix{Float64})

Generate predictions using the pipeline. The data is transformed by all steps, and the
final step (assumed to be an estimator) is used to make predictions.

# Arguments
- `pipeline::Pipeline`: The pipeline to use for prediction.
- `X::Matrix{Float64}`: Input data matrix.

# Returns
A vector of predictions.
"""
function predict(pipeline::Pipeline, X::Matrix{Float64})
    # Transform the input data
    X_transformed = transform(pipeline, X)

    # Get the final step (assumes it's an estimator with a `predict` method)
    final_step = last(values(pipeline.named_steps))
    return predict(final_step, X_transformed)
end


"""
    fit_predict!(pipeline::Pipeline, X::Matrix{Float64}, y::Vector{Any})

Fit the pipeline to the data and predict in one step.

# Arguments
- `pipeline::Pipeline`: The pipeline to fit and predict.
- `X::Matrix{Float64}`: Input data matrix.
- `y::Vector{Any}`: Target values.

# Returns
A vector of predictions.
"""
function fit_predict!(pipeline::Pipeline, X::Matrix{Float64}, y::Vector{Any})
    fit!(pipeline, X, y)
    return predict(pipeline, X)
end

"""
    score(pipeline::Pipeline, X::Matrix{Float64}, y_true::Vector{Any}; metric=nothing)

Evaluate the pipeline by comparing predictions with the ground truth using a user-defined
or default scoring metric. If `metric` is not provided, accuracy is used for classification
and mean squared error (MSE) is used for regression.

# Arguments
- `pipeline::Pipeline`: The pipeline to score.
- `X::Matrix{Float64}`: Input data matrix.
- `y_true::Vector{Any}`: True target values.
- `metric`: A user-defined scoring function of the form `(y_true, y_pred) -> score` (optional).

# Returns
A score computed using the specified or default metric.
"""
function score(pipeline::Pipeline, X::Matrix{Float64}, y_true::Vector{Any}; metric=nothing)
    # Predict using the pipeline
    y_pred = predict(pipeline, X)

    # Use the provided metric if available
    if metric !== nothing
        return metric(y_true, y_pred)
    end

    # Default metrics: choose based on task type
    if typeof(y_true[1]) <: Number  # Regression task
        mse = mean((y_true .- y_pred).^2)
        return mse
    else  # Classification task
        accuracy = sum(y_pred .== y_true) / length(y_true)
        return accuracy
    end
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
