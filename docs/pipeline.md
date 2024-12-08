# Pipeline
The PipelineModule provides a flexible and extensible structure for building machine learning pipelines in Julia. These pipelines consist of a series of named steps (transformers) applied sequentially to the input data.

## Overview 
`PipeLineStep`

The abstract base type for all steps in the pipeline. This serves as the foundation for specialized types like `Transformer`.

`Transformer`

An abstract type inheriting from PipelineStep that defines the basic transformation methods (`fit!` and `transform`) to be implemented by specific transformer types.

`PipeLine`

A structure representing a machine learning pipeline. The pipeline consists of a series of named steps applied sequentially to the input data.
- Fields:
  - `named_steps::Dict{String, Transformer}`: A dictionary of named transformer steps.
  - `classes_::Vector{String}`: Classes handled by the pipeline (if applicable).
  - `n_features_in_::Int`: Number of input features in the dataset.
  - `feature_names_in_::Vector{String}`: Names of the input features.

## Usage
...