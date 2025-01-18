using Test
using PreprocessingPipeline

using Statistics: mean, std
using LinearAlgebra: norm

include("test_feature_extraction.jl")
include("test_missing_value.jl")
# include("test_pipeline.jl") TODO write tests for pipeline with actual transformers
include("test_preprocessing.jl")