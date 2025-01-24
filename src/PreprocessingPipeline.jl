module PreprocessingPipeline

using DataFrames: DataFrame
using Statistics: mean, std
using LinearAlgebra: norm

# import required modules
include("Transformer.jl")
include("feature_extraction.jl")
include("missing_value.jl")
include("preprocessing/preprocessing.jl")
include("pipeline.jl")


using .TransformerModule
using .PipelineModule
using .FeatureExtraction
using .Preprocessing
using .MissingValue


# export modules
export Transformer, fit!, transform, inverse_transform, fit_transform!
export Pipeline, make_pipeline, add_step!
export extract_feature
export StandardScaler, MinMaxScaler, MaxAbsScaler, StandardNormalizer
export MissingValueTransformer
export OneHotEncoder

end