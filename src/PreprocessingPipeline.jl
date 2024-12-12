module PreprocessingPipeline

# import required modules
include("Transformer.jl")
include("feature_extraction.jl")
include("missing_value.jl")
include("preprocessing.jl")
include("pipeline.jl")


using .TransformerModule
using .PipelineModule
using .FeatureExtraction: extract_feature
using .Preprocessing: normalize
# using .MissingValue


# export modules
export Pipeline, make_pipeline, fit!, Transformer, AddTransformer, transform, predict, fit_transform!, add_step!,
extract_feature,
normalize,
MissingValueTransformer

end