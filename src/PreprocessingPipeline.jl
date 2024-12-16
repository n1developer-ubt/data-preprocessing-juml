module PreprocessingPipeline

# import required modules
include("Transformer.jl")
include("exampleTransformer.jl")
include("feature_extraction.jl")
include("missing_value.jl")
include("preprocessing.jl")
include("pipeline.jl")


using .TransformerModule
using .ExampleTransformers
using .PipelineModule
using .FeatureExtraction: extract_feature
using .Preprocessing: normalize
using .MissingValue: handle_missing_value


# export modules
export Pipeline, make_pipeline, fit!, Transformer, transform, predict, fit_transform!, add_step!,
extract_feature,
normalize,
handle_missing_value,
AddTransformer, MultiplyTransformer # TODO remove later

end