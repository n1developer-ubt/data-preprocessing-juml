module PreprocessingPipeline

# import required modules
include("feature_extraction.jl")
include("missing_value.jl")
include("preprocessing.jl")
include("pipeline.jl")

using .PipelineModule: Pipeline, PipelineStep, Transformer, NoScaler, make_pipeline, fit!, transform, predict, fit_transform!, fit_predict!, score, add_step!
using .FeatureExtraction: extract_feature
using .Preprocessing: normalize
using .MissingValue: handle_missing_value


# export modules
export Pipeline, PipelineStep, Transformer, NoScaler, make_pipeline, fit!, transform, predict, fit_transform!, fit_predict!, score, add_step!,
extract_feature,
normalize,
handle_missing_value

end