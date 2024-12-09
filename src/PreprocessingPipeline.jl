module PreprocessingPipeline

# import required modules
include("feature_extraction.jl")
include("missing_value.jl")
include("preprocessing/preprocessing.jl")
include("pipeline.jl")

using .PipelineModule: Pipeline, PipelineStep, Transformer, NoScaler, make_pipeline, fit!, transform, predict, fit_transform!, fit_predict!, score, add_step!
using .FeatureExtraction: extract_feature
import .Preprocessing: StandardScaler, fit! as scaler_fit!, 
                       fit_transform! as scaler_fit_transform!, 
                       transform as scaler_transform, 
                       inverse_transform as scaler_inverse_transform
using .MissingValue: handle_missing_value


# export modules
export Pipeline, PipelineStep, Transformer, NoScaler, make_pipeline, fit!, transform, predict, fit_transform!, fit_predict!, score, add_step!,
extract_feature,
handle_missing_value, StandardScaler, scaler_fit!, scaler_fit_transform!, scaler_transform, scaler_inverse_transform

end