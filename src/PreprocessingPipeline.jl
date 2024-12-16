module PreprocessingPipeline

# import required modules
include("Transformer.jl")
include("exampleTransformer.jl")
include("feature_extraction.jl")
include("missing_value.jl")
include("preprocessing/preprocessing.jl")
include("pipeline.jl")


using .TransformerModule
using .ExampleTransformers
using .PipelineModule
using .FeatureExtraction: extract_feature
using .Preprocessing:  Scaler.StandardScaler, Scaler.MinMaxScaler, 
                        Scaler.inverse_transform as inverse_transform,
                        Scaler.fit_transform! as scaler_fit_transform
using .MissingValue: MissingValueTransformer


# export modules
export Pipeline, make_pipeline, fit!, Transformer, transform, predict, fit_transform!, add_step!,
extract_feature,
handle_missing_value, StandardScaler, MinMaxScaler, scaler_fit!, inverse_transform, fit_transform!,
MissingValueTransformer,
AddTransformer, MultiplyTransformer # TODO remove later

end