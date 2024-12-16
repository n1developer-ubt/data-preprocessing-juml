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
                        Scaler.fit! as scaler_fit!, 
                        Scaler.inverse_transform as scaler_inverse_transform, 
                        Scaler.transform as scaler_transform, 
                        Scaler.fit_transform! as scaler_fit_transform!
using .MissingValue: MissingValueTransformer


# export modules
export Pipeline, make_pipeline, fit!, Transformer, transform, predict, fit_transform!, add_step!,
extract_feature,
handle_missing_value, StandardScaler, MinMaxScaler, scaler_fit!, scaler_inverse_transform, scaler_transform, scaler_fit_transform!,
MissingValueTransformer,
AddTransformer, MultiplyTransformer # TODO remove later


end