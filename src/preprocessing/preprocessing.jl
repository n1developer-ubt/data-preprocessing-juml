module Preprocessing

include("scaler/scaler.jl")
include("normalizer/normalizer.jl")
include("encoders/encoder.jl")

export BaseScaler, BaseNormalizer, StandardNormalizer, 
        StandardScaler, MinMaxScaler, MaxAbsScaler, fit!, transform, fit_transform!, inverse_transform,
        BaseEncoder, OneHotEncoder

end