module Preprocessing

include("scaler/scaler.jl")
include("normalizer/normalizer.jl")

export StandardNormalizer, StandardScaler, MinMaxScaler, fit!, transform, fit_transform!, inverse_transform

end