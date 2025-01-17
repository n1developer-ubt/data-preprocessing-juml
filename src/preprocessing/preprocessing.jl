module Preprocessing

include("scaler/scaler.jl")
include("normalizer/normalizer.jl")

export BaseScaler, BaseNormalizer, StandardNormalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, fit!, transform, fit_transform!, inverse_transform

end