module Preprocessing

# Scaler
include("scaler/scaler.jl")

# Normalizer
include("normalizer/normalizer.jl")

export StandardNormalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, fit!, transform, fit_transform!, inverse_transform

end