module Preprocessing

# Scaler
include("scaler/scaler.jl")

# Normalizer
include("normalizer/normalizer.jl")

# Encoder
include("encoders/encoder.jl")


export StandardNormalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, fit!, transform, fit_transform!, inverse_transform, OneHotEncoder

end