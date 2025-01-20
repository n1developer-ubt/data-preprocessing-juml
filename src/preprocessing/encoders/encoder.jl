module Encoders
include("base_encoder.jl")
include("one_hot_encoder.jl")

export BaseEncoder, OneHotEncoder, fit!, fit_transform!, transform, inverse_transform

end