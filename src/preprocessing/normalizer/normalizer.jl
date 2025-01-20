module Normalizers
include("base_normalizer.jl")
include("standard_normalizer.jl")

export BaseNormalizer, StandardNormalizer, fit!, transform, fit_transform!, inverse_transform

end