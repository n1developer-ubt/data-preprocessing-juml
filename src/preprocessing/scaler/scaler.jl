module Scaler

include("base_scaler.jl")
include("standard_scaler.jl")
include("min_max_scaler.jl")
include("max_abs_scaler.jl")

export BaseScaler, StandardScaler, MinMaxScaler, MaxAbsScaler, fit!, transform, fit_transform!, inverse_transform

end