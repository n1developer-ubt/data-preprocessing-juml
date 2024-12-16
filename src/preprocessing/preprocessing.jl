module Preprocessing

include("scaler/scaler.jl")

export StandardScaler, MinMaxScaler, fit!, transform, fit_transform!, inverse_transform

end