module FeatureExtraction

# Raw data
include("raw/raw.jl")

# Text data
include("text/text.jl")

# using .BaseRawExtractor
# using .BaseTextExtractor

export CountVectorizer, TfidfTransformer, TfidfVectorizer, DictVectorizer, fit!, transform, fit_transform!

end