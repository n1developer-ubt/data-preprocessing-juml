module FeatureExtraction

# Raw data
include("raw/raw.jl")

# Text data
include("text/text_utils.jl")
include("text/text.jl")

export get_tokenize, generate_ngrams, get_vocabulary, bag_of_words
export CountVectorizer, TfidfTransformer, TfidfVectorizer, DictVectorizer, fit!, transform, fit_transform!

end