import ...TransformerModule: Transformer, fit!, transform, inverse_transform, fit_transform!

include("base_text_extractor.jl")
include("count_vectorizer.jl")
include("tfidf_transformer.jl")
include("tfidf_vectorizer.jl")