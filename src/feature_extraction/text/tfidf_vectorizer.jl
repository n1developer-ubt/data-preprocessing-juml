"""
    TfidfVectorizer

A vectorizer that converts text data into a TF-IDF representation by extracting n-grams, computing term frequencies, and applying inverse document frequency scaling.

# Fields
- `vocabulary::Vector{String}`: The learned vocabulary extracted from the input text data.
- `idf::Vector{Float64}`: The inverse document frequency values for each term in the vocabulary.
- `n_gram_range::Tuple{Int, Int}`: The range of n-grams to extract from the text.

# Examples
```julia
vectorizer = TfidfVectorizer(n_gram_range=(1, 2))
```
"""
mutable struct TfidfVectorizer <: BaseTextExtractor
    vocabulary::Vector{String}
    idf::Vector{Float64}
    n_gram_range::Tuple{Int, Int}

    TfidfVectorizer(; n_gram_range=(1, 1)) = new([], [], n_gram_range)
end

"""
    fit!(tv::TfidfVectorizer, text_data::Vector{String})

Fits the TfidfVectorizer to the given text data and computes the vocabulary and IDF values.

# Arguments
- `tv::TfidfVectorizer`: The TfidfVectorizer instance.
- `text_data::Vector{String}`: A list of text documents.

# Returns
- `TfidfVectorizer`: The updated instance with learned vocabulary and IDF values.
"""
function fit!(tv::TfidfVectorizer, text_data::Vector{String})
    text_tokens = vcat([generate_ngrams(text_data, n) for n in tv.n_gram_range[1]:tv.n_gram_range[2]]...)
    tv.vocabulary = get_vocabulary(text_tokens)
    
    bow_matrix = hcat([bag_of_words(token, tv.vocabulary) for token in text_tokens]...)'
    doc_count = size(bow_matrix, 1)
    df = sum(bow_matrix .> 0, dims=1)
    tv.idf = vec(log.((doc_count .+ 1) ./ (df .+ 1)) .+ 1)  # Formula from scikit-learn 

    return tv
end

"""
    transform(tv::TfidfVectorizer, text_data::Vector{String})

Transforms the given text data into a TF-IDF feature representation.

# Arguments
- `tv::TfidfVectorizer`: The TfidfVectorizer instance.
- `text_data::Vector{String}`: A list of text documents.

# Returns
- `Matrix{Float64}`: The transformed TF-IDF feature matrix.
"""
function transform(tv::TfidfVectorizer, text_data::Vector{String})
    text_tokens = vcat([generate_ngrams(text_data, n) for n in tv.n_gram_range[1]:tv.n_gram_range[2]]...)
    bow_matrix = hcat([bag_of_words(token, tv.vocabulary) for token in text_tokens]...)'
    
    tf = bow_matrix ./ sum(bow_matrix, dims=2)
    tfidf_matrix = tf .* tv.idf'

    return tfidf_matrix
end

"""
    fit_transform!(tv::TfidfVectorizer, text_data::Vector{String})

Fits the TfidfVectorizer and transforms the text data into a TF-IDF matrix.

# Arguments
- `tv::TfidfVectorizer`: The TfidfVectorizer instance.
- `text_data::Vector{String}`: A list of text documents.

# Returns
- `Matrix{Float64}`: The transformed TF-IDF feature matrix.
"""
function fit_transform!(tv::TfidfVectorizer, text_data::Vector{String})
    fit!(tv, text_data)
    return transform(tv, text_data)
end