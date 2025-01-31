"""
    CountVectorizer

A text vectorizer that converts a collection of text documents into a matrix of token counts.

# Fields
- `vocabulary::Vector{String}`: The learned vocabulary extracted from the input text data.
- `n_gram_range::Tuple{Int, Int}`: The range of n-grams to extract from the text.

# Examples
```julia
vectorizer = CountVectorizer()
```
"""
mutable struct CountVectorizer <: BaseTextExtractor
    vocabulary::Vector{String}
    n_gram_range::Tuple{Int, Int}

    CountVectorizer(; n_gram_range=(1, 1)) = new([], n_gram_range)
end

"""
    fit!(cv::CountVectorizer, text_data::Vector{String})

Fits the CountVectorizer to the given text data and learns the vocabulary.

# Arguments
- `cv::CountVectorizer`: The CountVectorizer instance.
- `text_data::Vector{String}`: A list of text documents.

# Returns
- `CountVectorizer`: The updated instance with the learned vocabulary.
"""
function fit!(cv::CountVectorizer, text_data::Vector{String})
    text_tokens = vcat([generate_ngrams(text_data, n) for n in cv.n_gram_range[1]:cv.n_gram_range[2]]...)
    cv.vocabulary = get_vocabulary(text_tokens)
    return cv
end

"""
    transform(cv::CountVectorizer, text_data::Vector{String})

Transforms the given text data into a Bag-of-Words (BoW) matrix.

# Arguments
- `cv::CountVectorizer`: The CountVectorizer instance.
- `text_data::Vector{String}`: A list of text documents.

# Returns
- `Matrix{Float64}`: The Bag-of-Words feature matrix.
"""
function transform(cv::CountVectorizer, text_data::Vector{String})
    text_tokens = vcat([generate_ngrams(text_data, n) for n in cv.n_gram_range[1]:cv.n_gram_range[2]]...)
    bow_matrix = hcat([bag_of_words(token, cv.vocabulary) for token in text_tokens]...)'
    return bow_matrix
end

"""
    fit_transform!(cv::CountVectorizer, text_data::Vector{String})

Fits the CountVectorizer and transforms the text data into a BoW matrix.

# Arguments
- `cv::CountVectorizer`: The CountVectorizer instance.
- `text_data::Vector{String}`: A list of text documents.

# Returns
- `Matrix{Float64}`: The Bag-of-Words feature matrix.
"""
function fit_transform!(cv::CountVectorizer, text_data::Vector{String})
    fit!(cv, text_data)
    return transform(cv, text_data)
end