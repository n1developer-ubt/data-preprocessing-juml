"""
    TfidfTransformer

A transformer that converts a matrix of token counts into a TF-IDF representation.

# Examples
```julia
transformer = TfidfTransformer()
```
"""
mutable struct TfidfTransformer <: BaseTextExtractor
    idf::Vector{Float64}

    TfidfTransformer() = new([])
end

"""
    fit!(tfidf::TfidfTransformer, X::Matrix{Float64})

Computes the IDF values for the given Bag-of-Words matrix.

# Arguments
- `tfidf::TfidfTransformer`: The TfidfTransformer instance.
- `X::Matrix{Float64}`: The Bag-of-Words matrix.

# Returns
- `TfidfTransformer`: The updated transformer with computed IDF values.
"""
function fit!(tfidf::TfidfTransformer, X::Matrix{Float64})
    if isempty(X)
        tfidf.idf = Float64[]
        return tfidf
    end
    X = float(X)
    n_documents = size(X, 1)
    df = vec(sum(X .> 0, dims=1))
    tfidf.idf = log.((n_documents .+ 1) ./ (df .+ 1)) .+ 1
    # df = sum(X .> 0, dims=1)
    # tfidf.idf = vec(log.((n_documents .+ 1) ./ (df .+ 1)) .+ 1) # Formula from scikit-learn
    return tfidf
end

"""
    transform(tfidf::TfidfTransformer, X::Matrix{Float64})

Transforms a Bag-of-Words matrix into a TF-IDF representation.

# Arguments
- `tfidf::TfidfTransformer`: The fitted TfidfTransformer instance.
- `X::Matrix{Float64}`: The Bag-of-Words matrix.

# Returns
- `Matrix{Float64}`: The transformed TF-IDF matrix.
"""
function transform(tfidf::TfidfTransformer, X::Matrix{Float64})
    if isempty(X)
        return zeros(Float64, 0, length(tfidf.idf))
    end
    X = float(X)
    tf = X ./ sum(X, dims=2)
    tfidf_matrix = tf .* reshape(tfidf.idf, 1, :)
    return tfidf_matrix
end

"""
    fit_transform!(tfidf::TfidfTransformer, X::Matrix{Float64})

Fits the transformer and transforms the Bag-of-Words matrix into TF-IDF.

# Arguments
- `tfidf::TfidfTransformer`: The TfidfTransformer instance.
- `X::Matrix{Float64}`: The Bag-of-Words matrix.

# Returns
- `Matrix{Float64}`: The transformed TF-IDF matrix.
"""
function fit_transform!(tfidf::TfidfTransformer, X::Matrix{Float64})
    fit!(tfidf, X)
    return transform(tfidf, X)
end