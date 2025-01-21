using Test
using .FeatureExtraction

# Test for PCA Feature Extraction
@testset "FeatureExtractionTransformer: PCA" begin
    X = rand(100, 10)  # Random data with 100 samples and 10 features
    transformer = FeatureExtractionTransformer("pca")
    fit!(transformer, X)
    X_transformed = transform(transformer, X)
    
    @test size(X_transformed) == (2, 100)  # PCA should reduce to 2 dimensions by default
    @test typeof(X_transformed) == Matrix{Float64}
end

# Test for Bag-of-Words (BoW) Feature Extraction
@testset "FeatureExtractionTransformer: Bag-of-Words" begin
    text_data = [
        "This is the first sentence.",
        "Here is the second sentence.",
        "And this is the third sentence."
    ]
    transformer = FeatureExtractionTransformer("bow")
    fit!(transformer, text_data)
    X_transformed = transform(transformer, text_data)
    
    @test size(X_transformed)[1] == 3  # Number of documents
    @test size(X_transformed)[2] > 0   # Vocabulary size should be greater than 0
    @test typeof(X_transformed) == Matrix{Float64}
end

# Test for TF-IDF Feature Extraction
@testset "FeatureExtractionTransformer: TF-IDF" begin
    text_data = [
        "apple banana orange",
        "apple apple banana",
        "orange banana apple"
    ]
    transformer = FeatureExtractionTransformer("tfidf")
    fit!(transformer, text_data)
    X_transformed = transform(transformer, text_data)
    
    @test size(X_transformed)[1] == 3  # Number of documents
    @test size(X_transformed)[2] > 0   # Vocabulary size should be greater than 0
    @test typeof(X_transformed) == Matrix{Float64}
end

# Test for N-Grams Feature Extraction
@testset "FeatureExtractionTransformer: N-Grams" begin
    text_data = [
        "this is a test",
        "another test case"
    ]
    transformer = FeatureExtractionTransformer("ngrams")
    fit!(transformer, text_data)
    X_transformed = transform(transformer, text_data)
    
    @test typeof(X_transformed) == Vector{String}
    @test length(X_transformed) > 0  # Ensure n-grams are generated
end

# Test for Flatten Image Feature Extraction
@testset "FeatureExtractionTransformer: Flatten Image" begin
    image = rand(28, 28)  # Example of a 28x28 image
    transformer = FeatureExtractionTransformer("flatten_image")
    fit!(transformer, image)
    X_transformed = transform(transformer, image)
    
    @test size(X_transformed) == (28 * 28, 1)  # Flattened image should be 1D
    @test typeof(X_transformed) == Matrix{Float64}
end

# Test for Variance Filter
@testset "FeatureExtraction: Variance Filter" begin
    X = rand(100, 10)  # Random data
    X[:, 1] .= 0.0  # First column has zero variance
    X_filtered = filter_variance(X, 0.01)
    
    @test size(X_filtered, 2) < size(X, 2)  # At least one column should be removed
    @test typeof(X_filtered) == Matrix{Float64}
end

# Test for Correlation Filter
@testset "FeatureExtraction: Correlation Filter" begin
    X = rand(100, 10)  # Random data
    target = X[:, 1] + rand(100) * 0.1  # Target correlated with the first feature
    X_filtered = filter_correlation(X, target, 0.5)
    
    @test size(X_filtered, 2) > 0  # Ensure some features are selected
    @test typeof(X_filtered) == Matrix{Float64}
end

# Test for Low Cardinality Filter
@testset "FeatureExtraction: Low Cardinality Filter" begin
    X = hcat(rand(1:3, 100, 5), rand(1:10, 100, 2))  # Low and high cardinality columns
    X_filtered = filter_low_cardinality(X, 5)
    
    @test size(X_filtered, 2) < size(X, 2)  # Ensure low cardinality columns are removed
    @test typeof(X_filtered) == Matrix{Int}
end
