# using Test
# using .FeatureExtraction
# using .PreprocessingPipeline

@testset "Feature Extraction Tests" begin
    @testset "Tests for raw extraction methods" begin
        @testset "DictVectorizer Tests" begin
            
            # Fitting with mixed data (categorical + numeric)
            dicts = [
                Dict("city" => "New York", "temperature" => 30, "humidity" => 70),
                Dict("city" => "Los Angeles", "temperature" => 25, "humidity" => 50),
                Dict("city" => "New York", "temperature" => 28, "humidity" => 65)
            ]

            dv = DictVectorizer()
            fit!(dv, dicts)

            expected_features = ["city=Los Angeles", "city=New York", "temperature", "humidity"]
           
            @test Set(dv.feature_names) == Set(expected_features)
            
            # Checking transformation (correct matrix output)
            X = transform(dv, dicts)

            expected_X = [
                0.0  1.0  70.0  30.0
                1.0  0.0  50.0  25.0
                0.0  1.0  65.0  28.0
            ]

            @test X ≈ expected_X

            # Handling of unseen keys (should be ignored)
            new_dicts = Dict{String, Any}[
                Dict("city" => "Chicago", "temperature" => 22, "humidity" => 55),
                Dict("city" => "New York", "wind_speed" => 10)
            ]
            X_new = transform(dv, new_dicts)

            expected_X_new = [
                0.0  0.0  55.0  22.0  # "city=Chicago" ignored
                0.0  1.0   0.0   0.0  # "wind_speed" ignored
            ]

            @test X_new ≈ expected_X_new

            # Empty input
            empty_dicts = []
            X_empty = transform(dv, empty_dicts)
            println("🚀 Debugging X type: ", typeof(X_empty))

            @test size(X_empty) == (0, length(dv.feature_names))


            # Consistency between fit_transform! and fit! + transform
            dv_ft = DictVectorizer()
            X_ft = fit_transform!(dv_ft, dicts)
            
            dv_manual = DictVectorizer()
            fit!(dv_manual, dicts)
            X_manual = transform(dv_manual, dicts)

            @test X_ft ≈ X_manual
            @test dv_ft.feature_names == dv_manual.feature_names
        end
    end

    @testset "Tests for text extraction methods" begin
        @testset "Tokenization Tests" begin
            text = ["Hello, World!", "This is a test.", "Tokenization is fun!"]
            expected_tokens = [["hello", "world"], ["this", "is", "a", "test"], ["tokenization", "is", "fun"]]
    
            @test tokenize(text) == expected_tokens
            # Empty String
            @test tokenize([""]) == [[]]
        end
    
        @testset "N-gram Generation Tests" begin
            text = ["I love Julia"]
            
            # Unigram (n=1)
            expected_unigram = [["i", "love", "julia"]]
            @test generate_ngrams(text, 1) == expected_unigram
            
            # Bigram (n=2)
            expected_bigram = [["i love", "love julia"]]
            @test generate_ngrams(text, 2) == expected_bigram
    
            # Trigram (n=3)
            @test generate_ngrams(["short text"], 3) == [["short", "text"]]
            @test_throws ErrorException generate_ngrams(text, 0)
        end
    
        @testset "Vocabulary Extraction Tests" begin
            tokenized_text = [["hello", "world"], ["hello", "julia"], ["test", "world"]]
            expected_vocab = ["hello", "julia", "test", "world"]
    
            @test get_vocabulary(tokenized_text) == expected_vocab
            @test get_vocabulary([]) == []
        end
    
        @testset "Bag-of-Words Tests" begin
            vocabulary = ["hello", "world", "julia", "test"]
            text = ["hello", "hello", "world"]
            expected_bow = [2, 1, 0, 0]

            @test bag_of_words(text, vocabulary) == expected_bow
            @test bag_of_words([], vocabulary) == [0, 0, 0, 0]
            @test bag_of_words(text, []) == []
        end
    end
end



"""using Test
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
"""