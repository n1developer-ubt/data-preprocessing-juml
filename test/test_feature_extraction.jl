using Test
using PreprocessingLib.FeatureExtraction


@testset "FeatureExtractionTransformer: Bag-of-Words (BoW)" begin
    text_data = [
        "This is the first sentence.", 
        "Here is the second sentence.", 
        "And this is the third sentence."
    ]
 
    pipeline = make_pipeline("feature_extractor" => FeatureExtractionTransformer("bow"))
    fit!(pipeline, text_data)
    X_transformed = transform(pipeline, text_data)

    @test typeof(X_transformed) == DataFrame
    @test size(X_transformed, 1) == 3
    @test size(X_transformed, 2) > 0
end

@testset "FeatureExtractionTransformer: PCA" begin
    text_data = [
        "This is the first sentence.", 
        "Here is the second sentence.", 
        "And this is the third sentence."
    ]
    pipeline = make_pipeline("feature_extractor" => FeatureExtractionTransformer("pca"))
    fit!(pipeline, text_data)
    X_transformed = transform(pipeline, text_data)

    @test typeof(X_transformed) == DataFrame
    @test size(X_transformed, 1) == 3
    @test size(X_transformed, 2) > 0
end

@testset "FeatureExtractionTransformer: Basic" begin
    text_data = [
        "This is the first sentence.", 
        "Here is the second sentence.", 
        "And this is the third sentence."
    ]

    pipeline = make_pipeline("feature_extractor" => FeatureExtractionTransformer("basic"))
    fit!(pipeline, text_data)
    X_transformed = transform(pipeline, text_data)

    @test typeof(X_transformed) == DataFrame
    @test size(X_transformed, 1) == 3
    @test size(X_transformed, 2) > 0
end

@testset "FeatureExtractionTransformer: Invalid Strategy" begin
    # Test invalid strategy in pipeline
    @test_throws ArgumentError begin
        FeatureExtractionTransformer("invalid_strategy")
    end
end

@testset "FeatureExtractionTransformer: Fit and Transform with Matrix Input" begin
    # Test data
    text_data = [
        "This is a test.", 
        "Another test sentence.", 
        "This is yet another test."
    ]

    pipeline = make_pipeline("feature_extractor" => FeatureExtractionTransformer("bow"))
    fit!(pipeline, text_data)
    X_transformed = transform(pipeline, text_data)
    
    @test typeof(X_transformed) == DataFrame
    @test size(X_transformed, 1) == 3
    @test size(X_transformed, 2) > 0
end


# @testset "Feature Extraction Tests: bow" begin
#     text_data = ["This is a test.", "Another test sentence."]
#     pipeline = make_pipeline("feature_extracter" => FeatureExtractionTransformer("bow"))
#     fit!(pipeline, )
# end

@testset "Feature Extraction Functions Tests" begin
    # Tests for extract_feature(text_data::Vector{String}) mithilfe von ChatGPT 4o
    @testset "Extract features from Text Data" begin
        @testset "Base Case" begin
            text_data = ["This is a test.", "Another test sentence."]
            expected_bow = [
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1]
            ]
            result = extract_feature(text_data)
            @test result == expected_bow
        end
        
        @testset "Empty Input" begin
            text_data = []
            expected_bow = []
            result = extract_feature(text_data)
            @test result == expected_bow
        end
    
        @testset "Single Word Input" begin
            text_data = ["Word", "Another"]
            expected_bow = [[1, 0], [0, 1]]
            result = extract_feature(text_data)
            @test result == expected_bow
        end
    
        @testset "Repeated Words" begin
            text_data = ["word word word", "word"]
            expected_bow = [[3], [1]]
            result = extract_feature(text_data)
            @test result == expected_bow
        end
    
        @testset "Case Sensitivity" begin
            text_data = ["Word", "word"]
            expected_bow = [[1], [1]]
            result = extract_feature(text_data)
            @test result == expected_bow
        end
    
        @testset "Special Characters" begin
            text_data = ["Hello, world!", "Hello world"]
            expected_bow = [[1, 1], [1, 1]]
            result = extract_feature(text_data)
            @test result == expected_bow
        end
    
        @testset "Numerical Input" begin
            text_data = ["123", "456", "123 123"]
            expected_bow = [[1, 0], [0, 1], [2, 0]]
            result = extract_feature(text_data)
            @test result == expected_bow
        end
    end

    @testset "tokenize Tests" begin
        @testset "Base Cases" begin
            text_data = ["This is a test.", "Another test sentence."]
            expected_tokens = [["this", "is", "a", "test"], ["another", "test", "sentence"]]
            result = tokenize(text_data)
            @test result == expected_tokens
        end
        
        @testset "Empty Input" begin
            text_data = []
            expected_tokens = []
            result = tokenize(text_data)
            @test result == expected_tokens
        end

        @testset "Single Word Input" begin
            text_data = ["Word", "Another"]
            expected_tokens = [["word"], ["another"]]
            result = tokenize(text_data)
            @test result == expected_tokens
        end
 
        @testset "Case Sensitivity" begin
            text_data = ["Word", "word"]
            expected_tokens = [["word"], ["word"]]
            result = tokenize(text_data)
            @test result == expected_tokens
        end

        @testset "Special Characters" begin
            text_data = ["Hello, world!", "It's a test."]
            expected_tokens = [["hello", "world"], ["its", "a", "test"]]
            result = tokenize(text_data)
            @test result == expected_tokens
        end
    
        @testset "Numerical Input" begin
            text_data = ["123 easy as 456"]
            expected_tokens = [["123", "easy", "as", "456"]]
            result = tokenize(text_data)
            @test result == expected_tokens
        end
    
        @testset "Empty Strings" begin
            text_data = [""]
            expected_tokens = [[]]
            result = tokenize(text_data)
            @test result == expected_tokens
        end
    end
    
end

# @testset "Feature Extraction Tests" begin
#     data = [1, 2, 3, 4, 5]
#     result = extract_feature(data)
#     @test result == data
# end

# @testset "Feature Extraction Tests: bow" begin
#     text_data = ["This is a test.", "Another test sentence."]
#     pipeline = make_pipeline("feature_extracter" => FeatureExtractionTransformer("bow"))
#     fit!(pipeline, )
# end