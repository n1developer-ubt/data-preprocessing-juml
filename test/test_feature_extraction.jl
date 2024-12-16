using Test
using PreprocessingLib.FeatureExtraction

@testset "Feature Extraction Tests" begin
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