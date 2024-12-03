using Test
using PreprocessingLib.FeatureExtraction

@testset "Feature Extraction Tests" begin
    data = [1, 2, 3, 4, 5]
    result = extract_feature(data)
    @test result == data
end