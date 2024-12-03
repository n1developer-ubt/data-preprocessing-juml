using Test
using PreprocessingLib.Pipeline

@testset "Pipeline Tests" begin
    stage1 = x -> x .+ 1
    stage2 = x -> x .* 2

    # Create a pipeline
    pipeline = Pipeline(stage1, stage2)

    # Input data
    data = [1, 2, 3, 4, 5]

    # Apply the pipeline
    result = fit_transform!(pipeline, data)

    # Expected result
    expected = (data .+ 1) .* 2

    @test result == expected
end