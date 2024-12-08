using Test
using PreprocessingPipeline

@testset "Missing Value Handling Tests" begin
    data = [1, 2, missing, 4, 5]
    
    # Test drop strategy
    result_drop = handle_missing_value(data, strategy="drop")
    @test result_drop == [1, 2, 4, 5]
    @test !any(ismissing, result_drop)
    
    # Test mean strategy
    result_mean = handle_missing_value(data, strategy="mean")
    @test result_mean == [1, 2, 3, 4, 5]  # mean of [1,2,4,5] is 3
    @test !any(ismissing, result_mean)
    
    # Test invalid strategy
    @test_throws ArgumentError handle_missing_value(data, strategy="invalid")
    
    # Test empty array
    empty_data = []
    @test isempty(handle_missing_value(empty_data, strategy="drop"))
    @test isempty(handle_missing_value(empty_data, strategy="mean"))
    
    # Test array with all missing values
    all_missing = [missing, missing, missing]
    @test isempty(handle_missing_value(all_missing, strategy="drop"))
    @test isempty(handle_missing_value(all_missing, strategy="mean"))
    
    # Test array without missing values
    data_complete = [1, 2, 3, 4, 5]
    @test handle_missing_value(data_complete, strategy="drop") == data_complete
    @test handle_missing_value(data_complete, strategy="mean") == data_complete
end