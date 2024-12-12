


@testset "MissingValueTransformer Pipeline Tests" begin
    # Test data with missing values
    X = Matrix{Any}([1 missing 3; 4 5 6; 7 8 missing])
    
    # Test mean strategy in pipeline
    pipeline = make_pipeline("missing_handler" => MissingValueTransformer("mean"))
    fit!(pipeline, X)
    # X_transformed = transform(pipeline, X)
    # @test !any(ismissing, X_transformed)
    # @test size(X_transformed) == size(X)
    
    # Test drop strategy in pipeline
    # pipeline = make_pipeline("missing_handler" => MissingValueTransformer("drop"))
    # fit!(pipeline, X)
    # X_transformed = transform(pipeline, X)
    # @test !any(ismissing, X_transformed)
    # @test size(X_transformed, 2) == size(X, 2)  # Same number of columns
    # @test size(X_transformed, 1) == 1  # Only one row without missing values
end