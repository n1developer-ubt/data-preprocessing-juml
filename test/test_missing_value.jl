


@testset "MissingValueTransformer Pipeline Tests" begin
    # Test data with missing values
    X = Matrix{Any}([1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing])
    
    # Test mean strategy in pipeline
    pipeline = make_pipeline("missing_handler" => MissingValueTransformer("mean"))
    fit!(pipeline, X)
    X_transformed = transform(pipeline, X)
    @test !any(ismissing, X_transformed)
    @test X_transformed == Matrix{Any}([1.0 6.5 3.0; 4.0 5.0 6.0; 7.0 8.0 4.5])
    
    # Test drop strategy in pipeline
    pipeline = make_pipeline("missing_handler" => MissingValueTransformer("drop"))
    fit!(pipeline, X)
    X_transformed = transform(pipeline, X)
    @test !any(ismissing, X_transformed)
    @test size(X_transformed, 2) == size(X, 2)  # Same number of columns
    @test size(X_transformed, 1) == 1  # Only one row without missing values
end