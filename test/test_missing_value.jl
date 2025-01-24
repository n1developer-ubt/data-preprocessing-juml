
@testset "MissingValueTransformer: Mean" begin
    # Test data with missing values
    X = Matrix{Any}([1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing])
    
    # Test mean strategy in pipeline
    pipeline = make_pipeline("missing_handler" => MissingValueTransformer("mean"))
    fit!(pipeline, X)
    X_transformed = transform(pipeline, X)
    println(typeof(X_transformed))
    @test !any(ismissing, X_transformed)
    @test X_transformed == Matrix{Float64}([1.0 6.5 3.0; 4.0 5.0 6.0; 7.0 8.0 4.5])
end

@testset "MissingValueTransformer: Drop" begin
    # Test data with missing values
    X = Matrix{Any}([1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing])
    
    # Test drop strategy in pipeline
    pipeline = make_pipeline("missing_handler" => MissingValueTransformer("drop"))
    fit!(pipeline, X)
    X_transformed = transform(pipeline, X)
    @test !any(ismissing, X_transformed)
    @test size(X_transformed, 2) == size(X, 2)  # Same number of columns
    @test size(X_transformed, 1) == 1  # Only one row without missing values
end


@testset "MissingValueTransformer: Constant" begin
    X = Matrix{Any}([1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing])
    
    # Test with numeric constant
    transformer = MissingValueTransformer("constant", 0.0)
    X_transformed = transform(transformer, X)
    @test X_transformed == [1.0 0.0 3.0; 4.0 5.0 6.0; 7.0 8.0 0.0]
    
    # Test with string constant
    X = Matrix{Any}(["a" missing "c"; "d" "e" "f"; "g" "h" missing])
    transformer = MissingValueTransformer("constant", "unknown")
    X_transformed = transform(transformer, X)
    @test X_transformed == ["a" "unknown" "c"; "d" "e" "f"; "g" "h" "unknown"]
end