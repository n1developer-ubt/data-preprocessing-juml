
@testset "Pipeline Tests" begin
    X = [1 2 3
        4 5 6]

    X = Matrix{Any}(X) # stupid hack for now TODO

    # Create pipeline with steps
    pipeline = make_pipeline("exampleTransformer" => AddTransformer())

    # Fit the pipeline
    fit!(pipeline, X)
    @test pipeline.n_features_in_ == 3
    @test length(pipeline.feature_names_in_) == 3

    # Transform the data
    X_transformed = transform(pipeline, X)
    X_transformed = Matrix{Any}(X_transformed) # stupid hack for now TODO
    @test size(X_transformed) == size(X)
    # See if AddTransformer was applied
    @test X_transformed == [3 4 5
                            6 7 8]

    # Add another AddTransformer
    add_step!(pipeline, "anotherOne", AddTransformer())
    fit!(pipeline, X)
    @test length(pipeline.named_steps) == 2
    X_transformed = transform(pipeline, X)
    @test X_transformed == [5 6 7
                            8 9 10]
end