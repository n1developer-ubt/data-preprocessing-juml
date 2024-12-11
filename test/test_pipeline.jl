
@testset "Pipeline Tests" begin
    X = rand(100, 5)

    # Create pipeline with steps
    pipeline = make_pipeline("scaler" => NoScaler())

    # Fit the pipeline
    fit!(pipeline, X)
    @test pipeline.n_features_in_ == 5
    @test length(pipeline.feature_names_in_) == 5

    # Transform the data
    X_transformed = transform(pipeline, X)
    @test size(X_transformed) == size(X)

    # Fit and transform the data
    X_fit_transformed = fit_transform!(pipeline, X)
    @test size(X_fit_transformed) == size(X)

    # Add an additional step
    add_step!(pipeline, "scaler2", NoScaler())
    fit!(pipeline, X)
    @test length(pipeline.named_steps) == 2
end