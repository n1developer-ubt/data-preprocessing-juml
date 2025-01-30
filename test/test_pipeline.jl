@testset "Pipeline" begin

    @testset "Test Getting Started" begin

        data = Matrix{Any}([1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing])

        # first pipeline to handle missing values
        pipeline = make_pipeline("missing_handler" => MissingValueTransformer("constant", 5.0))

        fit!(pipeline, data)
        data_transformed = transform(pipeline, data)

        # second pipeline to normalize the data
        pipeline = make_pipeline("normalizer" => StandardNormalizer("l2"))

        fit!(pipeline, data_transformed)
        data_transformed = transform(pipeline, data_transformed)

        @test data_transformed == [0.1690308509457033 0.8451542547285166 0.50709255283711; 0.4558423058385518 0.5698028822981898 0.6837634587578276; 0.595879571531124 0.6810052246069989 0.4256282653793743]
    end

    @testset "StandardScaling and OneHotEncoding" begin

        # Sample data
        X = ["monkey", "cat", "dog", "dog", "cat", "monkey"]

        # Create transformers
        encoder = OneHotEncoder()
        scaler = StandardScaler()

        # Create pipeline
        pipeline = make_pipeline("encoder" => encoder, "scaler" => scaler)

        # Fit and transform the pipeline
        X_transformed = fit_transform!(pipeline, X)

        # Expected output after scaling and one-hot encoding
        expected_output = [1.414 -0.707 -0.707;
                           -0.707 1.414 -0.707;
                           -0.707 -0.707 1.414;
                           -0.707 -0.707 1.414;
                           -0.707 1.414 -0.707;
                           1.414 -0.707 -0.707]

        # Test
        @test isapprox(X_transformed, expected_output, atol=0.01)
    end

    @testset "MinMaxScaling and MissingValues" begin

        # Sample data with missing value
        X = [1.0 2.0 missing;
             4.0 5.0 6.0;
             7.0 8.0 9.0]

        # Create transformers
        missing_value_handler = MissingValueTransformer("mean")
        scaler = MinMaxScaler((0, 5))

        # Create an empty pipeline
        pipeline = Pipeline(Dict{String, Transformer}())

        # Add steps to the pipeline
        add_step!(pipeline, "missing_value_handler", missing_value_handler)
        add_step!(pipeline, "scaler", scaler)

        # Check pipeline length
        @test length(pipeline.named_steps) == 2

        # Fit and transform the pipeline
        X_transformed = fit_transform!(pipeline, X)

        # Expected output after handling missing values and scaling
        expected_output = [0.0 0.0 2.5;
                            2.5 2.5 0.0;
                           5.0 5.0 5.0]

        # Test transformed data
        @test X_transformed ≈ expected_output
    end
end





####### Test Abstract Transformer #######

@testset "Abstract Transformer Tests" begin
    # Create a concrete type that doesn't implement the required methods
    struct DummyTransformer <: Transformer end
    
    dummy = DummyTransformer()
    test_matrix = [1 2; 3 4]
    test_vector = [1, 2, 3]

    @testset "fit! method" begin
        @test_throws MethodError fit!(dummy, test_matrix)
        @test_throws MethodError fit!(dummy, test_vector)
    end

    @testset "transform method" begin
        @test_throws MethodError transform(dummy, test_matrix)
        @test_throws MethodError transform(dummy, test_vector)
    end

    @testset "inverse_transform method" begin
        @test_throws MethodError inverse_transform(dummy, test_matrix)
        @test_throws MethodError inverse_transform(dummy, test_vector)
    end

    @testset "fit_transform! method" begin
        @test_throws MethodError fit_transform!(dummy, test_matrix)
        @test_throws MethodError fit_transform!(dummy, test_vector)
    end
end



@testset "Pipeline fit_transform! Tests" begin
    @testset "Matrix Input" begin
        # Test data with missing values
        data = Matrix{Any}([1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing])
        
        # Create pipeline with missing value handler
        pipeline = make_pipeline("missing_handler" => MissingValueTransformer("mean"))
        
        # Test fit_transform!
        transformed_data = fit_transform!(pipeline, data)
        
        # Verify pipeline was fitted
        @test pipeline.n_features_in_ == 3
        @test pipeline.feature_names_in_ == ["feature_1", "feature_2", "feature_3"]
        
        # Verify data was transformed correctly
        @test transformed_data == [1.0 6.5 3.0; 4.0 5.0 6.0; 7.0 8.0 4.5]
    end


    @testset "Vector Input" begin
        # Test vector data
        data = [1.0, missing, 3.0, 4.0, 5.0]
        
        # Create pipeline
        pipeline = make_pipeline("missing_handler" => MissingValueTransformer("mean"))
        
        # This should throw an error
        @test_throws MethodError fit_transform!(pipeline, data)
    end
end