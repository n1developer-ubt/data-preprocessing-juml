
@testset "Test Getting Started" begin

    data = Matrix{Any}([1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing])
    
    pipeline = make_pipeline("missing_handler" => MissingValueTransformer("mean"))

    fit!(pipeline, data)
    data_transformed = transform(pipeline, data)

    @test data_transformed == [1.0 6.5 3.0; 4.0 5.0 6.0; 7.0 8.0 4.5]
end


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
    X = [1.0 2.0 3.0 "cat";
        4.0 5.0 6.0 "dog";
        7.0 8.0 9.0 "cat"]

    # Create transformers
    encoder = OneHotEncoder()
    scaler = StandardScaler()

    # Create pipeline
    pipeline = make_pipeline("encoder" => encoder, "scaler" => scaler)

    # Fit and transform the pipeline
    #X_transformed = fit_transform!(pipeline, X)  TODO fix parameter types

    # Expected output after scaling and one-hot encoding
    expected_output = [-1.2247 -1.2247 -1.2247 1.0 0.0;
                        0.0 0.0 0.0 0.0 1.0;
                        1.2247 1.2247 1.2247 1.0 0.0]

    # Test
    #@test X_transformed ≈ expected_output

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
    add_step!(pipeline, "scaler", scaler)
    add_step!(pipeline, "missing_value_handler", missing_value_handler)

    # Check pipeline length
    @test length(pipeline.named_steps) == 2

    # Fit and transform the pipeline
    #X_transformed = fit_transform!(pipeline, X) TODO

    # Expected output after handling missing values and scaling
    expected_output = [0.0 0.0 0.75;
                        0.5 0.5 0.0;
                        1.0 1.0 1.0]

    # Test transformed data
    #@test X_transformed ≈ expected_output

end
