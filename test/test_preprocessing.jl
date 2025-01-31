
@testset "Preprocessing Tests" begin
    @testset "Standard Scaler Tests" begin
        @testset "1 Dimensional Data Tests" begin
            data = [1, 2, 3, 4, 5]

            scaler = StandardScaler()

            fit!(scaler, data)

            transformed_data = transform(scaler, data)
            
            #the property of the transformed data is that it has a mean of 0 and a standard deviation of 1
            @test isapprox(mean(transformed_data), 0.0)
            @test isapprox(std(transformed_data, corrected=false), 1.0)

            inverse_transformed_data = inverse_transform(scaler, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "1 Dimensional Data Tests Without Fit" begin
            data = [1, 2, 3, 4, 5]

            scaler = StandardScaler()

            @test_throws ArgumentError transform(scaler, data)
        end

        @testset "1 Dimensional Data Test Inverse Transform Without Fit" begin
            data = [1, 2, 3, 4, 5]

            scaler = StandardScaler()

            @test_throws ArgumentError inverse_transform(scaler, data)
        end

        @testset "1 Dimensional Data Test With Pipeline" begin
            X = [1, 2, 3, 4, 5]
            # Test mean strategy in pipeline
            pipeline = make_pipeline("standard_scaler" => StandardScaler())
            fit!(pipeline, X)
            X_transformed = transform(pipeline, X)

            @test isapprox(mean(X_transformed), 0.0)
            @test isapprox(std(X_transformed, corrected=false), 1.0)
        end

        @testset "Multi Dimensional Data Tests" begin
            data = [1 2 3; 4 5 6; 7 8 9]

            scaler = StandardScaler()

            fit!(scaler, data)

            @test isapprox(scaler.mean, mean(data, dims=1)[:])
            @test isapprox(scaler.std, std(data, dims=1, corrected=false)[:])

            transformed_data = transform(scaler, data)
            # the property of the transformed data is that it has a mean of 0 and a standard deviation of 1
            @test isapprox(mean(transformed_data, dims=1), zeros(1, size(data, 2)))
            @test isapprox(std(transformed_data, dims=1, corrected=false), ones(1, size(data, 2)))

            inverse_transformed_data = inverse_transform(scaler, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "Multi Dimensional Data Tests Without Fit" begin
            data = [1 2 3; 4 5 6; 7 8 9]

            scaler = StandardScaler()

            @test_throws ArgumentError transform(scaler, data)
        end

        @testset "Multi Dimensional Data Test Inverse Transform Without Fit" begin
            data = [1 2 3; 4 5 6; 7 8 9]

            scaler = StandardScaler()

            @test_throws ArgumentError inverse_transform(scaler, data)
        end

        @testset "StandardScaler with Mismatched Dimensions" begin
            data = [1 2 3]
            mdata = [1 2; 3 4; 5 6; 7 8]

            scaler = StandardScaler()

            fit!(scaler, data)

            @test_throws ArgumentError transform(scaler, mdata)
        end
    end

    @testset "Min Max Scaler Tests" begin
        @testset "1 Dimensional Data Tests" begin
            data = [1, 2, 3, 4, 5]

            scaler = MinMaxScaler((0, 1))

            fit!(scaler, data)

            transformed_data = transform(scaler, data)

            @test isapprox(minimum(transformed_data), 0.0)
            @test isapprox(maximum(transformed_data), 1.0)

            inverse_transformed_data = inverse_transform(scaler, transformed_data)
            
            @test isapprox(inverse_transformed_data, data)
        end

        @testset "Multi Dimensional Data Tests" begin
            data = [1 2 3; 4 5 6; 7 8 9]

            scaler = MinMaxScaler((0, 1))

            fit!(scaler, data)

            @test isapprox(scaler.max, maximum(data, dims=1)[:])
            @test isapprox(scaler.min, minimum(data, dims=1)[:])

            transformed_data = transform(scaler, data)

            @test isapprox(transformed_data, [0.0 0.0 0.0; 0.5 0.5 0.5; 1.0 1.0 1.0])

            inverse_transformed_data = inverse_transform(scaler, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "Multi Dimensional Data Tests with Range 0" begin
            data = [5 10; 5 20;]

            scaler = MinMaxScaler((0, 1))

            fit!(scaler, data)

            @test isapprox(scaler.max, maximum(data, dims=1)[:])
            @test isapprox(scaler.min, minimum(data, dims=1)[:])

            transformed_data = transform(scaler, data)

            @test isapprox(transformed_data, [0.0 0.0; 0.0 1.0])

            inverse_transformed_data = inverse_transform(scaler, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end
    end

    @testset "Normalizer Tests" begin
        @testset "L1 Normalizer Tests" begin
            data = [1, 2, 3, 4, 5]

            normalizer = StandardNormalizer("l1")

            fit!(normalizer, data)

            transformed_data = transform(normalizer, data)

            @test isapprox(norm(transformed_data, 1), 1.0)

            inverse_transformed_data = inverse_transform(normalizer, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "L2 Normalizer Tests" begin
            data = [1, 2, 3, 4, 5]

            normalizer = StandardNormalizer("l2")

            fit!(normalizer, data)

            transformed_data = transform(normalizer, data)

            @test isapprox(norm(transformed_data, 2), 1.0)

            inverse_transformed_data = inverse_transform(normalizer, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "L2 Normalizer Without Constructur Argument" begin
            data = [1, 2, 3, 4, 5]

            normalizer = StandardNormalizer()

            fit!(normalizer, data)

            transformed_data = transform(normalizer, data)

            @test isapprox(norm(transformed_data, 2), 1.0)

            inverse_transformed_data = inverse_transform(normalizer, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "Max Normalizer Tests" begin
            data = [1, 2, 3, 4, 5]

            normalizer = StandardNormalizer("max")

            fit!(normalizer, data)

            transformed_data = transform(normalizer, data)

            @test isapprox(norm(transformed_data, Inf), 1.0)

            inverse_transformed_data = inverse_transform(normalizer, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "L1 Multi Dimensional Data Tests" begin
            data = [1 2 3; 4 5 6]

            normalizer = StandardNormalizer("l1")

            fit!(normalizer, data)

            transformed_data = transform(normalizer, data)

            @test isapprox([norm(row, 1) for row in eachrow(transformed_data)], ones(2))

            inverse_transformed_data = inverse_transform(normalizer, transform(normalizer, data))

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "L2 Multi Dimensional Data Tests" begin
            data = [1 2 3; 4 5 6]

            normalizer = StandardNormalizer("l2")

            fit!(normalizer, data)

            transformed_data = transform(normalizer, data)

            @test isapprox([norm(row, 2) for row in eachrow(transformed_data)], ones(2))

            inverse_transformed_data = inverse_transform(normalizer, transform(normalizer, data))

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "Max Multi Dimensional Data Tests" begin
            data = [1 2 3; 4 5 6]

            normalizer = StandardNormalizer("max")

            fit!(normalizer, data)

            transformed_data = transform(normalizer, data)

            @test isapprox([norm(row, Inf) for row in eachrow(transformed_data)], ones(2))

            inverse_transformed_data = inverse_transform(normalizer, transform(normalizer, data))

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "Max Multi Dimensional Data Tests" begin
            data = [1 2 3; 4 5 6]

            normalizer = StandardNormalizer("max")

            fit!(normalizer, data)

            transformed_data = transform(normalizer, data)

            @test isapprox([norm(row, Inf) for row in eachrow(transformed_data)], ones(2))

            inverse_transformed_data = inverse_transform(normalizer, transform(normalizer, data))

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "Normalizer Tests With Pipeline" begin
            X = [1 2 3; 4 5 6]
            # Test mean strategy in pipeline
            pipeline = make_pipeline("standard_normalizer" => StandardNormalizer("l2"))
            fit!(pipeline, X)
            X_transformed = transform(pipeline, X)

            @test isapprox([norm(row, 2) for row in eachrow(X_transformed)], ones(2))
        end

        @testset "1D Normalizer With Invalid Type" begin
            data = [1, 2, 3, 4, 5]

            normalizer = StandardNormalizer("test")

            @test_throws ArgumentError fit!(normalizer, data)
        end

        @testset "2D  Normalizer With Invalid Type" begin
            data = [1 2 3; 4 5 6]

            normalizer = StandardNormalizer("test")

            @test_throws ArgumentError fit!(normalizer, data)
        end

        @testset "1D  Normalizer Without Fit" begin
            data = [1, 2, 3, 4, 5, 6]

            normalizer = StandardNormalizer("l2")

            @test_throws ArgumentError transform(normalizer, data)
        end

        @testset "2D  Normalizer Without Fit" begin
            data = [1 2 3; 4 5 6]

            normalizer = StandardNormalizer("l2")

            @test_throws ArgumentError transform(normalizer, data)
        end

        @testset "1D  Normalizer Inverse Transform Without Fit" begin
            data = [1, 2, 3, 4, 5, 6]

            normalizer = StandardNormalizer("l2")

            @test_throws ArgumentError inverse_transform(normalizer, data)
        end

        @testset "2D  Normalizer Inverse Transform Without Fit" begin
            data = [1 2 3; 4 5 6]

            normalizer = StandardNormalizer("l2")

            @test_throws ArgumentError inverse_transform(normalizer, data)
        end
    end

    @testset "Max Abs Scaler Tests" begin
        @testset "1 Dimensional Data Tests" begin
            data = [1, 2, 3, 4, 5]

            scaler = MaxAbsScaler()

            fit!(scaler, data)

            transformed_data = transform(scaler, data)

            @test isapprox(maximum(abs.(transformed_data)), 1.0)

            inverse_transformed_data = inverse_transform(scaler, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end

        @testset "Multi Dimensional Data Tests" begin
            data = [1 2 3; 4 5 6]

            scaler = MaxAbsScaler()

            fit!(scaler, data)

            transformed_data = transform(scaler, data)

            @test isapprox(transformed_data, [0.25 0.4 0.5; 1.0 1.0 1.0])

            inverse_transformed_data = inverse_transform(scaler, transformed_data)

            @test isapprox(inverse_transformed_data, data)
        end
    end

    @testset "One Hot Encoder Tests" begin
        @testset "1 Dimensional Data Tests" begin
            data = ["a", "b", "c", "a", "b"]

            encoder = OneHotEncoder()

            fit!(encoder, data)

            transformed_data = transform(encoder, data)

            @test isapprox(transformed_data, [1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0])

            inverse_transformed_data = inverse_transform(encoder, transformed_data)

            @test inverse_transformed_data == data
        end

        @testset "1 Dimensional Data Test With Predefined Categories" begin
            data = ["a", "b", "c", "a", "b"]

            encoder = OneHotEncoder(["a", "b", "c"], "ignore")

            fit!(encoder, data)

            transformed_data = transform(encoder, data)

            @test isapprox(transformed_data, [1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0])

            inverse_transformed_data = inverse_transform(encoder, transformed_data)

            @test inverse_transformed_data == data
        end

        @testset "1 Dimensional Data Test With Unknown Error" begin
            data = ["a", "b", "c", "a", "b", "d"]

            encoder = OneHotEncoder(["a", "b", "c"], "error")

            fit!(encoder, data)

            @test_throws ArgumentError transform(encoder, data)
        end
    end
end
