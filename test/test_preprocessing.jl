using Test
using PreprocessingPipeline
using Statistics

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

        @testset "1 Dimensional Data Test With Pipeline" begin
            X = [1, 2, 3, 4, 5]
            # Test mean strategy in pipeline
            pipeline = make_pipeline("standard_scaler" => StandardScaler())
            fit!(pipeline, X)
            X_transformed = transform(pipeline, X)

            @test isapprox(mean(X_transformed), 0.0)
            @test isapprox(std(X_transformed, corrected=false), 1.0)
        end

        @testset "Multi 1 Dimensional Data Tests" begin
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
        end
    end
end