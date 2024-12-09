using Test
using PreprocessingPipeline
using Statistics

@testset "Preprocessing Tests" begin
    @testset "Standard Scaler Tests" begin
        @testset "Vector Tests" begin
            data = [1, 2, 3, 4, 5]

            scaler = StandardScaler()

            scaler_fit!(scaler, data)

            @test isapprox(scaler.mean, 3.0)
            @test isapprox(scaler.std, std(data, corrected=false))

            transformed_data = scaler_transform(scaler, data)

            @test isapprox(transformed_data, (data .- scaler.mean) ./ scaler.std)
        end

        @testset "Tabular Data Tests" begin
            data = [1 2 3; 4 5 6; 7 8 9]

            scaler = StandardScaler()

            scaler_fit!(scaler, data)

            @test isapprox(scaler.mean, mean(data, dims=1)[:])
            @test isapprox(scaler.std, std(data, dims=1, corrected=false)[:])

            transformed_data = scaler_transform(scaler, data)
            @test isapprox(transformed_data, (data .- mean(data, dims=1)[:]') ./ std(data, dims=1, corrected=false)[:]')
        end
    end
end