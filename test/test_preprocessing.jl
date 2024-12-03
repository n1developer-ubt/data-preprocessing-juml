using Test
using PreprocessingLib.Preprocessing

@testset "Preprocessing Tests" begin
    data = [1 2 3; 4 5 6; 7 8 9]

    # Normalize the data
    normalized = normalize(data)

    # implement the expected behavior
end