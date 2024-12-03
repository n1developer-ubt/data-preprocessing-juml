module Preprocessing
# export the function
export normalize 

# normalize the data
function normalize(data)
    return (data .- mean(data, dims=1)) ./ std(data, dims=1)
end

end