module PipelineProcessing

# Pipeline struct
struct Pipeline
    stages::Vector{Function}
end

function Pipeline(stages::Function...)
    return Pipeline(collect(stages))
end

# process the input data
function fit_transform!(pipeline::Pipeline, X)
    for stage in pipeline.stages
        X = stage(X)
    end
    return X
end

end