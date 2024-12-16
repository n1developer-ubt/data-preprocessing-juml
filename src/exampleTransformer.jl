module ExampleTransformers

export MultiplyTransformer, AddTransformer

# Important to extend transformer.fit! and .transform
import ..TransformerModule: Transformer, fit!, transform

# AddTransformer ###

mutable struct AddTransformer <: Transformer
    value::Int

    AddTransformer() = new(0)
end

function fit!(addTrans::AddTransformer, X::Matrix{Any})
    addTrans.value = 2 # Set fixed value for example
    return addTrans
end

function transform(addTrans::AddTransformer, X::Matrix{Any})
    return X .+ addTrans.value
end

function fit!(addTrans::AddTransformer, X::Matrix{Int64})
    addTrans.value = 2 # Set fixed value for example
    return addTrans
end

function transform(addTrans::AddTransformer, X::Matrix{Int64})
    return X .+ addTrans.value
end

# MultiplyTransformer ###

mutable struct MultiplyTransformer <: Transformer
    factor::Int

    MultiplyTransformer() = new(0)
end

function fit!(t::MultiplyTransformer, X::Matrix{Any})
    t.factor = maximum(X)
    return t
end

function transform(t::MultiplyTransformer, X::Matrix{Any})
    return X .* t.factor
end

function fit!(t::MultiplyTransformer, X::Matrix{Int64})
    t.factor = maximum(X)
    return t
end

function transform(t::MultiplyTransformer, X::Matrix{Int64})
    return X .* t.factor
end

end
