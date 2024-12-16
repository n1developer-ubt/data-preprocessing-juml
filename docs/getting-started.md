# Getting Started

## Installation

To install the package, use Julia's package manager:

```julia
pkg> add PreprocessingPipeline
```

## How to use
The pipeline consists of a series of transformers. Each transformer has a `fit!` and `transform` method. The `fit!` method is used to fit the transformer to the data, and the `transform` method is used to transform the data.


## Example Basic Usage

### Create a Pipeline
   ```julia
    data = [1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing]
    
    # Test mean strategy in pipeline
    pipeline = make_pipeline("missing_handler" => MissingValueTransformer("mean"))

    fit!(pipeline, data)
    data_transformed = transform(pipeline, data)
   ```

### Chaining Transformers

You can chain multiple transformers in a pipeline:

   ```julia
    # Test mean strategy in pipeline
    pipeline = make_pipeline("missing_handler" => MissingValueTransformer("drop"))
    add_step!(pipeline, "standard_scaler" => StandardScaler())
    add_step!(pipeline, "minmax_scaler" => MinMaxScaler())
   ```

## Available Transformers

   ```julia
    MissingValueTransformer("drop") # strategies: "drop", "mean", "constant"
    StandardScaler()
    MinMaxScaler()
    FeatureExtractionTransformer()
   ```

## Create Custom Transformers

You can create your own transformers by implementing the `Transformer` interface.

```julia
mutable struct AddTransformer <: Transformer
    value::Int

    AddTransformer() = new(0)
end

function fit!(addTrans::AddTransformer, X::Matrix{Int64})
    addTrans.value = 2 # Set fixed value for example
    return addTrans
end

function transform(addTrans::AddTransformer, X::Matrix{Int64})
    return X .+ addTrans.value
end
```

