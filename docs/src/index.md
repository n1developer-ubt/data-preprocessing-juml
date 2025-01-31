# Getting Started

## Installation

To install the package, use Julia's package manager:

```julia
pkg> add https://github.com/n1developer-ubt/data-preprocessing-juml
```

## How to use
The pipeline consists of a series of transformers. Each transformer has a `fit!` and `transform` method. The `fit!` method is used to fit the transformer to the data, and the `transform` method is used to transform the data.

First build a pipeline with the transformers you want to use. Then fit the pipeline to the data. Finally, transform the data.


## Example Basic Usage Example

```julia
data = [1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing]

pipe = make_pipeline("missing_handler" => MissingValueTransformer("mean"))

# Fit and transform the pipeline
data_transformed = fit_transform!(pipe, data)
```

## Chaining Transformers Example

You can chain multiple transformers in a pipeline:

```julia
# Sample data
data = # your data

# Create pipeline
pipe = make_pipeline("encoder" => OneHotEncoder(), "scaler" => StandardScaler())

# Fit and transform the pipeline
data_transformed = fit_transform!(pipe, data)
```

## Available Transformers

```julia
# Missing Value Transformer
MissingValueTransformer("drop") # strategies: "drop", "mean", "constant"

# Scalers
StandardScaler()
MinMaxScaler()
MaxAbsScaler()

# One Hot Encoder
OneHotEncoder()

# Standard Normalizer
StandardNormalizer()

# Feature Extraction Transformer
DictVectorizer()
CountVectorizer()
TfidfTransformer()
TfidfVectorizer()
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

