# PreprocessingPipeline

## Transformer
```@docs
fit!(Transformer)
transform(Transformer)
inverse_transform(Transformer)
```

## Pipeline
```@docs
Pipeline
add_step!(Pipeline)
make_pipeline(Pipeline)
fit!(Pipeline)
tranform(Pipeline)
fit_transform!(Pipeline)
```

## Preprocessing
### Scaler
```@docs
BaseScaler
StandardScaler
fit!(StandardScaler)
transform(StandardScaler)
inverse_transform(StandardScaler)
fit_transform!(StandardScaler)
MinMaxScaler
fit!(MinMaxScaler)
transform(MinMaxScaler)
inverse_transform(MinMaxScaler)
fit_transform!(MinMaxScaler)
MaxAbsScaler
fit!(MaxAbsScaler)
transform(MaxAbsScaler)
inverse_transform(MaxAbsScaler)
fit_transform!(MaxAbsScaler)
```
### Normalizer
```@docs
BaseNormalizer
StandardNormalizer
fit!(StandardNormalizer)
transform(StandardNormalizer)
inverse_transform(StandardNormalizer)
fit_transform!(StandardNormalizer)
```

## Feature Extraction

## Missing Values
```@docs
MissingValueTranformer
```

