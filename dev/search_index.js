var documenterSearchIndex = {"docs":
[{"location":"api/#PreprocessingPipeline","page":"API Reference","title":"PreprocessingPipeline","text":"","category":"section"},{"location":"api/#Transformer","page":"API Reference","title":"Transformer","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Transformer","category":"page"},{"location":"api/#PreprocessingPipeline.TransformerModule.Transformer","page":"API Reference","title":"PreprocessingPipeline.TransformerModule.Transformer","text":"Transformer\n\nAbstract type for a step in a pipeline. Defines the essential interface for all transformer types.\n\nMethods to be implemented by subtypes\n\nfit!(::Transformer, X::AbstractArray): Fit the transformer to the data.\ntransform(::Transformer, X::AbstractArray): Transform the data.\ninverse_transform(::Transformer, X::AbstractArray): Inverse transform the data.\nfit_transform!(::Transformer, X::AbstractArray): Fit the transformer to the data and transform it in one step.\n\n\n\n\n\n","category":"type"},{"location":"api/#Pipeline","page":"API Reference","title":"Pipeline","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Pipeline\nadd_step!\nmake_pipeline","category":"page"},{"location":"api/#PreprocessingPipeline.PipelineModule.Pipeline","page":"API Reference","title":"PreprocessingPipeline.PipelineModule.Pipeline","text":"struct Pipeline\n\nA structure representing a machine learning pipeline. The pipeline consists of a series of named steps (transformers) applied sequentially to the input data.\n\nFields\n\nnamed_steps::D: Dictionary of named transformer steps.\nn_features_in_::Int: Number of input features in the dataset.\nfeature_names_in_::V: Names of the input features.\n\n\n\n\n\n","category":"type"},{"location":"api/#PreprocessingPipeline.PipelineModule.add_step!","page":"API Reference","title":"PreprocessingPipeline.PipelineModule.add_step!","text":"add_step!(pipeline::Pipeline{D, V}, name::String, step::T) where {T<:Transformer, D<:AbstractDict{String, T}, V<:AbstractVector{String}}\n\nAdd a transformer step to an existing pipeline.\n\nArguments\n\npipeline::Pipeline{D, V}: The pipeline to which the step will be added.\nname::String: Name of the transformer step.\nstep::T: The transformer to be added.\n\nReturns\n\nNothing. Modifies the pipeline in place.\n\n\n\n\n\n","category":"function"},{"location":"api/#PreprocessingPipeline.PipelineModule.make_pipeline","page":"API Reference","title":"PreprocessingPipeline.PipelineModule.make_pipeline","text":"make_pipeline(steps::Pair{String, <:Transformer}...)\n\nCreate a pipeline from a sequence of named transformer steps.\n\nArguments\n\nsteps::Pair{String, <:Transformer}...: Pairs of step names and transformers.\n\nReturns\n\nA new Pipeline instance.\n\n\n\n\n\n","category":"function"},{"location":"api/#Preprocessing","page":"API Reference","title":"Preprocessing","text":"","category":"section"},{"location":"api/#Scaler","page":"API Reference","title":"Scaler","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"StandardScaler\nMinMaxScaler\nMaxAbsScaler","category":"page"},{"location":"api/#PreprocessingPipeline.Preprocessing.StandardScaler","page":"API Reference","title":"PreprocessingPipeline.Preprocessing.StandardScaler","text":"struct StandardScaler\n\nA structure representing a standard scaler. The standard scaler standardizes features by removing the mean.\n\nFields\n\nmean::Union{Vector{Float64}, Float64, Nothing}: Mean of the features.\nstd::Union{Vector{Float64}, Float64, Nothing}: Standard deviation of the features.\n\n\n\n\n\n","category":"type"},{"location":"api/#PreprocessingPipeline.Preprocessing.MinMaxScaler","page":"API Reference","title":"PreprocessingPipeline.Preprocessing.MinMaxScaler","text":"struct MinMaxScaler\n\nA structure representing a min-max scaler. The min-max scaler scales features to a given range. The range is defined by the feature_range parameter.\n\nFields\n\nfeature_range::Tuple{<:Real, <:Real}: The range to scale the features to.\nmin::Union{Vector{<:Real}, <:Real, Nothing}: Minimum value of the features.\nmax::Union{Vector{<:Real}, <:Real, Nothing}: Maximum value of the features.\n\n\n\n\n\n","category":"type"},{"location":"api/#PreprocessingPipeline.Preprocessing.MaxAbsScaler","page":"API Reference","title":"PreprocessingPipeline.Preprocessing.MaxAbsScaler","text":"MaxAbsScaler\n\nMax Abs Scaler is a scaler that scales the data using the maximum absolute value of the data.\n\nExamples\n\nscaler = MaxAbsScaler()\n\n\n\n\n\n","category":"type"},{"location":"api/#Normalizer","page":"API Reference","title":"Normalizer","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"StandardNormalizer","category":"page"},{"location":"api/#PreprocessingPipeline.Preprocessing.StandardNormalizer","page":"API Reference","title":"PreprocessingPipeline.Preprocessing.StandardNormalizer","text":"StandardNormalizer\n\nStandard Normalizer is a normalizer that normalizes the data using the standard normalization technique.\n\nArguments\n\ntype::String: The type of normalization to apply. It can be one of the following:\n\"l1\": L1 normalization.\n\"l2\": L2 normalization.\n\"max\": Max normalization.\n\nExamples\n\nnormalizer = StandardNormalizer(\"l2\")\n\n\n\n\n\n","category":"type"},{"location":"api/#Feature-Extraction","page":"API Reference","title":"Feature Extraction","text":"","category":"section"},{"location":"api/#Feature-Extraction-from-Dictionaries","page":"API Reference","title":"Feature Extraction from Dictionaries","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"DictVectorizer","category":"page"},{"location":"api/#PreprocessingPipeline.FeatureExtraction.DictVectorizer","page":"API Reference","title":"PreprocessingPipeline.FeatureExtraction.DictVectorizer","text":"DictVectorizer\n\nA vectorizer that converts dictionaries of features into a matrix representation.\n\nExamples\n\nvectorizer = DictVectorizer()\n\n\n\n\n\n","category":"type"},{"location":"api/#Feature-Extraction-from-Text","page":"API Reference","title":"Feature Extraction from Text","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"CountVectorizer\nTfidfTransformer\nTfidfVectorizer","category":"page"},{"location":"api/#PreprocessingPipeline.FeatureExtraction.CountVectorizer","page":"API Reference","title":"PreprocessingPipeline.FeatureExtraction.CountVectorizer","text":"CountVectorizer\n\nA text vectorizer that converts a collection of text documents into a matrix of token counts.\n\nExamples\n\nvectorizer = CountVectorizer()\n\n\n\n\n\n","category":"type"},{"location":"api/#PreprocessingPipeline.FeatureExtraction.TfidfTransformer","page":"API Reference","title":"PreprocessingPipeline.FeatureExtraction.TfidfTransformer","text":"TfidfTransformer\n\nA transformer that converts a matrix of token counts into a TF-IDF representation.\n\nExamples\n\ntransformer = TfidfTransformer()\n\n\n\n\n\n","category":"type"},{"location":"api/#PreprocessingPipeline.FeatureExtraction.TfidfVectorizer","page":"API Reference","title":"PreprocessingPipeline.FeatureExtraction.TfidfVectorizer","text":"TfidfVectorizer\n\nA vectorizer that converts text data into a TF-IDF representation by extracting n-grams, computing term frequencies, and applying inverse document frequency scaling.\n\nExamples\n\nvectorizer = TfidfVectorizer(n_gram_range=(1, 2))\n\n\n\n\n\n","category":"type"},{"location":"api/#Missing-Values","page":"API Reference","title":"Missing Values","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"MissingValueTransformer","category":"page"},{"location":"api/#PreprocessingPipeline.MissingValue.MissingValueTransformer","page":"API Reference","title":"PreprocessingPipeline.MissingValue.MissingValueTransformer","text":"MissingValueTransformer(strategy::String=\"drop\", constant_value::Any=nothing)\n\nTransformer for handling missing values using different strategies:\n\n\"drop\": Remove rows with missing values\n\"mean\": Replace with column means\n\"constant\": Replace with specified value\n\nArguments\n\nstrategy::String: Strategy to handle missing values (\"drop\", \"mean\", or \"constant\")\nconstant_value::Any: Value to use when strategy is \"constant\"\n\nReturns\n\nMissingValueTransformer: A transformer object with the specified strategy and constant value\n\n\n\n\n\n","category":"type"},{"location":"api/#fit!,-transform,-fit_transform!,-inverse_transform","page":"API Reference","title":"fit!, transform, fit_transform!, inverse_transform","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"fit!\ntransform\nfit_transform!\ninverse_transform","category":"page"},{"location":"api/#PreprocessingPipeline.TransformerModule.fit!","page":"API Reference","title":"PreprocessingPipeline.TransformerModule.fit!","text":"fit!(transformer::Transformer, X::AbstractArray) -> Transformer\n\nFits the transformer to the given data X. This method should be implemented for any concrete subtype of Transformer.\n\nArguments\n\ntransformer::Transformer: The transformer instance.\nX::AbstractArray: The input data to fit the transformer.\n\nReturns\n\nThe fitted transformer.\n\n\n\n\n\nfit!(dv::DictVectorizer, dicts::Vector{Dict{String, Any}})\n\nFits the DictVectorizer to extract feature names from the given dictionaries.\n\nArguments\n\ndv::DictVectorizer: The DictVectorizer instance.\ndicts::Vector{Dict{String, Any}}: A list of dictionaries representing data samples.\n\nReturns\n\nDictVectorizer: The updated instance with extracted feature names.\n\n\n\n\n\nfit!(cv::CountVectorizer, text_data::Vector{String})\n\nFits the CountVectorizer to the given text data and learns the vocabulary.\n\nArguments\n\ncv::CountVectorizer: The CountVectorizer instance.\ntext_data::Vector{String}: A list of text documents.\n\nReturns\n\nCountVectorizer: The updated instance with the learned vocabulary.\n\n\n\n\n\nfit!(tfidf::TfidfTransformer, X::Matrix{Float64})\n\nComputes the IDF values for the given Bag-of-Words matrix.\n\nArguments\n\ntfidf::TfidfTransformer: The TfidfTransformer instance.\nX::Matrix{Float64}: The Bag-of-Words matrix.\n\nReturns\n\nTfidfTransformer: The updated transformer with computed IDF values.\n\n\n\n\n\nfit!(tv::TfidfVectorizer, text_data::Vector{String})\n\nFits the TfidfVectorizer to the given text data and computes the vocabulary and IDF values.\n\nArguments\n\ntv::TfidfVectorizer: The TfidfVectorizer instance.\ntext_data::Vector{String}: A list of text documents.\n\nReturns\n\nTfidfVectorizer: The updated instance with learned vocabulary and IDF values.\n\n\n\n\n\nfit!(transformer::MissingValueTransformer, X::Matrix{Any})\n\nFit the transformer to the input data. Updates the transformer's internal state.\n\nArguments\n\ntransformer::MissingValueTransformer: The transformer to fit.\nX::Matrix{Any}: Input data matrix.\n\nReturns\n\nThe updated transformer.\n\n\n\n\n\nfit!(scaler::MaxAbsScaler, X::Vector{<:Real})\n\nFit the max-abs scaler to the data.\n\nArguments\n\nscaler::MaxAbsScaler: An instance of MaxAbsScaler.\nX::Vector{<:Real}: The data to fit the scaler.\n\nReturns\n\nThe fitted MaxAbsScaler.\n\n\n\n\n\nfit!(scaler::MaxAbsScaler, X::Matrix{<:Real})\n\nFit the max-abs scaler to the data.\n\nArguments\n\nscaler::MaxAbsScaler: An instance of MaxAbsScaler.\nX::Matrix{<:Real}: The data to fit the scaler.\n\nReturns\n\nThe fitted MaxAbsScaler.\n\n\n\n\n\nfit!(scaler::MinMaxScaler, X::Vector{<:Real})\n\nFit the min-max scaler to the data.\n\nArguments\n\nscaler::MinMaxScaler: An instance of MinMaxScaler.\nX::Vector{<:Real}: The data to fit the scaler.\n\nReturns\n\nThe fitted MinMaxScaler.\n\n\n\n\n\nfit!(scaler::MinMaxScaler, X::Matrix{<:Real})\n\nFit the min-max scaler to the data.\n\nArguments\n\nscaler::MinMaxScaler: An instance of MinMaxScaler.\nX::Matrix{<:Real}: The data to fit the scaler.\n\nReturns\n\nThe fitted MinMaxScaler.\n\n\n\n\n\nfit!(scaler::StandardScaler, X::Vector{<:Real})\n\nFit the standard scaler to the data.\n\nArguments\n\nscaler::StandardScaler: An instance of StandardScaler.\nX::Vector{<:Real}: The data to fit the scaler.\n\nReturns\n\nThe fitted StandardScaler.\n\n\n\n\n\nfit!(scaler::StandardScaler, X::Matrix{<:Real})\n\nFit the standard scaler to the data.\n\nArguments\n\nscaler::StandardScaler: An instance of StandardScaler.\nX::Matrix{<:Real}: The data to fit the scaler.\n\nReturns\n\nThe fitted StandardScaler.\n\n\n\n\n\nfit!(normalizer::StandardNormalizer, X::Vector{<:Real})\n\nFit the normalizer to the data.\n\nArguments\n\nnormalizer::StandardNormalizer: An instance of StandardNormalizer.\nX::Vector{<:Real}: The data to fit the normalizer.\n\nReturns\n\nThe fitted StandardNormalizer.\n\n\n\n\n\nfit!(normalizer::StandardNormalizer, X::Matrix{<:Real})\n\nFit the normalizer to the data.\n\nArguments\n\nnormalizer::StandardNormalizer: An instance of StandardNormalizer.\nX::Matrix{<:Real}: The data to fit the normalizer.\n\nReturns\n\nThe fitted StandardNormalizer.\n\n\n\n\n\nfit!(encoder::OneHotEncoder, X::Vector{<:String})\n\nFit the encoder to the data.\n\nArguments\n\nencoder::OneHotEncoder: An instance of OneHotEncoder.\nX::Vector{<:String}: The data to fit the encoder.\n\nReturns\n\nThe fitted OneHotEncoder.\n\n\n\n\n\nfit!(pipeline::Pipeline{D, V}, X::Matrix{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}\n\nFit the pipeline to the input data by fitting each step sequentially. Updates the transformers in pipeline.named_steps with the results of their fit! method.\n\nArguments\n\npipeline::Pipeline{D, V}: The pipeline to fit.\nX::Matrix{T}: Input data matrix.\n\nReturns\n\nThe updated pipeline.\n\n\n\n\n\nfit!(pipeline::Pipeline{D, V}, X::Vector{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}\n\nFit the pipeline to the input data by fitting each step sequentially. Updates the transformers in pipeline.named_steps with the results of their fit! method.\n\nArguments\n\npipeline::Pipeline{D, V}: The pipeline to fit.\nX::Vector{T}: Input data vector.\n\nReturns\n\nThe updated pipeline.\n\n\n\n\n\n","category":"function"},{"location":"api/#PreprocessingPipeline.TransformerModule.transform","page":"API Reference","title":"PreprocessingPipeline.TransformerModule.transform","text":"transform(transformer::Transformer, X::AbstractArray) -> AbstractArray\n\nApplies the transformation defined by the transformer to the input data X. This method should be implemented for any concrete subtype of Transformer.\n\nArguments\n\ntransformer::Transformer: The transformer instance.\nX::AbstractArray: The input data to transform.\n\nReturns\n\nA transformed version of X.\n\n\n\n\n\ntransform(cv::CountVectorizer, text_data::Vector{String})\n\nTransforms the given text data into a Bag-of-Words (BoW) matrix.\n\nArguments\n\ncv::CountVectorizer: The CountVectorizer instance.\ntext_data::Vector{String}: A list of text documents.\n\nReturns\n\nMatrix{Float64}: The Bag-of-Words feature matrix.\n\n\n\n\n\ntransform(tfidf::TfidfTransformer, X::Matrix{Float64})\n\nTransforms a Bag-of-Words matrix into a TF-IDF representation.\n\nArguments\n\ntfidf::TfidfTransformer: The fitted TfidfTransformer instance.\nX::Matrix{Float64}: The Bag-of-Words matrix.\n\nReturns\n\nMatrix{Float64}: The transformed TF-IDF matrix.\n\n\n\n\n\ntransform(tv::TfidfVectorizer, text_data::Vector{String})\n\nTransforms the given text data into a TF-IDF feature representation.\n\nArguments\n\ntv::TfidfVectorizer: The TfidfVectorizer instance.\ntext_data::Vector{String}: A list of text documents.\n\nReturns\n\nMatrix{Float64}: The transformed TF-IDF feature matrix.\n\n\n\n\n\ntransform(transformer::MissingValueTransformer, X::Matrix{Any})\n\nTransform the input data by handling missing values according to the chosen strategy.\n\nArguments\n\ntransformer::MissingValueTransformer: The fitted transformer\nX::Matrix{Any}: Input data matrix with potential missing values\n\nReturns\n\nTransformed matrix with missing values handled according to the strategy\n\n\n\n\n\ntransform(scaler::MaxAbsScaler, X::Vector{<:Real})\n\nTransform the data using the max-abs scaler.\n\nArguments\n\nscaler::MaxAbsScaler: An instance of MaxAbsScaler.\nX::Vector{<:Real}: The data to transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\ntransform(scaler::MaxAbsScaler, X::Matrix{<:Real})\n\nTransform the data using the max-abs scaler.\n\nArguments\n\nscaler::MaxAbsScaler: An instance of MaxAbsScaler.\nX::Matrix{<:Real}: The data to transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\ntransform(scaler::MinMaxScaler, X::Vector{<:Real})\n\nTransform the data using the min-max scaler.\n\nArguments\n\nscaler::MinMaxScaler: An instance of MinMaxScaler.\nX::Vector{<:Real}: The data to transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\ntransform(scaler::MinMaxScaler, X::Matrix{<:Real})\n\nTransform the data using the min-max scaler.\n\nArguments\n\nscaler::MinMaxScaler: An instance of MinMaxScaler.\nX::Matrix{<:Real}: The data to transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\ntransform(scaler::StandardScaler, X::Vector{<:Real})\n\nTransform the data using the fitted standard scaler.\n\nArguments\n\nscaler::StandardScaler: An instance of StandardScaler.\nX::Vector{<:Real}: The data to transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\ntransform(scaler::StandardScaler, X::Matrix{<:Real})\n\nTransform the data using the fitted standard scaler.\n\nArguments\n\nscaler::StandardScaler: An instance of StandardScaler.\nX::Matrix{<:Real}: The data to transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\ntransform(normalizer::StandardNormalizer, X::Vector{<:Real})\n\nTransform the data using the fitted normalizer.\n\nArguments\n\nnormalizer::StandardNormalizer: An instance of StandardNormalizer.\nX::Vector{<:Real}: The data to transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\ntransform(normalizer::StandardNormalizer, X::Matrix{<:Real})\n\nTransform the data using the fitted normalizer.\n\nArguments\n\nnormalizer::StandardNormalizer: An instance of StandardNormalizer.\nX::Matrix{<:Real}: The data to transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\ntransform(encoder::OneHotEncoder, X::Vector{<:String})\n\nTransform the data using the encoder.\n\nArguments\n\nencoder::OneHotEncoder: An instance of OneHotEncoder.\nX::Vector{<:String}: The data to transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\ntransform(pipeline::Pipeline{D, V}, X::Matrix{Any}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}\n\nTransform the input data using the pipeline by applying each step sequentially.\n\nArguments\n\npipeline::Pipeline{D, V}: The pipeline to use for transformation.\nX::Matrix{T}: Input data matrix.\n\nReturns\n\nTransformed data matrix.\n\n\n\n\n\ntransform(pipeline::Pipeline{D, V}, X::Vector{Any}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}\n\nTransform the input data using the pipeline by applying each step sequentially.\n\nArguments\n\npipeline::Pipeline{D, V}: The pipeline to use for transformation.\nX::Vector{T}: Input data vector.\n\nReturns\n\nTransformed data vector.\n\n\n\n\n\n","category":"function"},{"location":"api/#PreprocessingPipeline.TransformerModule.fit_transform!","page":"API Reference","title":"PreprocessingPipeline.TransformerModule.fit_transform!","text":"fit_transform!(transformer::Transformer, X::AbstractArray) -> AbstractArray\n\nFits the transformer to the given data X and then also applies the transformation.\n\nArguments\n\ntransformer::Transformer: The transformer instance.\nX::AbstractArray: The input data to fit the transformer.\n\nReturns\n\nA transformed version of X.\n\n\n\n\n\nfit_transform!(dv::DictVectorizer, dicts::Vector{Dict{String, Any}})\n\nFits the DictVectorizer and transforms the given dictionaries into a feature matrix.\n\nArguments\n\ndv::DictVectorizer: The DictVectorizer instance.\ndicts::Vector{Dict{String, Any}}: A list of dictionaries representing data samples.\n\nReturns\n\nMatrix{Float64}: The transformed feature matrix.\n\n\n\n\n\nfit_transform!(cv::CountVectorizer, text_data::Vector{String})\n\nFits the CountVectorizer and transforms the text data into a BoW matrix.\n\nArguments\n\ncv::CountVectorizer: The CountVectorizer instance.\ntext_data::Vector{String}: A list of text documents.\n\nReturns\n\nMatrix{Float64}: The Bag-of-Words feature matrix.\n\n\n\n\n\nfit_transform!(tfidf::TfidfTransformer, X::Matrix{Float64})\n\nFits the transformer and transforms the Bag-of-Words matrix into TF-IDF.\n\nArguments\n\ntfidf::TfidfTransformer: The TfidfTransformer instance.\nX::Matrix{Float64}: The Bag-of-Words matrix.\n\nReturns\n\nMatrix{Float64}: The transformed TF-IDF matrix.\n\n\n\n\n\nfit_transform!(tv::TfidfVectorizer, text_data::Vector{String})\n\nFits the TfidfVectorizer and transforms the text data into a TF-IDF matrix.\n\nArguments\n\ntv::TfidfVectorizer: The TfidfVectorizer instance.\ntext_data::Vector{String}: A list of text documents.\n\nReturns\n\nMatrix{Float64}: The transformed TF-IDF feature matrix.\n\n\n\n\n\nfit_transform!(scaler::MinMaxScaler, X::Vector{<:Real})\n\nFit and transform the data using the min-max scaler.\n\nArguments\n\nscaler::MinMaxScaler: An instance of MinMaxScaler.\nX::Vector{<:Real}: The data to fit and transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\nfit_transform!(scaler::MinMaxScaler, X::Matrix{<:Real})\n\nFit and transform the data using the min-max scaler.\n\nArguments\n\nscaler::MinMaxScaler: An instance of MinMaxScaler.\nX::Matrix{<:Real}: The data to fit and transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\nfit_transform!(scaler::StandardScaler, X::Vector{<:Real})\n\nFit and transform the data using the standard scaler.\n\nArguments\n\nscaler::StandardScaler: An instance of StandardScaler.\nX::Vector{<:Real}: The data to fit and transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\nfit_transform!(scaler::StandardScaler, X::Matrix{<:Real})\n\nFit and transform the data using the standard scaler.\n\nArguments\n\nscaler::StandardScaler: An instance of StandardScaler.\nX::Matrix{<:Real}: The data to fit and transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\nfit_transform!(normalizer::StandardNormalizer, X::Vector{<:Real})\n\nFit and transform the data using the normalizer.\n\nArguments\n\nnormalizer::StandardNormalizer: An instance of StandardNormalizer.\nX::Vector{<:Real}: The data to fit and transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\nfit_transform!(normalizer::StandardNormalizer, X::Matrix{<:Real})\n\nFit and transform the data using the normalizer.\n\nArguments\n\nnormalizer::StandardNormalizer: An instance of StandardNormalizer.\nX::Matrix{<:Real}: The data to fit and transform.\n\nReturns\n\nThe transformed data.\n\n\n\n\n\nfit_transform!(encoder::OneHotEncoder, X::Vector{<:String}, y::Vector{Any})\n\nFit the encoder to the data and transform the data.\n\n# Arguments\n- `encoder::OneHotEncoder`: An instance of `OneHotEncoder`.\n- `X::Vector{<:String}`: The data to fit and transform.\n- `y::Vector{Any}`: The target data.\n\n# Returns\nThe transformed data.\n\n\n\n\n\nfit_transform!(pipeline::Pipeline{D, V}, X::Matrix{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}\n\nFit the pipeline to the data and transform it in one step.\n\nArguments\n\npipeline::Pipeline{D, V}: The pipeline to fit and transform.\nX::Matrix{T}: Input data matrix.\n\nReturns\n\nTransformed data matrix.\n\n\n\n\n\nfit_transform!(pipeline::Pipeline{D, V}, X::Vector{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}\n\nFit the pipeline to the data and transform it in one step.\n\nArguments\n\npipeline::Pipeline{D, V}: The pipeline to fit and transform.\nX::Vector{T}: Input data vector.\n\nReturns\n\nTransformed data vector.\n\n\n\n\n\n","category":"function"},{"location":"api/#PreprocessingPipeline.TransformerModule.inverse_transform","page":"API Reference","title":"PreprocessingPipeline.TransformerModule.inverse_transform","text":"inverse_transform(transformer::Transformer, X::AbstractArray) -> AbstractArray\n\nInverse transform the data using the transformer.\n\nArguments\n\ntransformer::Transformer: The transformer to use for inverse transformation.\nX::AbstractArray: Input data array.\n\nReturns\n\nThe inverse transformed data array.\n\n\n\n\n\ninverse_transform(scaler::MaxAbsScaler, X::Vector{<:Real})\n\nInverse transform the data using the max-abs scaler.\n\nArguments\n\nscaler::MaxAbsScaler: An instance of MaxAbsScaler.\nX::Vector{<:Real}: The data to inverse transform.\n\nReturns\n\nThe inverse transformed data.\n\n\n\n\n\ninverse_transform(scaler::MaxAbsScaler, X::Matrix{<:Real})\n\nInverse transform the data using the max-abs scaler.\n\nArguments\n\nscaler::MaxAbsScaler: An instance of MaxAbsScaler.\nX::Matrix{<:Real}: The data to inverse transform.\n\nReturns\n\nThe inverse transformed data.\n\n\n\n\n\ninverse_transform(scaler::MinMaxScaler, X::Vector{<:Real})\n\nInverse transform the data using the min-max scaler.\n\nArguments\n\nscaler::MinMaxScaler: An instance of MinMaxScaler.\nX::Vector{<:Real}: The data to inverse transform.\n\nReturns\n\nThe inverse transformed data.\n\n\n\n\n\ninverse_transform(scaler::MinMaxScaler, X::Matrix{<:Real})\n\nInverse transform the data using the min-max scaler.\n\nArguments\n\nscaler::MinMaxScaler: An instance of MinMaxScaler.\nX::Matrix{<:Real}: The data to inverse transform.\n\nReturns\n\nThe inverse transformed data.\n\n\n\n\n\ninverse_transform(scaler::StandardScaler, X::Vector{<:Real})\n\nInverse transform the data using the fitted standard scaler.\n\nArguments\n\nscaler::StandardScaler: An instance of StandardScaler.\nX::Vector{<:Real}: The data to inverse transform.\n\nReturns\n\nThe inverse transformed data.\n\n\n\n\n\ninverse_transform(scaler::StandardScaler, X::Matrix{<:Real})\n\nInverse transform the data using the fitted standard scaler.\n\nArguments\n\nscaler::StandardScaler: An instance of StandardScaler.\nX::Matrix{<:Real}: The data to inverse transform.\n\nReturns\n\nThe inverse transformed data.\n\n\n\n\n\ninverse_transform(normalizer::StandardNormalizer, X::Vector{<:Real})\n\nInverse transform the data using the fitted normalizer.\n\nArguments\n\nnormalizer::StandardNormalizer: An instance of StandardNormalizer.\nX::Vector{<:Real}: The data to inverse transform.\n\nReturns\n\nThe inverse transformed data.\n\n\n\n\n\ninverse_transform(normalizer::StandardNormalizer, X::Matrix{<:Real})\n\nInverse transform the data using the fitted normalizer.\n\nArguments\n\nnormalizer::StandardNormalizer: An instance of StandardNormalizer.\nX::Matrix{<:Real}: The data to inverse transform.\n\nReturns\n\nThe inverse transformed data.\n\n\n\n\n\ninverse_transform!(encoder::OneHotEncoder, X::Matrix{<:Real})\n\nInverse transform the data using the encoder.\n\nArguments\n\nencoder::OneHotEncoder: An instance of OneHotEncoder.\nX::Matrix{<:Real}: The data to inverse transform.\n\nReturns\n\nThe inverse transformed data.\n\n\n\n\n\ninverse_transform(pipeline::Pipeline{D, V}, X::Matrix{Any}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}\n\nInverse transform the data using the pipeline by applying each step's inverse transform sequentially in reverse order.\n\nArguments\n\npipeline::Pipeline{D, V}: The pipeline to use for inverse transformation.\nX::Matrix{T}: Input data matrix.\n\nReturns\n\nInverse transformed data matrix.\n\n\n\n\n\ninverse_transform(pipeline::Pipeline{D, V}, X::Vector{T}) where {D<:AbstractDict{String, <:Transformer}, V<:AbstractVector{String}}\n\nInverse transform the data using the pipeline by applying each step's inverse transform sequentially in reverse order.\n\nArguments\n\npipeline::Pipeline{D, V}: The pipeline to use for inverse transformation.\nX::Vector{T}: Input data vector.\n\nReturns\n\nInverse transformed data vector.\n\n\n\n\n\n","category":"function"},{"location":"#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"#Installation","page":"Getting Started","title":"Installation","text":"","category":"section"},{"location":"","page":"Getting Started","title":"Getting Started","text":"To install the package, use Julia's package manager:","category":"page"},{"location":"","page":"Getting Started","title":"Getting Started","text":"pkg> add https://github.com/n1developer-ubt/data-preprocessing-juml","category":"page"},{"location":"#How-to-use","page":"Getting Started","title":"How to use","text":"","category":"section"},{"location":"","page":"Getting Started","title":"Getting Started","text":"The pipeline consists of a series of transformers. Each transformer has a fit! and transform method. The fit! method is used to fit the transformer to the data, and the transform method is used to transform the data.","category":"page"},{"location":"","page":"Getting Started","title":"Getting Started","text":"First build a pipeline with the transformers you want to use. Then fit the pipeline to the data. Finally, transform the data.","category":"page"},{"location":"#Example-Basic-Usage-Example","page":"Getting Started","title":"Example Basic Usage Example","text":"","category":"section"},{"location":"","page":"Getting Started","title":"Getting Started","text":"data = Matrix{Any}([1.0 missing 3.0; 4.0 5.0 6.0; 7.0 8.0 missing])\n\npipe = make_pipeline(\"missing_handler\" => MissingValueTransformer(\"mean\"))\n\n# Fit and transform the pipeline\ndata_transformed = fit_transform!(pipe, data)","category":"page"},{"location":"#Chaining-Transformers-Example","page":"Getting Started","title":"Chaining Transformers Example","text":"","category":"section"},{"location":"","page":"Getting Started","title":"Getting Started","text":"You can chain multiple transformers in a pipeline:","category":"page"},{"location":"","page":"Getting Started","title":"Getting Started","text":"# Sample data\ndata = # your data\n\n# Create pipeline\npipe = make_pipeline(\"encoder\" => OneHotEncoder(), \"scaler\" => StandardScaler())\n\n# Fit and transform the pipeline\ndata_transformed = fit_transform!(pipe, data)","category":"page"},{"location":"#Available-Transformers","page":"Getting Started","title":"Available Transformers","text":"","category":"section"},{"location":"","page":"Getting Started","title":"Getting Started","text":"MissingValueTransformer(\"drop\") # strategies: \"drop\", \"mean\", \"constant\"\nStandardScaler()\nMinMaxScaler()\nFeatureExtractionTransformer(\"bow\") # strategies: \"bow\" (Bag-of-words), \"pca\" (Principal Component Analysis) ","category":"page"},{"location":"#Create-Custom-Transformers","page":"Getting Started","title":"Create Custom Transformers","text":"","category":"section"},{"location":"","page":"Getting Started","title":"Getting Started","text":"You can create your own transformers by implementing the Transformer interface.","category":"page"},{"location":"","page":"Getting Started","title":"Getting Started","text":"mutable struct AddTransformer <: Transformer\n    value::Int\n\n    AddTransformer() = new(0)\nend\n\nfunction fit!(addTrans::AddTransformer, X::Matrix{Int64})\n    addTrans.value = 2 # Set fixed value for example\n    return addTrans\nend\n\nfunction transform(addTrans::AddTransformer, X::Matrix{Int64})\n    return X .+ addTrans.value\nend","category":"page"}]
}
