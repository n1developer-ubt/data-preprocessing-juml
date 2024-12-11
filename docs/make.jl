using Documenter, PreprocessingPipeline

makedocs(
    sitename = "PreprocessingPipeline.jl",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(
    repo = "https://github.com/n1developer-ubt/data-preprocessing-juml.git",
)
