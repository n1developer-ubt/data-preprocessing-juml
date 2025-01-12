using Documenter, PreprocessingPipeline

makedocs(
    sitename = "PreprocessingPipeline.jl",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md"
        "Getting Started" => "getting-started.md"
    ]
)

deploydocs(
    repo = "github.com/n1developer-ubt/data-preprocessing-juml.git",
    devbranch = "main",
)
