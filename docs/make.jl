push!(LOAD_PATH, "../src")
using Documenter, PreprocessingPipeline

makedocs(
    sitename = "PreprocessingPipeline.jl",
    format = Documenter.HTML(),
    pages = [
        "Getting Started" => "getting-started.md",
        "index" => "index.md",
        "Pipeline" => "pipeline.md",
        "API Reference" => "api.md"
    ]
    # pages = [
    #     "Home" => "index.md"
    # ]
)

deploydocs(
    repo = "github.com/n1developer-ubt/data-preprocessing-juml.git",
    devbranch = "main",
)
