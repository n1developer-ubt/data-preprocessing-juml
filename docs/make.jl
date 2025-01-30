push!(LOAD_PATH, "../src")
using Documenter, PreprocessingPipeline


makedocs(;
    modules=[PreprocessingPipeline],
    sitename = "PreprocessingPipeline.jl",
    format = Documenter.HTML(;
        canonical="https://github.com/n1developer-ubt/data-preprocessing-juml",
        edit_link="main",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md"
        "Getting Started" => "getting-started.md"
    ],
)

deploydocs(;
    repo = "github.com/n1developer-ubt/data-preprocessing-juml",
    devbranch = "main",
)
