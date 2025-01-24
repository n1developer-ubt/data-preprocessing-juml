<<<<<<< HEAD
using PreprocessingPipeline
using Documenter 
=======
push!(LOAD_PATH, "../src")
using Documenter, PreprocessingPipeline
>>>>>>> 6a6200c7d5e273ad9d57894f1a38ee588af76522

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
