# TODO: Remove
# """
#     tokenize(text_data::Vector{String})

# Tokenize a vector of strings into lowercase words without special characters.

# # Arguments
# - `text_data::Vector{String}`: A vector of strings to tokenize.

# # Returns
# - `Vector{Vector{String}}`: Tokenized text data.
# """
# function tokenize(text_data::Vector{String})
#     text_data = lowercase.(text_data)  
#     text_data = replace.(text_data, r"[^\w\s]" => "")  
#     text_tokens = [split(item) for item in text_data]  
#     return text_tokens
# end

# """
#     generate_ngrams(text_data::Vector{String}, n::Int)

# Generate n-grams from a vector of strings.

# # Arguments
# - `text_data::Vector{String}`: A vector of strings to process.
# - `n::Int`: The size of the n-grams to generate.

# # Returns
# - `Vector{Vector{String}}`: A vector where each document contains its n-grams.
# """
# function generate_ngrams(text_data::Vector{String}, n::Int)
#     tokenized = tokenize(text_data)
#     if n < 1
#         error("n must be >= 1")
#     end
#     ngram_list = Vector{Vector{String}}()  # Leere Liste für jedes Dokument

#     for token in tokenized
#         if length(token) < n
#             push!(ngram_list, token)  # Falls das Token zu kurz ist, übernehme es direkt
#         else
#             ngrams = [join(token[i:i+n-1], " ") for i in 1:(length(token) - n + 1)]
#             push!(ngram_list, ngrams)  # Speichere die N-Gramme für dieses Dokument
#         end
#     end

#     return ngram_list  # Jedes Dokument bleibt in seiner eigenen Liste
# end


# """
#     get_vocabulary(tokenized_data::Vector{Vector{String}})

# Get the unique vocabulary from tokenized data.

# # Arguments
# - `tokenized_data::Vector{Vector{String}}`: Tokenized text data.

# # Returns
# - `Vector{String}`: Unique vocabulary from the tokenized data.
# """
# function get_vocabulary(tokenized_data::Vector{Vector{String}})
#     vocabulary = unique(vcat(tokenized_data...))
#     return sort(vocabulary)
# end

# """
#     bag_of_words(text_data::Vector{String}, vocabulary::Vector{String})

# Generate a Bag-of-Words vector for a given text based on the vocabulary.

# # Arguments
# - `text_data::Vector{String}`: Tokenized text data.
# - `vocabulary::Vector{String}`: Vocabulary to use for the Bag-of-Words.

# # Returns
# - `Vector{Float64}`: Bag-of-Words feature vector.
# """
# function bag_of_words(text_data::Vector{String}, vocabulary::Vector{String})
#     word_count = Dict(word => count(==(word), text_data) for word in vocabulary)
#     return [word_count[word] for word in vocabulary]
# end


