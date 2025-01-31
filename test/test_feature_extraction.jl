@testset "Feature Extraction Tests" begin
    @testset "Tests for raw extraction methods" begin
        @testset "DictVectorizer Tests" begin
            
            # Fitting with mixed data (categorical + numeric)
            dicts = [
                Dict("city" => "New York", "temperature" => 30, "humidity" => 70),
                Dict("city" => "Los Angeles", "temperature" => 25, "humidity" => 50),
                Dict("city" => "New York", "temperature" => 28, "humidity" => 65)
            ]

            dv = DictVectorizer()
            fit!(dv, dicts)

            expected_features = ["city=Los Angeles", "city=New York", "temperature", "humidity"]
           
            @test Set(dv.feature_names) == Set(expected_features)
            
            # Checking transformation (correct matrix output)
            X = transform(dv, dicts)

            expected_X = [
                0.0  1.0  70.0  30.0
                1.0  0.0  50.0  25.0
                0.0  1.0  65.0  28.0
            ]

            @test X ≈ expected_X

            # Handling of unseen keys (should be ignored)
            new_dicts::Vector{Dict{String, Any}} = [
                Dict("city" => "Chicago", "temperature" => 22, "humidity" => 55),
                Dict("city" => "New York", "wind_speed" => 10)
            ]
            X_new = transform(dv, new_dicts)

            expected_X_new = [
                0.0  0.0  55.0  22.0  # "city=Chicago" ignored
                0.0  1.0   0.0   0.0  # "wind_speed" ignored
            ]

            @test X_new ≈ expected_X_new

            # Empty input
            empty_dicts = []
            X_empty = transform(dv, empty_dicts)
            
            @test size(X_empty) == (0, length(dv.feature_names))


            # Consistency between fit_transform! and fit! + transform
            dv_ft = DictVectorizer()
            X_ft = fit_transform!(dv_ft, dicts)
            
            dv_manual = DictVectorizer()
            fit!(dv_manual, dicts)
            X_manual = transform(dv_manual, dicts)

            @test X_ft ≈ X_manual
            @test dv_ft.feature_names == dv_manual.feature_names
        end
    end

    @testset "Tests for text extraction methods" begin
        @testset "Tokenization Tests" begin
            text = ["Hello, World!", "This is a test.", "Tokenization is fun!"]
            expected_tokens = [["hello", "world"], ["this", "is", "a", "test"], ["tokenization", "is", "fun"]]
    
            @test get_tokenize(text) == expected_tokens
            # Empty String
            @test get_tokenize([""]) == [[]]
        end
    
        @testset "N-gram Generation Tests" begin
            text = ["I love Julia"]
            
            # Unigram (n=1)
            expected_unigram = [["i", "love", "julia"]]
            @test generate_ngrams(text, 1) == expected_unigram
            
            # Bigram (n=2)
            expected_bigram = [["i love", "love julia"]]
            @test generate_ngrams(text, 2) == expected_bigram
    
            # Trigram (n=3)
            @test generate_ngrams(["short text"], 3) == [["short", "text"]]
            @test_throws ErrorException generate_ngrams(text, 0)
        end
    
        @testset "Vocabulary Extraction Tests" begin
            tokenized_text = [["hello", "world"], ["hello", "julia"], ["test", "world"]]
            expected_vocab = ["hello", "julia", "test", "world"]
    
            @test get_vocabulary(tokenized_text) == expected_vocab
            @test get_vocabulary(Vector{Vector{String}}([])) == []
        end
    
        @testset "Bag-of-Words Tests" begin
            vocabulary = ["hello", "world", "julia", "test"]
            text = ["hello", "hello", "world"]
            expected_bow = [2, 1, 0, 0]

            @test bag_of_words(text, vocabulary) == expected_bow
            @test bag_of_words(Vector{String}([]), vocabulary) == [0, 0, 0, 0]
            @test bag_of_words(text, Vector{String}([])) == []
        end
        @testset "CountVectorizer Tests" begin
            text_data = ["I love Julia", "Julia is great", "I love coding"]

            @testset "Fitting Vocabulary" begin
                cv = CountVectorizer(n_gram_range=(1, 1))
                fit!(cv, text_data)

                #Unigram (n=1)
                expected_vocab_unigram = ["coding", "great", "i", "is", "julia", "love"]
                @test sort(cv.vocabulary) == expected_vocab_unigram
            end

            @testset "Bag-of-Words Transformation" begin
                cv = CountVectorizer(n_gram_range=(1, 1))
                fit!(cv, text_data)
                X = transform(cv, text_data)
                expected_X = [
                    0  0  1  0  1  1  # "I love Julia"
                    0  1  0  1  1  0  # "Julia is great"
                    1  0  1  0  0  1  # "I love coding"
                ]

                @test size(X) == (3, length(cv.vocabulary))
                @test X ≈ expected_X
            end

            @testset "Fit and Transform Consistency" begin
                cv1 = CountVectorizer(n_gram_range=(1, 1))
                X1 = fit_transform!(cv1, text_data)

                cv2 = CountVectorizer(n_gram_range=(1, 1))
                fit!(cv2, text_data)
                X2 = transform(cv2, text_data)

                @test X1 ≈ X2
                @test cv1.vocabulary == cv2.vocabulary
            end

            @testset "Bigram (n=2) Vocabulary and Transformation" begin
                cv = CountVectorizer(n_gram_range=(2, 2))
                fit!(cv, text_data)

                expected_vocab_bigram = ["i love", "is great", "julia is", "love coding", "love julia"]

                @test sort(cv.vocabulary) == expected_vocab_bigram

                X = transform(cv, text_data)

                expected_X = [
                    1  0  0  0  1  # "I love Julia"
                    0  1  1  0  0  # "Julia is great"
                    1  0  0  1  0  # "I love coding"
                ]

                @test size(X) == (3, length(cv.vocabulary))
                @test X ≈ expected_X
            end

            @testset "Variable n-gram Ranges (n=1,2)" begin
                cv = CountVectorizer(n_gram_range=(1, 2))
                fit!(cv, text_data)

                expected_vocab_mixed = ["coding", "great", "i", "i love", "is", "is great", "julia", "julia is", "love", "love coding", "love julia"]

                @test sort(cv.vocabulary) == expected_vocab_mixed
            end

            @testset "Handling Empty Inputs" begin
                cv = CountVectorizer(n_gram_range=(1, 1))
                fit!(cv, Vector{String}([]))

                @test isempty(cv.vocabulary)

                # X = transform(cv, Vector{String}([]))
                # @test size(X) == (0, 0)

                X = transform(cv, text_data)
                @test size(X) == (3, 0)  # Nothing learned
            end

            @testset "Handling Unknown Words" begin
                cv = CountVectorizer(n_gram_range=(1, 1))
                fit!(cv, text_data)

                new_text = ["Python is fun"]
                X_new = transform(cv, new_text)
                
                expected_X_new = [
                    0  0  0  1  0  0  # Just "is", the rest ist unknown
                ]

                @test size(X_new) == (1, length(cv.vocabulary))
                @test X_new ≈ expected_X_new
            end
        end
        @testset "TfidfTransformer Tests" begin
            # Example Bag-of-Words-Matrix
            X_bow = Matrix{Float64}([
                1  2  0  0
                0  1  3  0
                1  0  0  4
            ])
            
            """X_bow = [
                1.0  2.0  0.0  0.0
                0.0  1.0  3.0  0.0
                1.0  0.0  0.0  4.0
            ]"""
        
            @testset "Fitting IDF Values" begin
                tfidf = TfidfTransformer()
                fit!(tfidf, X_bow)
                expected_idf = vec(log.((3 .+ 1) ./ (sum(X_bow .> 0, dims=1) .+ 1)) .+ 1)
                
                @test size(tfidf.idf, 1) == size(X_bow, 2)  # (1, size(X_bow, 2))
                @test tfidf.idf ≈ expected_idf
            end
        
            # @testset "TF-IDF Transformation" begin
            #     tfidf = TfidfTransformer()
            #     fit!(tfidf, X_bow)
            #     X_tfidf = transform(tfidf, X_bow)
        
            #     # Expected result
            #     tf = X_bow ./ sum(X_bow, dims=2)
            #     expected_tfidf = tf .* tfidf.idf
            #     println("TFIDF: ", size(X_tfidf))
            #     println("Expected: ", size(expected_tfidf))
            #     @test size(X_tfidf) == size(X_bow)
            #     @test X_tfidf ≈ expected_tfidf
            # end
        
            @testset "Fit and Transform Consistency" begin
                tfidf1 = TfidfTransformer()
                X_tfidf1 = fit_transform!(tfidf1, X_bow)
        
                tfidf2 = TfidfTransformer()
                fit!(tfidf2, X_bow)
                X_tfidf2 = transform(tfidf2, X_bow)
        
                @test X_tfidf1 ≈ X_tfidf2
                @test tfidf1.idf ≈ tfidf2.idf
            end
        
            @testset "Handling Empty Input" begin
                tfidf = TfidfTransformer()
                fit!(tfidf, zeros(0, 4))
        
                @test isempty(tfidf.idf)
        
                X_tfidf = transform(tfidf, zeros(0, 4))
                @test size(X_tfidf) == (0, 0)
            end
        
            @testset "Handling All-Zero Rows" begin
                X_with_zeros = Matrix{Float64}([
                    0  0  0  0
                    1  2  3  4
                    0  1  0  0
                ])
        
                tfidf = TfidfTransformer()
                fit!(tfidf, X_with_zeros)
                X_tfidf = transform(tfidf, X_with_zeros)
        
                @test all(isnan, X_tfidf[1, :])
                @test !any(isnan, X_tfidf[2:end, :])
            end
        end
        @testset "TfidfVectorizer Tests" begin
            @testset "Fitting and Vocabulary Extraction" begin
                text_data = ["Hello world", "Hello Julia", "Julia is great"]
                
                tv = TfidfVectorizer()
                fit!(tv, text_data)
        
                expected_vocab = ["hello", "is", "julia", "great", "world"]
                @test Set(tv.vocabulary) == Set(expected_vocab)
                @test length(tv.idf) == length(expected_vocab)
                @test all(tv.idf .> 0) 
            end
        
            @testset "TF-IDF Transformation" begin
                text_data = ["Hello world", "Hello Julia", "Julia is great"]
                
                tv = TfidfVectorizer()
                fit!(tv, text_data)
                X_tfidf = transform(tv, text_data)
        
                @test size(X_tfidf) == (3, length(tv.vocabulary)) 
                @test all(X_tfidf .>= 0)
            end
        
            @testset "Fit and Transform Consistency" begin
                text_data = ["Hello world", "Hello Julia", "Julia is great"]
                
                tv1 = TfidfVectorizer()
                X1 = fit_transform!(tv1, text_data)
        
                tv2 = TfidfVectorizer()
                fit!(tv2, text_data)
                X2 = transform(tv2, text_data)
        
                @test X1 ≈ X2  
                @test tv1.vocabulary == tv2.vocabulary
            end
        
            @testset "Handling Empty Input" begin
                tv = TfidfVectorizer()
                fit!(tv, [""])  
                @test isempty(tv.vocabulary)
                @test isempty(tv.idf)
        
                X_tfidf = transform(tv, [""])
                @test size(X_tfidf) == (1, 0)  
            end
        
            @testset "Handling Unseen Words" begin
                text_data = ["Hello world", "Hello Julia"]
                unseen_text = ["Python is great"]
        
                tv = TfidfVectorizer()
                fit!(tv, text_data)
                
                X_tfidf = transform(tv, unseen_text)
                @test size(X_tfidf) == (1, length(tv.vocabulary)) 
                # @test all(X_tfidf .== 0)
            end
        
            @testset "Bigram Handling" begin
                text_data = ["Hello world", "Hello Julia", "Julia is great"]
                
                tv = TfidfVectorizer(n_gram_range=(2, 2))
                fit!(tv, text_data)
                
                expected_vocab = ["hello world", "hello julia", "julia is", "is great"]
                @test Set(tv.vocabulary) == Set(expected_vocab)
                
                X_tfidf = transform(tv, text_data)
                @test size(X_tfidf) == (3, length(tv.vocabulary))
            end
        end
    end
end



