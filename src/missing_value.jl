module MissingValue

export handle_missing_value

function handle_missing_value(data; strategy="drop")
    # Check if data is empty
    if isempty(data)
        return data
    end
    
    
    if strategy == "drop"
        # Drop missing values
        return filter(!ismissing, data)


    elseif strategy == "mean"
        # Calculate mean of non-missing values
        valid_values = filter(!ismissing, data)
        if isempty(valid_values)
            return Float64[]
        end
        mean_value = sum(valid_values) / length(valid_values)
        # Replace missing values with mean
        return map(x -> ismissing(x) ? mean_value : x, data)


    else
        throw(ArgumentError("Unknown strategy: $strategy. Supported strategies are: 'drop', 'mean'"))
    end
end

end