using CSV
using DataFrames

#save data as csv file.
#the gap is how often save the number from the data
function save_csv(;data, file_name, labels=[], gap=1)
    step_axis = [i for i=1:gap:size(data)[1]]
    graph = hcat([data[i, :] for i in step_axis]...)'
    graph = hcat(step_axis, graph)
    
    if length(labels)+1 != size(graph)[2]
        symbols = [Symbol(i) for i=1:size(graph)[2]]
    else
        symbols = [Symbol(i) for i in vcat(["step"], labels)]
    end
    
    df = DataFrame(graph, symbols);
    df |> CSV.write(file_name,delim=',')
end