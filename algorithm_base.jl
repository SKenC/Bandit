abstract type Algorithm end

function init!(algo::Algorithm, reset_actval::Bool)
    if reset_actval
        algo.actionValues = zeros(algo.env.arm_num)
    end
    algo.counts = zeros(algo.env.arm_num)
    algo.sum_rewards = zeros(algo.env.arm_num)
end

function greedy(algo::Algorithm)
    max = typemin(eltype(algo.actionValues))  #minimum value.
    max_indices = Vector{Int}()

    #get maximum values and those indices.
    for i=1:length(algo.actionValues)
        if algo.actionValues[i] > max
            max = algo.actionValues[i]
            empty!(max_indices)
            push!(max_indices, i)
        elseif algo.actionValues[i] == max
            push!(max_indices, i)
        end
    end

    max_num = length(max_indices)

    if max_num == 1
        #return maximum index
        return max_indices[1]
    else
        #retrun random index of maximum values
        return max_indices[rand(1:max_num)]
    end
end
