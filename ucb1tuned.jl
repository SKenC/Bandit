include("algorithm_base.jl")
include("environment.jl")

mutable struct UCB1 <: Algorithm
    env::Environment
    actionValues::Vector{Float64}
    counts::Vector{Int}             #numbers of selection of each arm.
    total_count::Int
    sum_rewards::Vector             #sum of an earned reward of each arm
    sq_sum_rewards::Vector
    tuned::Bool
    #constructor
    function UCB1(env::Environment, tuned::Bool)
        return new( env,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    0,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    tuned)
    end
end

#epsilon greedy
function select_arm(algo::UCB1)
    min, minidx = findmin(algo.actionValues)
    if min == 0
        return minidx
    end
    #return index of maximum value in the action values.
    return greedy(algo)
end

function calc_value(algo::UCB1, selected)
    average = algo.sum_rewards[selected] / algo.counts[selected]
    if algo.tuned
        varience =
            (algo.sq_sum_rewards[selected] / algo.counts[selected]) - average^2
                +  sqrt((2*log(algo.total_count))/algo.counts[selected])
        v = (varience > 0.25) ? varience : 0.25
        algo.actionValues[selected] =
            average + sqrt((log(algo.total_count)/algo.counts[selected]) * v)
    else
        algo.actionValues[selected] =
            average + sqrt((2*log(algo.total_count))/algo.counts[selected])
    end
    #@show algo.actionValues
end

#chose arm and update each parameter.
function update!(algo::UCB1)
    selected = select_arm(algo)
    reward = get_reward(algo.env.arm_pros, selected)

    #update this experiment's current state.
    algo.counts[selected] += 1
    algo.total_count += 1
    algo.sum_rewards[selected] += reward
    algo.sq_sum_rewards[selected] += reward^2

    #calculation of action value and save.
    calc_value(algo, selected)

    #calc regret.
    regret = algo.env.max_pro - algo.env.arm_pros[selected]

    return selected, regret, reward

end
