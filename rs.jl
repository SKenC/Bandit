include("algorithm_base.jl")
include("environment.jl")

mutable struct RS <: Algorithm
    env::Environment
    actionValues::Vector{Float64}
    counts::Vector{Int}             #numbers of selection of each arm.
    sum_rewards::Vector             #sum of an earned reward of each arm
    r::Float64
    #constructor
    function RS(env::Environment)
        sorted_pro = sort(env.arm_pros, rev=true)
        r = (sorted_pro[1] + sorted_pro[2]) / 2
        #@show r
        return new( env,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    r)
    end
end

#update reference r by best value.
function update_r!(algo::RS)
    sorted_pro = sort(algo.env.arm_pros, rev=true)
    algo.r = (sorted_pro[1] + sorted_pro[2]) / 2
end


function select_arm(algo::RS)
    #return index of maximum value in the action values.
    return greedy(algo)
end

function calc_value(algo::RS, selected)
    average = algo.sum_rewards[selected] / algo.counts[selected]
    algo.actionValues[selected] = algo.counts[selected] * (average - algo.r)
end

#chose arm and update each parameter.
function update!(algo::RS)
    selected = select_arm(algo)
    reward = get_reward(algo.env.arm_pros, selected)

    #update this experiment's current state.
    algo.counts[selected] += 1
    algo.sum_rewards[selected] += reward

    #calculation of action value and save.
    calc_value(algo, selected)

    #calc regret.
    regret = algo.env.max_pro - algo.env.arm_pros[selected]

    return selected, regret, reward

end

#simple version of the RS Algorithm.
function simple_update!(algo::RS)

    selected = select_arm(algo)
    reward = get_reward(algo.env.arm_pros, selected)

    #RS = n(E-R) <==> delta RS is only (r-R).
    algo.actionValues[selected] += reward - algo.r
    regret = algo.env.max_pro - algo.env.arm_pros[selected]

    return selected, regret, reward
end
