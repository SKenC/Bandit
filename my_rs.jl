include("algorithm_base.jl")
include("environment.jl")

mutable struct MYRS <: Algorithm
    env::Environment
    actionValues::Vector{Float64}
    counts::Vector{Int}             #numbers of selection of each arm.
    sum_rewards::Vector             #sum of an earned reward of each arm
    r::Float64
    alpha::Float64
    test_name::String
    #constructor
    function MYRS(env::Environment, r::Float64, alpha::Float64, test_name="")
        return new( env,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    r,
                    alpha,
                    test_name)
    end
end


function select_arm(algo::MYRS)
    #return index of maximum value in the action values.
    return greedy(algo)
end

function calc_value(algo::MYRS, selected)
    average = algo.sum_rewards[selected] / algo.counts[selected]
    algo.actionValues[selected] = algo.counts[selected] * (average - algo.r)

    if algo.test_name == "merge"
        algo.r = (1. - algo.alpha) * algo.r + algo.alpha * (average - algo.r)
    else
        algo.r += algo.alpha * (average - algo.r)
    end
    #@show algo.r selected
end

#update each variables and calc parameters for epsilon greedy algorithm
function update!(algo::MYRS)
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
function simple_update!(algo::MYRS)
    selected = select_arm(algo)
    reward = get_reward(algo.env.arm_pros, selected)

    #RS = n(E-R) <==> delta RS is only (r-R).
    algo.actionValues[selected] += reward - algo.r

    regret = algo.env.max_pro - algo.env.arm_pros[selected]

    return selected, regret
end
