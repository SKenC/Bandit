include("algorithm_base.jl")
include("environment.jl")

mutable struct MYRS <: Algorithm
    env::Environment
    actionValues::Vector{Float64}
    counts::Vector{Int}             #numbers of selection of each arm.
    sum_rewards::Vector             #sum of an earned reward of each arm
    average::Vector
    r::Float64
    gamma::Float64
    alpha_r::Float64
    n::Vector
    test_name::String
    #constructor
    function MYRS(env::Environment, r::Float64, gamma::Float64, alpha_r::Float64, test_name="")
        return new( env,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    r,
                    gamma,
                    alpha_r,
                    zeros(env.arm_num),
                    test_name)
    end
end

function init!(algo::MYRS)
    init_algo!(algo)
    algo.r = 1.
end

function select_arm(algo::MYRS)
    #return index of maximum value in the action values.
    return greedy(algo)
end

function calc_value(algo::MYRS, selected, reward)

    if algo.test_name == "merge"
        #algo.n = 1 + algo.gamma * algo.n
        mean = algo.sum_rewards[selected] / algo.counts[selected]
        # algo.average[selected] =
        #     (mean + algo.gamma * algo.n * algo.average[selected]) / (1. + algo.gamma * algo.n)
        algo.average[selected] =
            (mean + algo.gamma * algo.average[selected]) / (1. + algo.gamma)
        algo.r += algo.alpha_r * (algo.average[selected] - algo.r)
    else
        algo.average[selected] = algo.sum_rewards[selected] / algo.counts[selected]
        algo.r += algo.alpha_r * (algo.average[selected] - algo.r)
    end

    algo.n[selected] = algo.gamma * algo.n + 1
    algo.actionValues[selected] = algo.n[selected] * (algo.average[selected] - algo.r)


    #algo.actionValues[selected] = algo.gamma * algo.actionValues[selected]


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
    calc_value(algo, selected, reward)

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
