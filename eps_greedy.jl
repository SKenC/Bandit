# include("algorithm_base.jl")
# include("environment.jl")

mutable struct Egreedy <: Algorithm
    e::Float64
    env::Environment
    actionValues::Vector{Float64}
    counts::Vector{Int}             #numbers of selection of each arm.
    sum_rewards::Vector             #sum of an earned reward of each arm

    #constructor
    function Egreedy(e::Float64, env::Environment)
        return new( e,
                    env,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    zeros(env.arm_num))
    end
end

#epsilon greedy
function select_arm(algo::Egreedy)
    #@show algo.actionValues
    if rand() < algo.e
        #@show "random"
        #return a random number from 1 to number of arms
        return rand(1:length(algo.actionValues))
    else
        #idx = greedy(algo.actionValues)
        #@show "greedy=" idx
        #return index of maximum value in the action values.
        return greedy(algo)
    end
end

function calc_value(algo::Egreedy, selected)
    algo.actionValues[selected] = algo.sum_rewards[selected] / algo.counts[selected]
end

#update each variables and calc parameters for epsilon greedy algorithm
function update!(algo::Egreedy)
    selected = select_arm(algo)
    reward = get_reward(algo.env.arm_pros, selected)

    #update this experiment's current state.
    algo.counts[selected] += 1
    algo.sum_rewards[selected] += reward

    #calculation of action value and save.
    calc_value(algo, selected)

    #calc regret.
    regret = algo.env.max_pro - algo.env.arm_pros[selected]

    #@show algo.sum_rewards

    return selected, regret, reward

end
