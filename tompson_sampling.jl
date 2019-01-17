include("algorithm_base.jl")
include("environment.jl")

using Distributions

mutable struct Tompson <: Algorithm
    env::Environment
    actionValues::Vector{Float64}
    w::Vector{Float64}             #numbers of selection of each arm.
    l::Vector{Float64}             #sum of an earned reward of each arm
    #constructor
    function Tompson(env::Environment)
        return new( env,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    zeros(env.arm_num))
    end
end

function init!(algo::Tompson)
    algo.actionValues = zeros(algo.env.arm_num)
    algo.w = zeros(algo.env.arm_num)
    algo.l = zeros(algo.env.arm_num)
end


function select_arm(algo::Tompson)
#     min, minidx = findmin(algo.actionValues)
#     if min == 0
#         return minidx
#     end
    #return index of maximum value in the action values.
    return greedy(algo)
end

function calc_value(algo::Tompson, selected)
    
    #beta_dist = Beta(algo.w[selected]+1., algo.l[selected]+1.)
    beta_dists = [Beta(algo.w[i]+1., algo.l[i]+1.) for i=1:algo.env.arm_num]
    for i=1:algo.env.arm_num
        algo.actionValues[i] = rand(beta_dists[i])
    end
    
end

#chose arm and update each parameter.
function update!(algo::Tompson)
    selected = select_arm(algo)
    reward = get_reward(algo.env.arm_pros, selected)

    #calculation of action value and save.
    calc_value(algo, selected)
    
        #update this experiment's current state.
    if reward == 1
        algo.w[selected] += 1.
    else
        algo.l[selected] += 1.
    end

    #calc regret.
    regret = algo.env.max_pro - algo.env.arm_pros[selected]

    return selected, regret, reward

end
