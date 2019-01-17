include("algorithm_base.jl")
include("environment.jl")

mutable struct UCB1 <: Algorithm
    env::Environment
    actionValues::Vector{Float64}
    counts::Vector{Int}             #numbers of selection of each arm.
    total_count::Int
    sum_rewards::Vector             #sum of an earned reward of each arm
    averages::Vector
    variences::Vector
    tuned::Bool
    #constructor
    function UCB1(env::Environment, tuned::Bool)
        return new( env,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    0,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    tuned)
    end
end

function init!(algo::UCB1)
    init_algo!(algo)
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
    
    algo.averages[selected] = algo.sum_rewards[selected] / algo.counts[selected]
    if algo.tuned
        algo.variences[selected] = algo.averages[selected] - algo.averages[selected]^2     
        
        for i=1:algo.env.arm_num
            v = (algo.variences[i] < 1/4) ? algo.variences[i] + sqrt((2*log(algo.total_count))/algo.counts[i]) : 1/4
            algo.actionValues[i] =
                algo.averages[i] + sqrt((log(algo.total_count)/algo.counts[i]) * v)
        end
    else
        algo.actionValues[selected] =
            algo.averages[selected] + sqrt((2*log(algo.total_count))/algo.counts[selected])
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
    

    #calculation of action value and save.
    calc_value(algo, selected)

    #calc regret.
    regret = algo.env.max_pro - algo.env.arm_pros[selected]

    return selected, regret, reward

end
