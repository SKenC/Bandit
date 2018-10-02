include("algorithm_base.jl")
include("environment.jl")

mutable struct MYRS <: Algorithm
    env::Environment
    actionValues::Vector{Float64}
    counts::Vector{Int}             #numbers of selection of each arm.
    sum_rewards::Vector             #sum of an earned reward of each arm
    r::Float64
    #constructor
    function MYRS(env::Environment)
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
function update_r!(algo::MYRS)
    sorted_pro = sort(algo.env.arm_pros, rev=true)
    algo.r = (sorted_pro[1] + sorted_pro[2]) / 2
end

#epsilon greedy
function select_arm(algo::MYRS)
    #return index of maximum value in the action values.
    positive = algo.actionValues + minimum(alog.actionValues)
    pros_idx = sort!([(idx, val) for (idx, val) in enumerate(positive)], by = x -> x[2])
    r = rand()
    for i=1:length(pros)
        if r < pros_idx[i][2]
            return pros_idx[i][1]
        end
    end
    return greedy(algo)
end

function calc_value(algo::MYRS, selected)
    average = algo.sum_rewards[selected] / algo.counts[selected]
    algo.actionValues[selected] = algo.counts[selected] * (average - algo.r)
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
