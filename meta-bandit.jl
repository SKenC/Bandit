include("algorithm_base.jl")
include("environment.jl")
include("rs.jl")

mutable struct MetaBandit <: Algorithm
    algo::Algorithm
    rewards::Matrix
    sum_reward::Vector
    chosen_nums::Vector{Int}
    sum_devs::Matrix
    mt
    max_mt
    delta
    threshold
    function MetaBandit(algo::Algorithm, steps::Int, delta, threshold)
        arms = algo.env.arm_num
        new(algo, zeros(arms, steps), zeros(arms),zeros(arms), zeros(arms, steps), 0., 0., delta, threshold)
    end
end

function init!(algo::MetaBandit, reset_actval::Bool)
    init!(algo.algo, reset_actval)
    arm_num = algo.algo.env.arm_num
    algo.rewards = zeros(arm_num, size(algo.rewards)[2])
    algo.sum_reward = zeros(arm_num)
    algo.chosen_nums = zeros(arm_num)
    algo.sum_devs = zeros(arm_num, size(algo.sum_devs)[2])
    algo.mt = 0.
    algo.max_mt = 0.
end

function page_hinkley!(mb::MetaBandit, selected::Int)
    step = mb.chosen_nums[selected]
    rewards = mb.rewards[selected, 1:step]

    mb.sum_reward[selected] += rewards[step]

    rlen = length(rewards)
    avg = mb.sum_reward[selected] / rlen

    if step <= 1
        mb.sum_devs[selected, 1] = rewards[1] - avg + mb.delta
    else
        mb.sum_devs[selected, step] =
            mb.sum_devs[selected, step-1] + rewards[step] - avg + mb.delta
    end

    if mb.max_mt < mb.sum_devs[selected, step]
        mb.max_mt = mb.sum_devs[selected, step]
    end

    return mb.max_mt - mb.sum_devs[selected, step]
end

function update!(algo::MetaBandit)
    mb = algo

    #chose arm and update parameter by the algorithm.
    selected, regret, reward =  update!(mb.algo)
    mb.chosen_nums[selected] += 1
    mb.rewards[selected, mb.chosen_nums[selected]] = reward

    ph = page_hinkley!(mb, selected)
    @show ph selected

    if ph > algo.threshold
        println("Environment has chenged at $(mb.chosen_nums[selected])")
    end

    return selected, regret, mb.rewards[mb.chosen_nums[selected]], ph

end
