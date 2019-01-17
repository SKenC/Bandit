include("algorithm_base.jl")
include("environment.jl")

mutable struct LSX <: Algorithm
    env::Environment
    actionValues::Vector{Float64}
    counts::Vector{Int}             #numbers of selection of each arm.
    w::Vector{Float64}
    l::Vector{Float64}
    r::Float64
    gamma::Float64
    alpha::Float64
    model::String
    opt::Bool
    #constructor
    function LSX(;env::Environment, alpha::Float64, model="", gamma=1.0, opt=false)
        if opt
            r = opt_r(env.arm_pros)
        else
            r = 0.5
        end

        return new( env,
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    zeros(env.arm_num),
                    r,
                    gamma,
                    alpha,
                    model,
                    opt)
    end
end

function init!(algo::LSX)
    algo.actionValues = zeros(algo.env.arm_num)
    algo.counts = zeros(algo.env.arm_num)
    if algo.opt
        algo.r = opt_r(algo.env.arm_pros)
    else
        algo.r = 0.5
    end

    for i=1:algo.env.arm_num
        algo.w[i] = eps(0.0)
        algo.l[i] = eps(0.0)
    end
end

#epsilon greedy
function select_arm(algo::LSX)
    #return index of maximum value in the action values.
    return greedy(algo)
end

#chose arm and update each parameter.
function update!(algo::LSX)
    _, a_mt = findmax(algo.counts)
    _, a_lt = findmin(algo.counts)
    b_ebar = (algo.l[a_mt] * algo.l[a_lt]) / (algo.l[a_mt] + algo.l[a_lt])
    b_e = (algo.w[a_mt] * algo.w[a_lt]) / (algo.w[a_mt] + algo.w[a_lt])
    n_k = b_ebar + b_e

    for i=1:algo.env.arm_num
        algo.actionValues[i] =
            (algo.w[i] + 2*algo.r*n_k - b_e) / (algo.w[i] + algo.l[i] + n_k)
    end

    selected = select_arm(algo)
    reward = get_reward(algo.env.arm_pros, selected)

    for i=1:algo.env.arm_num
        algo.w[i] = algo.gamma * algo.w[i]
        algo.l[i] = algo.gamma * algo.l[i]
    end

    if reward == 1
        algo.w[selected] += 1
    else
        algo.l[selected] += 1
    end

    #update this experiment's current state.
    algo.counts[selected] += 1

    if !algo.opt
        algo.r = algo.r + algo.alpha * (algo.w[selected]/algo.counts[selected] - algo.r)
    end
    #calc regret.
    regret = algo.env.max_pro - algo.env.arm_pros[selected]

    return selected, regret, reward

end

function opt_r(arms)
    sorted_pro = sort(arms, rev=true)
    r = (sorted_pro[1] + sorted_pro[2]) / 2
    return r
end
