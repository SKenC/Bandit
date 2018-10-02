include("MAB_module.jl")

using .MAB_MODULE
using Plots
using Statistics

function simulation(sim_num, steps, arm_num=4)
    e_start = 1.
    #distribution = [0.49, 0.51]
    #distribution = [0.2, 0.4, 0.8, 0.9]
    #ds = [[0.49, 0.51], [0.83, 0.8], [0.51, 0.49]]

    update_per = 500
    update_num = convert(Int64, steps/update_per)
    ds = rand(update_num, arm_num)
    @show ds
    ds_idx = 1
    env = Environment(ds[1, :])
    #eps_greedy = Egreedy(e_start, env)
    rs = RS(env)
    #my_rs = MYRS(env)
    rs_mb = MetaBandit(rs, steps, 5*10^-3, 10)
    #ucb1 = UCB1(env)

    reward_means = Vector{}()
    win_means = Vector{}()
    action_vals = Vector{}()
    phs = zeros(steps)          #for debug
    for algorithm in [rs_mb]
        regrets = zeros(sim_num, steps)
        wins = zeros(sim_num, steps)
        action_val = Vector{}()
        for sim in 1:sim_num
            regret = 0.
            init!(algorithm, true)
            @progress for step in 1:steps
                #selected, rgt, _ = update!(algorithm)
                #selected, rgt, _ = simple_update!(algorithm)
                selected, rgt, _, phs[step] = update!(algorithm)   #for debug.

                #save each parameters.
                regret += rgt
                regrets[sim, step] = regret
                #push!(action_val, algorithm.actionValues[:]')

                if selected == env.correct_arm
                    wins[sim, step] = 1
                end

                # if typeof(algorithm) == Egreedy && algorithm.e > 1/200.
                #     algorithm.e -= 1/200.
                # end

                if step % update_per == 0
                    ds_idx = (ds_idx % size(ds)[1]) + 1
                    update_env!(env, ds[ds_idx, :])
                    #println("<----------updated------------->")
                    if typeof(algorithm) == RS || typeof(algorithm) == MYRS
                        update_r!(algorithm)
                    end
                end

            end
        end
        push!(reward_means, [mean(regrets[:, i]) for i=1:steps])
        push!(win_means, [mean(wins[:, i]) for i=1:steps])
        #push!(action_vals, vcat(action_val...))

    end

    println("DONE.")
    #@show action_vals

    graph_data = phs
    #graph_data = hcat(win_means...)
    #graph_data = hcat(action_vals...)
    time = Vector{Int}(1:steps)
    #xscale=:log
    plot(time, graph_data)

end


@time simulation(1, 1000)
