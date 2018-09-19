include("MAB_module.jl")

using .MAB_MODULE
using Plots
using Statistics


function simulation(sim_num, steps, arm_num=4)
    e_start = 1.
    #A Probability of each arm
    #distribution = [0.3, 0.8]
    #distribution = [0.49, 0.51]
    #distribution = [0.2, 0.4, 0.8, 0.9]
    #ds = [[0.49, 0.51], [0.83, 0.8], [0.51, 0.49]]
    # ds = [[0.2, 0.4, 0.8, 0.9],
    #         [0.9, 0.4, 0.8, 0.2],
    #         [0.2, 0.9, 0.8, 0.4]]
    update_per = 1000
    update_num = convert(Int64, steps/update_per)
    ds = rand(update_num, arm_num)
    @show ds
    ds_idx = 1
    env = Environment(ds[1, :])
    eps_greedy = Egreedy(e_start, env)
    rs = RS(env)

    reward_means = Vector{}()
    win_means = Vector{}()
    for algorithm in [rs, eps_greedy]
        regrets = zeros(sim_num, steps)
        wins = zeros(sim_num, steps)
        for sim in 1:sim_num
            regret = 0.
            init!(algorithm, true)
            @progress for step in 1:steps
                selected, rgt = update!(algorithm)

                #save each parameters.3
                regret += rgt
                regrets[sim, step] = regret

                #@show rgt
                if selected == env.correct_arm
                    wins[sim, step] = 1
                end

                if typeof(algorithm) == Egreedy && algorithm.e > 1/200.
                    algorithm.e -= 1/200.
                end

                if step % update_per == 0
                    ds_idx = (ds_idx % size(ds)[1]) + 1
                    update_env!(env, ds[ds_idx, :])
                    init!(algorithm, false)

                    if typeof(algorithm) == RS
                        update_r!(algorithm)
                    end
                end

                #@show algorithm.actionValues

            end

            percent = sim/sim_num * 100
            #println("$percent % Done.")
        end
        push!(reward_means, [mean(regrets[:, i]) for i=1:steps])
        push!(win_means, [mean(wins[:, i]) for i=1:steps])
        #@show wins
    end


    println("DONE.")

    graph_data = hcat(win_means...)
    time = Vector{Int}(1:steps)
    #xscale=:log
    plot(time, graph_data)

    # for i=1:sim_num
    #     plot(time, regrets[i, :])
    # end
    #plot(time, w_means)
    #@show wins[1, :]

end


@time simulation(100, 4000)
