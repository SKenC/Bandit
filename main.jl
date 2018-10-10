include("MAB_module.jl")

using .MAB_MODULE
using Plots
using Statistics

function simulation(sim_num, steps, arm_num=4, dynamic=false)
    update_per = 10000
    #update_num = convert(Int64, steps/update_per)
    #ds = rand(update_num, arm_num)
    ds = [0.3 0.5]
    ds_idx = 1
    env = Environment(ds[1, :])
    #env = Environment([0.49, 0.51])
    #eps_greedy = Egreedy(e_start, env)
    algo_list = Vector{Any}()
    # for i=2:4
    #     push!(algo_list, MYRS(env, 1., 1/(10^i)))
    # end
    #push!(algo_list, RS(env))
    push!(algo_list, UCB1(env, false))
    push!(algo_list, UCB1(env, true))
    #push!(algo_list, Egreedy(0.5, env))
    #rs_mb = MetaBandit(rs, steps, 5*10^-3, 10)
    #ucb1 = UCB1(env)

    reward_means = Vector{}()
    win_means = Vector{}()
    action_vals = Vector{}()
    for algorithm in algo_list
        regrets = zeros(sim_num, steps)
        wins = zeros(sim_num, steps)
        action_val = Vector{}()
        @progress for sim in 1:sim_num
            regret = 0.
            init!(algorithm, true)
            for step in 1:steps

                selected, rgt, _ = update!(algorithm)
                #selected, rgt, _ = simple_update!(algorithm)        #for RS algorithm.
                #selected, rgt, _, phs[step] = update!(algorithm)   #for debug.

                #save each parameter.
                regret += rgt
                regrets[sim, step] = regret
                push!(action_val, algorithm.actionValues[:]')

                if selected == env.correct_arm
                    wins[sim, step] = 1
                end

                if dynamic
                    if step % update_per == 0
                        ds_idx = (ds_idx % size(ds)[1]) + 1
                        update_env!(env, ds[ds_idx, :])
                        #println("<----------updated------------->")
                        if typeof(algorithm) == RS
                            update_r!(algorithm)
                        end
                    end
                end

            end
        end
        push!(reward_means, [mean(regrets[:, i]) for i=1:steps])
        push!(win_means, [mean(wins[:, i]) for i=1:steps])
        push!(action_vals, vcat(action_val...))
    end

    println("DONE.")
    @show ds
    #@show action_vals

    graph_data = hcat(win_means...)
    time = Vector{Int}(1:steps)
    #xscale=:log

    graph_data2 = hcat(action_vals...)
    #plot(
        #plot(time, graph_data, label=["RS","RS_tuned"], title="Accuracy")
        #plot(time, graph_data2[:,5:8], label=[string(i) for i=5:size(graph_data2)[2]], title="RS values")
    #)
    return graph_data, graph_data2
end


@time g1, g2 = simulation(100, 3000, 4, false)

#plot(1:size(g1)[1], g1, label=["RS","RS_tuned"], title="Accuracy")
plot(1:size(g1)[1], g1, label=[i for i=1:size(g1)[2]], title="Accuracy")

#plot(1:size(g2)[1], g2[:,5:size(g2)[2]], label=["Arm"*string(i) for i=5:size(g2)[2]], title="RS values")
