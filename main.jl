include("MAB_module.jl")

using .MAB_MODULE
using Plots
using Statistics

function simulation(;sim_num::Int, steps::Int, update_per::Int, arm_num=4, dynamic=false)
    #argument checking.
    if dynamic && update_per <= steps && steps % update_per != 0
        println("update number error.")
        return
    end

    update_num = div(steps, update_per)#convert(Int64, steps/update_per)
    env = Environment(arm_num)

    algo_dict = Dict()
    # for gamma in [0.99, 0.999, 0.8, 0.7]
    #     alpha = 1/(10^3)
    #     algo_dict["MYRS gamma=$gamma"] = MYRS(env, 1., gamma, alpha)
    # end
    for gamma in [0.7, 0.5, 0.3]
        algo_dict["MYRS merge gamma=$gamma"] = MYRS(env, 1., gamma, 0.001, "merge")
    end

    algo_dict["RS"] = RS(env)

    #algo_dict["UCB1"] = UCB1(env, false)
    #push!(algo_list, UCB1(env, true))
    #push!(algo_list, Egreedy(0.5, env))
    #rs_mb = MetaBandit(rs, steps, 5*10^-3, 10)
    #ucb1 = UCB1(env)

    reward_means, win_means, action_vals = Vector{}(), Vector{}(), Vector{}()
    for algorithm in values(algo_dict)
        regrets, wins = zeros(sim_num, steps), zeros(sim_num, steps)
        action_val = Vector{}()
        @progress for sim in 1:sim_num
            ds = rand(update_num,arm_num)
            update_env!(env, ds[1, :])
            init!(algorithm)
            regret = 0.
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
                    if step == update_per
                        ds_idx = div(step, update_per)
                        update_env!(env, rand(arm_num))
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
    #@show ds
    #@show action_vals

    graph_data = hcat(win_means...)
    time = Vector{Int}(1:steps)
    #xscale=:log

    graph_data2 = hcat(action_vals...)

    return graph_data, graph_data2, algo_dict
end


@time g1, g2, algo_dict = simulation(sim_num=100,
                            steps=40000,
                            update_per=10000,
                            arm_num=20,
                            dynamic=true)

step_axis = [i for i=1:10:size(g1)[1]]
graph = [g1[i, :] for i in step_axis]
graph = hcat(graph...)'
#plot(1:size(g1)[1], g1, label=["RS","RS_tuned"], title="Accuracy")
plot(step_axis, graph, title="Accuracy", label=[key for key in keys(algo_dict)], legend=:bottomright)

#plot(1:size(g2)[1], g2[:,5:size(g2)[2]], label=["Arm"*string(i) for i=5:size(g2)[2]], title="RS values")
