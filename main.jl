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
    # for gamma in [0.8]
    #     alpha = 1/(10^3)
    #     algo_dict["MYRS gamma=$gamma"] = MYRS(env, 1., gamma, alpha)
    # end
    # for gamma in [0.9, 0.8]
    #     algo_dict["MYRS merge gamma=$gamma"] = MYRS(env, 1., gamma, 0.001, "merge")
    # end

    #algo_dict["RS"] = RS(env)
    #algo_dict["UCB1"] = UCB1(env, false)
    #algo_dict["LSX"] = LSX(env)
    # for alpha in [0.1, 0.2]
    #     algo_dict["LSX alpha=$alpha"] = LSX(env=env,
    #                                         alpha=alpha)
    # end
    algo_dict["LSX opt"] = LSX(env=env,
                                        alpha=0.1,
                                        model="opt")

    #rlist = zeros(steps)
    #rsum = 0.
    regret_means, win_means, action_vals = Vector{}(), Vector{}(), Vector{}()
    for algorithm in values(algo_dict)
        regrets, wins = zeros(sim_num, steps), zeros(sim_num, steps)
        action_val = Vector{}()
        @progress for sim in 1:sim_num
            ds = rand(update_num,arm_num)
            update_env!(env, ds[1, :])
            init!(algorithm)
            regret = 0.
            for step in 1:steps-1
                selected, rgt, reward = update!(algorithm)
                #selected, rgt, _ = simple_update!(algorithm)        #for RS algorithm.
                #selected, rgt, _, phs[step] = update!(algorithm)   #for debug.

                #save each parameter.
                regret += rgt
                regrets[sim, step] = regret
                push!(action_val, algorithm.actionValues[:]')

                if selected == env.correct_arm
                    wins[sim, step] = 1
                end
                # if typeof(algorithm) == MYRS
                #     rsum += (reward - algorithm.r)
                #     rlist[step] = rsum#algorithm.actionValues[env.correct_arm]
                # end

                if dynamic
                    if step == update_per
                        @show env.arm_pros[env.correct_arm]
                        ds_idx = div(step, update_per)
                        update_env!(env, ds[ds_idx+1, :])
                        #println("<----------updated------------->")
                        if typeof(algorithm) == RS
                            update_r!(algorithm)
                        end
                    end
                end

            end
        end
        push!(regret_means, [mean(regrets[:, i]) for i=1:steps])
        push!(win_means, [mean(wins[:, i]) for i=1:steps])
        push!(action_vals, vcat(action_val...))
    end

    println("DONE.")
    #@show ds
    #@show action_vals

    graph_data = hcat(win_means...)
    time = Vector{Int}(1:steps)
    #xscale=:log

    #graph_data2 = hcat(action_vals...)
    graph_data2 = hcat(regret_means...)
    #graph_data2 = rlist

    return graph_data, graph_data2, algo_dict
end


@time g1, g2, algo_dict = simulation(sim_num=100,
                            steps=10000,
                            update_per=10000,
                            arm_num=20,
                            dynamic=false)

# step_axis2 = [i for i=1:10:size(g2)[1]]
# graph2 = [g2[i, :] for i in step_axis2]
# graph2 = hcat(graph2...)'
# #plot(1:size(g1)[1], g1, label=["RS","RS_tuned"], title="Accuracy")
# plot(step_axis2, graph2, title="Reference", label=["r","ans_arm pro first","ans_arm pro second"])

step_axis = [i for i=1:10:size(g1)[1]]
graph = [g1[i, :] for i in step_axis]
graph = hcat(graph...)'
#plot(1:size(g1)[1], g1, label=["RS","RS_tuned"], title="Accuracy")
plot(step_axis, graph, title="Accuracy", label=[key for key in keys(algo_dict)], legend=:bottomright)

#plot(1:size(g2)[1], g2[:,5:size(g2)[2]], label=["Arm"*string(i) for i=5:size(g2)[2]], title="RS values")
