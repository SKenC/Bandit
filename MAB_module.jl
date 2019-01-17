module MAB_MODULE
    include("environment.jl")
    include("algorithm_base.jl")
    include("eps_greedy.jl")
    include("rs.jl")
    include("my_rs.jl")
    #include("meta-bandit.jl")
    include("ucb1tuned.jl")
    include("lsx.jl")
    include("tompson_sampling.jl")
    export
        Algorithm,
        Environment,
        Egreedy,
        RS,
        MYRS,
        LSX,
        #MetaBandit,
        UCB1,
        Tompson,
        update!,
        init!,
        update_env!,
        update_r!,
        simple_update!

end
