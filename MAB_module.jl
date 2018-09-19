module MAB_MODULE
    include("environment.jl")
    include("algorithm_base.jl")
    include("eps_greedy.jl")
    include("rs.jl")
    export
        Algorithm,
        Environment,
        Egreedy,
        RS,
        update!,
        init!,
        update_env!,
        update_r!

end
