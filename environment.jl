mutable struct Environment
    arm_pros::Vector{Float64}
    arm_num::Int64
    max_pro
    correct_arm::Int64
    function Environment(arm_num::Int)
            new(zeros(arm_num), arm_num, 0., 0)
    end
end

function get_reward(pros::Vector, selected::Int)
    if rand() < pros[selected]
        return 1
    else
        return 0
    end
end

#update each arm's reward probabilty. it represent non-steady environment.
function update_env!(env::Environment, arm_pros::Vector)
    env.arm_pros = arm_pros
    env.max_pro, env.correct_arm = findmax(env.arm_pros)
end
