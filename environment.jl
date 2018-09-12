struct Environment
    arm_pros::Vector{Float64}
    arm_num::Int64
    max_pro
    correct_arm::Int64
    function Environment(arm_pros::Vector{Float64})
            max, max_idx = findmax(arm_pros)
            @show max
            new(arm_pros, length(arm_pros), max, max_idx)
    end
end

function get_reward(pros::Vector, selected::Int)
    if rand() < pros[selected]
        return 1
    else
        return 0
    end
end
