function initial_conditions_numerical(slob::SLOB, pₙ, V₀)
    ud = (-V₀/(2.0*slob.Δx) + slob.D/(slob.Δx^2)) * ones(Float64, slob.M)
    md = ((-2.0*slob.D)/(slob.Δx^2) - slob.nu) * ones(Float64, slob.M+1)
    ld = (V₀/(2.0*slob.Δx) + slob.D/(slob.Δx^2)) * ones(Float64, slob.M)
    A = Tridiagonal(ld, md, ud)

    A[1,2] = 2*slob.D/(slob.Δx^2)
    A[end, end-1] = 2*slob.D/(slob.Δx^2)

    B = [-slob.source_term(xᵢ, pₙ) for xᵢ in slob.x]
    φ = A \ B
    return φ
end

function initial_conditions_numerical(slob::SLOB, pₙ)
    ϵ = rand(Normal(0.0, 1.0))
    V₀ =sign(ϵ) * min(abs(slob.σ * ϵ), slob.Δx / slob.Δt)
    return initial_conditions_numerical(slob, pₙ, V₀)
end

function extract_mid_price(slob, lob_density)
    mid_price_ind = 2
    while (lob_density[mid_price_ind] > 0) | (lob_density[mid_price_ind+1]>lob_density[mid_price_ind])
        mid_price_ind += 1
    end
    y1 = lob_density[mid_price_ind-1]
    y2 = lob_density[mid_price_ind]
    x1 = slob.x[mid_price_ind-1]

    mid_price = round(-(y1 * slob.Δx)/(y2 - y1) + x1, digits = 2)
    return mid_price
end


function calculate_left_jump_probability(Z)
    return (1.0)/(exp(2*Z) + 1.0)
end


function calculate_right_jump_probability(Z)
    return (1.0)/(exp(-2*Z) + 1.0)
end


function calculate_jump_probabilities(slob, Vₜ)
    Z = (Vₜ * slob.Δx) / (2* slob.D)
    p⁻ = calculate_left_jump_probability(Z)
    p⁺ = calculate_right_jump_probability(Z)
    return p⁺, p⁻
end


function get_sub_period_time(slob, t, time_steps)
    τ = rand(Exponential(slob.α))
    remaining_time = time_steps - t + 1
    τ_periods = min(floor(Int, τ/slob.Δt), remaining_time)
    # @info "Waiting time=$(round(τ, digits=4)) which equates to $τ_periods time periods"
    return τ, τ_periods
end


function intra_time_period_simulate(slob, φ, p)
    ϵ = rand(Normal(0.0, 1.0))
    Vₜ = sign(ϵ) * min(abs(slob.σ * ϵ), slob.Δx / slob.Δt)

    P⁺, P⁻ = calculate_jump_probabilities(slob, Vₜ)

    φ₋₁ = φ[1]
    φₘ₊₁ = φ[end]
    φ_next = zeros(Float64, size(φ,1))

    φ_next[1] = P⁺ * φ₋₁ + P⁻ * φ[2] -slob.nu * slob.Δt * φ[1] + slob.Δt * slob.source_term(slob.x[1], p)
    φ_next[end] = P⁻ * φₘ₊₁ + P⁺ * φ[end-1] -slob.nu * slob.Δt * φ[end] + slob.Δt * slob.source_term(slob.x[end], p)
    φ_next[2:end-1] = P⁺ * φ[1:end-2] + P⁻ * φ[3:end] -slob.nu * slob.Δt * φ[2:end-1] +
        [slob.Δt * slob.source_term(xᵢ, p) for xᵢ in slob.x[2:end-1]]

    return φ_next
end

function dtrw_solver(slob::SLOB)
    time_steps = get_time_steps(slob.T, slob.Δt)
    p = ones(Float64, time_steps + 1)
    p[1] = slob.p₀

    t = 1
    φ = initial_conditions_numerical(slob, p[t], 0.0)

    while t <= time_steps
        τ, τ_periods = get_sub_period_time(slob, t, time_steps)

        for τₖ = 1:τ_periods
            t += 1
            φ = intra_time_period_simulate(slob, φ, p[t-1])
            p[t] = extract_mid_price(slob, φ)

        end
        if t > time_steps
            mid_prices = sample_mid_price_path(slob, p)
            return mid_prices
        end
        t += 1
        φ = initial_conditions_numerical(slob, p[t-1])
        p[t] = extract_mid_price(slob, φ)
    end

    mid_prices = sample_mid_price_path(slob, p)
    return mid_prices
end
