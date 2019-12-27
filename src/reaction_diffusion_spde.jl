function initial_conditions_steady_state(slob::SLOB, p₀)
    φ = [(xᵢ - p₀) * slob.source_term.λ /
         (2 * slob.D * slob.source_term.μ) *
         exp((-slob.source_term.μ * slob.L^2) / 4) +
         (sqrt(pi) * slob.source_term.λ *
          erf(sqrt(slob.source_term.μ) * (p₀ - xᵢ))) /
         (4 * slob.D * slob.source_term.μ^(3 / 2)) for xᵢ in slob.x]
    return φ
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
    return 1/(exp(Z) + 1)
end


function calculate_jump_probabilities(slob, Vₜ)
    Z = (slob.β * Vₜ * slob.Δx) / (slob.D)
    P⁻ = calculate_left_jump_probability(Z)
    P⁺ = 1 - P⁻
    return P⁺, P⁻
end


function get_sub_period_time(slob, t, time_steps)
    τ = rand(Exponential(slob.α))
    remaining_time = time_steps - t + 1
    τ_periods = min(floor(Int, τ/slob.Δt), remaining_time)
    @info "Waiting time=$(round(τ, digits=4)) which equates to $τ_periods time periods"
    return τ, τ_periods
end


function intra_time_period_simulate(slob, φ, p)
    ϵ = rand(Normal(0.0, 1.0))
    Vₜ = sign(ϵ) * min(abs(slob.σ * ϵ), slob.Δx / slob.Δt)

    P⁺, P⁻ = calculate_jump_probabilities(slob, Vₜ)

    φ₋₁ = φ[1]
    φₘ₊₁ = φ[end]
    φ_next = zeros(Float64, size(φ,1))

    φ_next[1] = P⁺ * φ₋₁ + P⁻ * φ[2] +  slob.source_term(slob.x[1], p)
    φ_next[end] = P⁻ * φₘ₊₁  + P⁺ * φ[end-1] + slob.source_term(slob.x[end], p)
    φ_next[2:end-1] = P⁺ * φ[1:end-2] + P⁻ * φ[3:end] +
        [slob.source_term(xᵢ, p) for xᵢ in slob.x[2:end-1]]
    return φ_next, P⁺, P⁻
end


function dtrw_solver(slob::SLOB)
    time_steps = get_time_steps(slob.T, slob.Δt)
    φ = ones(Float64, slob.M + 1, time_steps + 1)

    p = ones(Float64, time_steps + 1)
    mid_prices = ones(Float64, slob.T + 1)

    p[1] = slob.p₀
    mid_prices[1] = slob.p₀

    P⁺s = fill(1/2, time_steps)
    P⁻s = fill(1/2, time_steps)

    t = 1
    φ[:, t] = initial_conditions_numerical(slob, p[t], 0.0)

    while t <= time_steps
        τ, τ_periods = get_sub_period_time(slob, t, time_steps)

        for τₖ = 1:τ_periods
            t += 1
            φ[:, t], P⁺s[t-1], P⁻s[t-1]  = intra_time_period_simulate(slob,
                φ[:, t-1], p[t-1])
            try
                p[t] = extract_mid_price(slob, φ[:, t])
            catch e
                println("Bounds Error at t=$t")
                return φ, p, mid_prices, P⁺s, P⁻s
            end

            @info "Intra-period simulation. tick price = R$(p[t]) @t=$t"
        end
        if t > time_steps
            mid_prices = sample_mid_price_path(slob, p)
            return φ, p, mid_prices, P⁺s, P⁻s
        end
        t += 1
        φ[:, t] = initial_conditions_numerical(slob, p[t-1])
        p[t] = extract_mid_price(slob, φ[:, t])
        @info "LOB Density recalculated. tick price = R$(p[t]) @t=$t"
    end

    mid_prices = sample_mid_price_path(slob, p)
    return φ, p, mid_prices, P⁺s, P⁻s
end
