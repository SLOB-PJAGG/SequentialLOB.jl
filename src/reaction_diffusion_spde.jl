function initial_conditions_numerical(slob::SLOB, pₙ, V₀)
    ud = (-V₀/(2.0*slob.Δx) + slob.D/(slob.Δx^2)) * ones(Float64, slob.M)
    md = ((-2.0*slob.D)/(slob.Δx^2) - slob.nu) * ones(Float64, slob.M+1)
    ld = (V₀/(2.0*slob.Δx) + slob.D/(slob.Δx^2)) * ones(Float64, slob.M)
    A = Tridiagonal(ud, md, ld)

    A[1,2] = 2*slob.D/(slob.Δx^2)
    A[end, end-1] = 2*slob.D/(slob.Δx^2)

    B = .-[slob.source_term(xᵢ, pₙ) for xᵢ in slob.x]
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


function calculate_right_jump_probability(Z)
    if Z > 10
        return 1.0
    elseif Z < -10
        return 0.0
    else
        return (exp(Z))/(exp(-Z) + exp(Z) + 1)
    end
end


function calculate_left_jump_probability(Z)
    if Z > 10
        return 0.0
    elseif Z < -10
        return 1.0
    else
        return (exp(-Z))/(exp(-Z) + exp(Z) + 1)
    end
end


function calculate_self_jump_probability(Z)
    if Z > 10
        return 0.0
    elseif Z < -10
        return 0.0
    else
        return 1/(exp(-Z) + exp(Z) + 1)
    end
end


function calculate_jump_probabilities(rdpp, Vₜ)
    Z = (rdpp.β * Vₜ * rdpp.Δx) / (2* rdpp.D)
    return calculate_right_jump_probability(Z),
        calculate_left_jump_probability(Z), calculate_self_jump_probability(Z)
end


function get_sub_period_time(rdpp, t, time_steps)
    τ = rand(Exponential(rdpp.α))
    remaining_time = time_steps - t + 1
    τ_periods = min(floor(Int, τ/rdpp.Δt), remaining_time)
    @info "Waiting time=$(round(τ, digits=4)) which equates to $τ_periods time periods"
    return τ, τ_periods
end


function get_adaptive_price_grid(rdpp, p)
    x₀ = p - 0.5*rdpp.L
    xₘ = p + 0.5*rdpp.L
    @assert x₀ >= 0
    @info "Price grid changed from $(rdpp.x[1]):$(rdpp.x[end]) to $x₀:$xₘ"
    return collect(Float64, range(x₀, xₘ, length=rdpp.M+1))
end


function intra_time_period_simulate(rdpp, φ, p)
    ϵ = rand(Normal(0.0, 1.0))
    Vₜ = sign(ϵ) * min(abs(rdpp.σ * ϵ), rdpp.Δx / rdpp.Δt)

    P⁺, P⁻, P = calculate_jump_probabilities(rdpp, Vₜ)

    φ₋₁ = φ[1]
    φₘ₊₁ = φ[end]
    φ_next = zeros(Float64, size(φ,1))

    φ_next[1] = P⁺ * φ₋₁ + P * φ[1] + P⁻ * φ[2] - slob.nu * φ[1] + slob.source_term(slob.x[1], p)
    φ_next[end] = P⁻ * φₘ₊₁ +  P * φ[end] + P⁺ * φ[end-1] - slob.nu * φ[end] + slob.source_term(slob.x[end], p)
    φ_next[2:end-1] = P⁺ * φ[1:end-2] + P * φ[2:end-1] + P⁻ * φ[3:end] - slob.nu * φ[2:end-1] +
        [slob.source_term(xᵢ, p) for xᵢ in slob.x[2:end-1]]

    return φ_next, P⁺, P⁻, P
end


function dtrw_solver(rdpp::ReactionDiffusionPricePaths)
    time_steps = floor(Int, rdpp.T / rdpp.Δt)

    φ = ones(Float64, rdpp.M + 1, time_steps + 1)

    p = ones(Float64, time_steps + 1)
    mid_prices = ones(Float64, rdpp.T + 1)

    p[1] = rdpp.p₀
    mid_prices[1] = rdpp.p₀

    P⁺s = ones(Float64, time_steps)
    P⁻s = ones(Float64, time_steps)
    Ps = ones(Float64, time_steps)

    t = 1
    calendar_time_index = 1
    φ[:, t] = initial_conditions_numerical(rdpp, p[t], 0.0)

    while t <= time_steps
        τ, τ_periods = get_sub_period_time(rdpp, t, time_steps)

        for τₖ = 1:τ_periods
            t += 1
            φ[:, t], P⁺s[t-1], P⁻s[t-1], Ps[t-1]  = intra_time_period_simulate(rdpp,
                φ[:, t-1], p[calendar_time_index])
            p[t] = extract_mid_price(rdpp, φ[:, t])
            @info "Intra-period simulation. tick price = R$(p[t]) @t=$t"
            if t*rdpp.Δt > calendar_time_index
                calendar_time_index += 1
                mid_prices[calendar_time_index] = p[t]
                @info "Mid price R$(mid_prices[calendar_time_index]) sampled @n=$calendar_time_index"
            end
        end
        if t > time_steps
            return φ, p, mid_prices, P⁺s, P⁻s, Ps
        end
        t += 1
        φ[:, t] = initial_conditions_numerical(rdpp, p[t-1])
        p[t] = extract_mid_price(rdpp, φ[:, t])
        @info "LOB Density recalculated. tick price = R$(p[t]) @t=$t"
        rdpp.x = get_adaptive_price_grid(rdpp, p[t])
        if (t)*rdpp.Δt > calendar_time_index
            calendar_time_index += 1
            mid_prices[calendar_time_index] = p[t]
            @info "Mid price R$(mid_prices[calendar_time_index]) sampled @n=$calendar_time_index"
        end
    end
    return φ, p, mid_prices, P⁺s, P⁻s, Ps
end