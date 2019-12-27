mutable struct SLOB
    num_paths::Int64
    T::Int64
    p₀::Float64
    M::Int64
    β::Float64
    L::Float64
    D::Float64
    σ::Float64
    α::Float64
    source_term::SourceTerm
    x::Array{Float64, 1}
    Δx::Float64
    Δt::Float64
end
function SLOB(num_paths::Int64, T::Int64, p₀::Float64,
    M::Int64, β::Float64, L::Real, D::Float64, σ::Float64,
    α::Float64, source_term::SourceTerm)
    logger = FileLogger(Dict(Logging.Info => "info.log", Logging.Error => "error.log"), append=false)
    global oldglobal = global_logger(logger)
    x₀ = p₀ - 0.5*L
    xₘ = p₀ + 0.5*L
    @assert x₀ >= 0
    # if D < 1.5 * σ
    #     @warn "D=$D is less than 1.5 times σ=$σ which runs the risk of unstable LOB ghost points"
    # end
    # Δx = L/M
    # if (β*Δx/D < log(1/3)/2) | (β*Δx/D > log(3)/2)
    #     @warn "The choice of β, D, L, M might result in jump probabilities > 0.75, thus causing large shifts in the LOB density"
    # end

    x = collect(Float64, range(x₀, xₘ, length=M+1))
    Δx = L/M
    Δt = (Δx^2) / (2.0*D)
    return SLOB(num_paths, T, p₀, M, β,
        L, D, σ, α, source_term, x, Δx, Δt)
end


SLOB(dict) = SLOB(
    dict["num_paths"], dict["T"], dict["p₀"], dict["M"],
    dict["β"], dict["L"], dict["D"],
    dict["σ"], dict["α"], SourceTerm(dict["λ"], dict["μ"]))


SLOB(;num_paths::Int64=1, T::Int64=100, p₀::Real=100.0,
    M::Int64=100, β::Real=1.0, L::Real=50.0, D::Real=4.0, σ::Real=1.0,
    α::Real=1.0, λ::Real=1.0, μ::Real=0.5) =
    SLOB(num_paths, T, p₀, M, β, L, D, σ, α,
    SourceTerm(λ, μ))


function (slob::SLOB)(seed::Int=-1)
    if seed == -1
            seeds = Int.(rand(MersenneTwister(), UInt32, slob.num_paths))
        else
            seeds = Int.(rand(MersenneTwister(seed), UInt32, slob.num_paths))
    end
    time_steps = get_time_steps(slob.T, slob.Δt)

    raw_price_paths = ones(Float64, time_steps + 1, slob.num_paths)
    raw_price_paths[1, :] .= slob.p₀

    sample_price_paths = ones(Float64, slob.T + 1, slob.num_paths)
    sample_price_paths[1, :] .= slob.p₀

    lob_densities = zeros(Float64, slob.M+1, time_steps + 1, slob.num_paths)
    P⁺s = ones(Float64, time_steps, slob.num_paths)
    P⁻s = ones(Float64, time_steps, slob.num_paths)

    for path in 1:slob.num_paths
        Random.seed!(seeds[path])
        @info "path $path with seed $(seeds[path])"
        lob_densities[:, :, path], raw_price_paths[:, path],
            sample_price_paths[:, path], P⁺s[:, path],
            P⁻s[:, path]  = dtrw_solver(slob)
    end

    return lob_densities, raw_price_paths, sample_price_paths, P⁺s, P⁻s
end

function get_time_steps(T::Int, Δt::Float64)
    time_steps =  T / Δt
    return floor(Int, time_steps + eps(time_steps))
end


function sample_mid_price_path(slob::SLOB, price_path)
    Δt = slob.Δt
    get_mid_price_inds(x) = get_time_steps(x, Δt)
    mid_price_inds = get_mid_price_inds.(0:slob.T) .+ 1
    mid_prices = price_path[mid_price_inds]
    return mid_prices
end
