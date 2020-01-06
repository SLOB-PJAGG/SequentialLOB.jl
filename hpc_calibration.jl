using Distributed

@everywhere using BlackBoxOptim

@everywhere using SequentialLOB, SMM, DelimitedFiles, JLD

@everywhere observed_prices = readdlm(
    "Original_Price_Bars_2300.csv", ',',
    Float64, '\n')[:,1]
@everywhere observed_prices = round.(observed_prices, digits=4)
@everywhere observed_log_prices = log.(observed_prices)
# wm_seed = 7311421241
# weight_matrix_estimator = WeightMatrix(wm_seed, observed_log_prices, 50, 30000)
# weight_matrix = weight_matrix_estimator(block_bootstrap_estimator)
# save("weight_matrix.jld", "weight_matrix", weight_matrix)
@everywhere weight_matrix = load("weight_matrix.jld")["weight_matrix"]
@everywhere method_of_moments = MoM(observed_log_prices, weight_matrix)

@everywhere function smm_slob(parameters)
    try
        slob = SLOB(parameters...)
        price_paths = slob()
        log_price_paths = log.(price_paths)
        return method_of_moments(log_price_paths)
    catch e
        return Inf
    end
end

@everywhere function slob_optim(x)
    num_paths = 50; M = 400 ; T = 2299 ; p₀ = 238.75 ; L = 200 ; λ = 1.0
    D, σ, nu, α, μ = x

    obj_value = smm_slob([num_paths,
        T, p₀, M, L, D, σ, nu, α, SourceTerm(λ, μ)])
    return obj_value
end

lb = [0.125, 0.01, 0.01, 5.0, 0.1]
ub = [3.0, 2.0, 0.95, 1000.0, 1.0]
sr = [(lb[i],ub[i]) for i=1:5]
res = bboptimize(slob_optim; Method=:dxnes, SearchRange = sr,
    NumDimensions = 5, MaxFuncEvals = 200, Workers=workers())
