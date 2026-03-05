module PISampler

using StaticArrays, Random, Distributions
export generate_dataset

function random_feature_path(dynamics, dims, tspan, sigma)
    H = 101

    W1 = rand(Normal(0.0f0, sigma), H) 
    b1 = rand(Normal(0.0f0, sigma), H)
    W2 = rand(Normal(0.0f0, 1.0f0), dims, H)
    b2 = rand(Normal(0.0f0, 1.0f0), dims)

    tlen = length(tspan)
    t = reshape(collect(Float32, tspan), 1, tlen)

    inner = (W1 * t) .+ b1 .+ (Float32(pi)/4)
    h = sqrt(2.0f0 / H) .* sin.(inner)
    X = (W2 * h) .+ b2
    Ẋ = W2 * (W1 * sqrt(2.0f0 / H) .* cos.(inner))

    f = zeros(Float32, dims, tlen)

    for i in 1:tlen
        Xt  = SVector{dims}(X[:, i])
        dXt = SVector{dims}(Ẋ[:, i])
        dXp = dynamics(Xt)
        f[:, i] = dXt .- dXp
    end

    return (f=f, X=X, t=t)
end

function generate_dataset(dynamics, dims, n; tspan=0:0.01:1, sigma=2.0f0)
    data = Vector{Any}(undef, n)
    
    Threads.@threads for i in 1:n
        data[i] = random_feature_path(dynamics, dims, tspan, sigma)
    end

    return data
end

end