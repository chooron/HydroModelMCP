module Sampling

using ..Statistics
using Random

export generate_samples, pie_share_sampling, delta_method_sampling

# ==============================================================================
# 对外接口
# ==============================================================================

"""
    generate_samples(bounds; method, n_samples) -> Matrix{Float64}

为给定参数范围生成样本矩阵。

返回 Matrix{Float64}，维度 (n_params × n_samples)。

方法 (Strategy 5):
- "lhs": 拉丁超立方采样 — 保证各维度均匀覆盖
- "sobol": Sobol 准随机序列
- "random": 纯随机采样 (不推荐，仅作对比基线)
"""
function generate_samples(bounds::Vector{Tuple{Float64,Float64}};
                          method::String="lhs",
                          n_samples::Int=100)
    n_dims = length(bounds)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]

    if method == "lhs"
        return _latin_hypercube(n_dims, n_samples, lb, ub)
    elseif method == "sobol"
        return _sobol_sequence(n_dims, n_samples, lb, ub)
    elseif method == "random"
        return _random_uniform(n_dims, n_samples, lb, ub)
    else
        throw(ArgumentError("未知采样方法: $(method)。可选: lhs, sobol, random"))
    end
end

# ==============================================================================
# 内置 LHS 实现
# ==============================================================================

"""标准拉丁超立方采样"""
function _latin_hypercube(n_dims::Int, n_samples::Int, lb::Vector{Float64}, ub::Vector{Float64})
    samples = zeros(n_dims, n_samples)
    for i in 1:n_dims
        # 在每个区间内随机采样一个点
        perm = randperm(n_samples)
        for j in 1:n_samples
            low = (perm[j] - 1) / n_samples
            high = perm[j] / n_samples
            samples[i, j] = lb[i] + (ub[i] - lb[i]) * (low + rand() * (high - low))
        end
    end
    return samples
end

# ==============================================================================
# Sobol 序列 (简易实现，基于 Van der Corput 序列)
# ==============================================================================

"""基于 Van der Corput 序列的准随机采样"""
function _sobol_sequence(n_dims::Int, n_samples::Int, lb::Vector{Float64}, ub::Vector{Float64})
    # 使用不同素数基底的 Van der Corput 序列近似 Sobol
    primes = _first_primes(n_dims)
    samples = zeros(n_dims, n_samples)
    for i in 1:n_dims
        for j in 1:n_samples
            val = _van_der_corput(j, primes[i])
            samples[i, j] = lb[i] + (ub[i] - lb[i]) * val
        end
    end
    return samples
end

"""Van der Corput 序列第 n 项，基底 base"""
function _van_der_corput(n::Int, base::Int)
    result = 0.0
    f = 1.0 / base
    i = n
    while i > 0
        result += f * (i % base)
        i = div(i, base)
        f /= base
    end
    return result
end

"""返回前 n 个素数"""
function _first_primes(n::Int)
    primes = Int[]
    candidate = 2
    while length(primes) < n
        if all(candidate % p != 0 for p in primes)
            push!(primes, candidate)
        end
        candidate += 1
    end
    return primes
end

# ==============================================================================
# 纯随机采样
# ==============================================================================

function _random_uniform(n_dims::Int, n_samples::Int, lb::Vector{Float64}, ub::Vector{Float64})
    samples = zeros(n_dims, n_samples)
    for i in 1:n_dims
        samples[i, :] = lb[i] .+ (ub[i] - lb[i]) .* rand(n_samples)
    end
    return samples
end

# ==============================================================================
# Pie-share 约束采样 (Strategy 2A: 参数加和为常数)
# ==============================================================================

"""
    pie_share_sampling(n_params, n_samples; total=1.0) -> Matrix{Float64}

Pie-share 采样 (Appendix D, Eq. D.1)。
生成 n_params 个参数的样本，保证每个样本中所有参数之和等于 `total`。
基于 Dirichlet 分布实现。

返回 Matrix{Float64} (n_params × n_samples)，每列之和 = total。
"""
function pie_share_sampling(n_params::Int, n_samples::Int; total::Float64=1.0)
    samples = zeros(n_params, n_samples)
    for j in 1:n_samples
        # Dirichlet(1,1,...,1) 采样：对每个维度生成 Gamma(1,1) 随机数
        raw = -log.(rand(n_params))  # Exponential(1) = Gamma(1,1)
        samples[:, j] = total .* raw ./ sum(raw)
    end
    return samples
end

# ==============================================================================
# Delta Method 约束采样 (Strategy 2B: 不等式约束 x1 < x2)
# ==============================================================================

"""
    delta_method_sampling(bounds, constraints; n_samples=100) -> Matrix{Float64}

Delta Method 采样。处理 x_i < x_j 类型的不等式约束。

参数:
- bounds: Vector{Tuple{Float64,Float64}} 各参数的原始范围
- constraints: Vector{Tuple{Int,Int}} 约束列表，每个 (i,j) 表示 param_i < param_j
- n_samples: 采样数

实现方式：先用 LHS 在变换空间采样，再通过 delta 变换恢复原始空间。
对于约束 x_i < x_j：采样 x_i 和 delta_ij >= 0，令 x_j = x_i + delta_ij。
"""
function delta_method_sampling(bounds::Vector{Tuple{Float64,Float64}},
                               constraints::Vector{Tuple{Int,Int}};
                               n_samples::Int=100)
    n_params = length(bounds)
    # 先用 LHS 生成无约束样本
    samples = _latin_hypercube(n_params, n_samples,
                               [b[1] for b in bounds], [b[2] for b in bounds])

    # 对每个约束进行修正，迭代直到所有约束都满足
    max_iterations = 100
    for s in 1:n_samples
        for iter in 1:max_iterations
            all_satisfied = true
            for (i, j) in constraints
                if samples[i, s] >= samples[j, s]
                    all_satisfied = false
                    # 违反约束：重新分配，确保 samples[i, s] < samples[j, s]
                    lo_i = bounds[i][1]
                    hi_i = bounds[i][2]
                    lo_j = bounds[j][1]
                    hi_j = bounds[j][2]

                    # 在有效范围内重新采样
                    # 选择一个分割点，使得 i < split < j
                    effective_range = min(hi_i, hi_j) - max(lo_i, lo_j)
                    if effective_range > 0
                        split = max(lo_i, lo_j) + effective_range * rand()
                        samples[i, s] = lo_i + (split - lo_i) * rand()
                        samples[j, s] = split + (hi_j - split) * rand()
                    else
                        # 如果范围无效，使用简单策略
                        samples[i, s] = lo_i + (hi_i - lo_i) * rand() * 0.4
                        samples[j, s] = lo_j + (hi_j - lo_j) * rand() * 0.6 + (hi_j - lo_j) * 0.4
                    end
                end
            end
            if all_satisfied
                break
            end
        end
    end

    return samples
end

end # module
