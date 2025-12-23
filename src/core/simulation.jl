module Simulation

using ..DataLoader
using ..HydroModels
using ..HydroModelLibrary
using ..ComponentArrays
using ..DataInterpolations
using ..Statistics

# ==============================================================================
# 1. 求解器与插值器配置 (Multiple Dispatch)
# ==============================================================================

# 使用 Val 进行派发，零运行时开销，且易于扩展
_resolve_solver(::Val{:ODE}) = HydroModels.ODESolver
_resolve_solver(::Val{:DISCRETE}) = HydroModels.DiscreteSolver
_resolve_solver(::Val{:MUTABLE}) = HydroModels.MutableSolver
_resolve_solver(::Val{:IMMUTABLE}) = HydroModels.ImmutableSolver
_resolve_solver(::Val{T}) where T = throw(ArgumentError("未实现的求解器类型: $T"))

_resolve_interpolator(::Val{:LINEAR}) = Val(LinearInterpolation)
_resolve_interpolator(::Val{:CONSTANT}) = Val(ConstantInterpolation)
_resolve_interpolator(::Val{:DIRECT}) = Val(HydroModels.DirectInterpolation)
_resolve_interpolator(::Val{T}) where T = throw(ArgumentError("未实现的插值器类型: $T"))

# ==============================================================================
# 2. 对外接口
# ==============================================================================

# ==============================================================================
# 3. 核心计算逻辑
# ==============================================================================

function _execute_core(
    model::M, forcing_nt::NamedTuple,
    params_vec::ComponentVector, init_states::ComponentVector,
    solver_sym::Symbol, interp_sym::Symbol, config_dict::AbstractDict
) where M
    # --- D. 数据对齐 ---
    input_names = HydroModels.get_input_names(model)
    # 检查缺失变量
    missing_vars = setdiff(input_names, keys(forcing_nt))
    if !isempty(missing_vars)
        throw(ArgumentError("输入数据缺失变量: $missing_vars"))
    end
    # 构建矩阵 (Vars x Time)
    input_matrix = stack([Float64.(forcing_nt[n]) for n in input_names], dims=1)

    # --- E. 模拟执行 ---
    hydro_config = HydroModels.HydroConfig(
        solver=_resolve_solver(Val(solver_sym)),
        interpolator=_resolve_interpolator(Val(interp_sym))
    )

    result_matrix = model(
        input_matrix,
        params_vec;
        initstates=init_states,
        config=hydro_config
    )

    # --- F. 结果提取策略 ---
    # 如果没指定，默认返回最后一列 (通常是 Q/Runoff)
    output_names = HydroModels.get_output_names(model)
    target_var = get(config_dict, "output_variable", nothing)

    out_vector = if !isnothing(target_var)
        idx = findfirst(==(Symbol(target_var)), output_names)
        if isnothing(idx)
            throw(ArgumentError("输出变量 '$target_var' 不存在于模型输出中"))
        end
        result_matrix[idx, :]
    else
        result_matrix[end, :]
    end

    return out_vector
end

"""
    run_simulation(args::AbstractDict)

MCP Tool 入口。
"""
function run_simulation(args::AbstractDict)
    # 1. 提取必需参数
    model_name = get(args, "model_name", nothing)
    # --- A. 模型加载 ---
    # 假设 load_model 返回包含 .model, .model_parameters 等信息的模块
    model_module = HydroModelLibrary.load_model(Symbol(model_name), reload=false)

    # 使用 invokelatest 调用它
    model = Base.invokelatest(m -> m.model, model_module)

    forcing_config = get(args, "forcing", nothing)

    if isnothing(model_name) || isnothing(forcing_config)
        throw(ArgumentError("必须提供 model_name 和 forcing"))
    end

    # 2. 提取可选配置 (用于后续派发)
    # 默认为 ODE 和 LINEAR
    solver_sym = Symbol(uppercase(get(args, "solver", "DISCRETE")))
    interp_sym = Symbol(uppercase(get(args, "interpolator", "LINEAR")))

    config_dict = get(args, "config", Dict()) # 用于 output 变量选择等

    # --- B. 参数处理 (可选参数逻辑) ---
    params_vec = if haskey(args, "params")
        (; params=NamedTuple(Symbol(k) => v for (k, v) in args["params"])[
            HydroModels.get_param_names(model)
        ]) |> ComponentVector
    else
        HydroModelLibrary.get_random_params(model_name)
    end

    # --- C. 状态初始化 (可选状态逻辑) ---
    state_names = HydroModels.get_state_names(model)
    init_states = if haskey(args, "init_states") && !isempty(state_names)
        # 如果用户提供了状态字典
        user_states = args["init_states"]
        # 匹配状态名，缺失的补0
        vals = [get(user_states, String(s), 0.0) for s in state_names]
        ComponentVector(NamedTuple{Tuple(state_names)}(vals))
    else
        # 默认全0初始化
        ComponentVector(NamedTuple{Tuple(state_names)}(zeros(length(state_names))))
    end

    # 3. 闭包封装核心逻辑
    # 我们把 params, model_name 等封装起来，传给 DataLoader.process_io
    # 这样 DataLoader 负责 I/O，这个匿名函数负责纯计算
    core_task = (data) -> _execute_core(
        model,
        data,
        params_vec,
        init_states,
        solver_sym,
        interp_sym,
        config_dict
    )

    # 4. 委托给 DataLoader 处理输入输出的多态性
    return DataLoader.process_io(core_task, forcing_config)
end

using ModelContextProtocol
using JSON3
using ..Simulation # 引用之前写好的业务模块

# 定义 run_simulation 工具
run_simulation_tool = MCPTool(
    name = "run_simulation",
    description = "执行水文模型模拟。支持动态加载模型、多种数据源(CSV/JSON/Redis)、自动参数补全及结果持久化。",
    
    # 使用 Complex Input Schema 定义复杂参数结构
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            # 1. 模型名称
            "model_name" => Dict{String,Any}(
                "type" => "string",
                "description" => "水文模型的唯一标识符 (例如 'hbv', 'xinanjiang', 'gr4j')。"
            ),
            
            # 2. 驱动数据配置 (嵌套对象)
            "forcing" => Dict{String,Any}(
                "type" => "object",
                "description" => "驱动数据配置。需指定 source_type 及对应的 path/key/data。",
                "properties" => Dict{String,Any}(
                    "source_type" => Dict{String,Any}(
                        "type" => "string",
                        "enum" => ["csv", "json", "redis"],
                        "description" => "数据源类型"
                    ),
                    "path" => Dict{String,Any}(
                        "type" => "string",
                        "description" => "文件路径 (仅适用于 csv/json)"
                    ),
                    "key" => Dict{String,Any}(
                        "type" => "string",
                        "description" => "Redis 键名 (仅适用于 redis)"
                    ),
                    "data" => Dict{String,Any}(
                        "type" => "object",
                        "description" => "直接传入的数据对象 (仅适用于 json)"
                    ),
                    "host" => Dict{String,Any}("type" => "string", "default" => "127.0.0.1"),
                    "port" => Dict{String,Any}("type" => "integer", "default" => 6379)
                ),
                "required" => ["source_type"]
            ),
            
            # 3. 模型参数 (数组)
            "params" => Dict{String,Any}(
                "type" => "array",
                "items" => Dict{String,Any}("type" => "number"),
                "description" => "模型参数数组。如果省略，将自动生成随机参数。"
            ),
            
            # 4. 求解器 (枚举)
            "solver" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["ODE", "DISCRETE", "MANUAL"],
                "default" => "ODE",
                "description" => "微分方程求解器类型。"
            ),
            
            # 5. 插值器 (枚举)
            "interpolator" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["LINEAR", "CONSTANT"],
                "default" => "LINEAR",
                "description" => "输入数据的插值方式。"
            ),
            
            # 6. 其他配置
            "config" => Dict{String,Any}(
                "type" => "object",
                "description" => "高级模拟配置 (例如指定输出变量)。",
                "properties" => Dict{String,Any}(
                    "output_variable" => Dict{String,Any}("type" => "string")
                )
            ),
            
            # 7. 初始状态
            "init_states" => Dict{String,Any}(
                "type" => "object",
                "description" => "自定义初始状态 (状态名 -> 值)。"
            )
        ),
        "required" => ["model_name", "forcing"]
    ),

    # Handler 函数
    handler = function(params)
        try
            # 调用之前的业务逻辑核心
            # Simulation.run_simulation_handler 返回的是一个 Dict
            result = Simulation.run_simulation_handler(params)
            
            # 直接返回 Dict，MCP 库会自动将其转换为 JSON 格式的 TextContent
            return result
            
        catch e
            # 捕获错误并按照文档规范返回 Error Result
            # 这样 Agent 能知道出错了，而不是让 Server 崩溃
            err_msg = sprint(showerror, e)
            stack_trace = sprint(Base.show_backtrace, catch_backtrace())
            
            return CallToolResult(
                content = [
                    Dict("type" => "text", "text" => "Simulation Failed: $err_msg\n\nDetails:\n$stack_trace")
                ],
                is_error = true
            )
        end
    end
)

end