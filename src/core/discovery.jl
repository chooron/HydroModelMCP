module Discovery

using ..HydroModelLibrary
using ..HydroModels

export list_models, find_model, get_model_info

"""
    list_models() -> Vector{String}

列出 HydroModelLibrary 中所有可用的水文模型名称。
确保返回 String 类型数组，便于 JSON 序列化。
"""
function list_models()
    # 假设 AVAILABLE_MODELS 是 Symbol 或 String 集合
    # 强制转换为 String 数组
    return String.(HydroModelLibrary.AVAILABLE_MODELS)
end

"""
    find_model(name::String) -> Union{String, Nothing}

查找模型名称是否存在（忽略大小写）。
如果存在，返回库中定义的原始名称（保留大小写）；如果不存在，返回 nothing。

Example:
    find_model("hbv") -> "HBV" (假设库中是大写)
"""
function find_model(name::String)
    if isempty(name)
        return nothing
    end
    
    target = lowercase(strip(name))
    all_models = list_models()
    
    # 遍历查找，确保大小写不敏感匹配
    # findfirst 返回索引，我们通过索引取回原始名称
    idx = findfirst(m -> lowercase(m) == target, all_models)
    
    return isnothing(idx) ? nothing : all_models[idx]
end

"""
    get_model_info(name::String)

加载指定模型，并返回其详细元数据（输入、参数、状态、输出以及可视化描述）。
"""
function get_model_info(name::String)
    # 1. 验证模型是否存在
    valid_name = find_model(name)
    if isnothing(valid_name)
        throw(ArgumentError("模型 '$name' 未找到。请先调用 list_models 查看可用列表。"))
    end

    # 2. 动态加载模型
    # 注意：这里复用了 HydroModelLibrary 的加载逻辑
    # reload=false 以提高性能
    model_module = HydroModelLibrary.load_model(Symbol(valid_name), reload=false)
    model = Base.invokelatest(m -> m.model, model_module)

    # 3. 提取结构化信息 (供 Agent 逻辑使用)
    # 将 Symbol 转换为 String，方便 JSON 序列化
    inputs = string.(HydroModels.get_input_names(model))
    states = string.(HydroModels.get_state_names(model))
    outputs = string.(HydroModels.get_output_names(model))
    params = string.(HydroModels.get_param_names(model))

    # 4. 获取人类可读的描述 (即 print(model) 的内容)
    # 使用 sprint 配合 text/plain MIME 类型来捕获美化后的输出
    description_str = sprint(show, MIME("text/plain"), model)

    # 5. 组装返回字典
    return Dict(
        "model_name" => valid_name,
        "inputs" => inputs,      # e.g. ["P", "T", "Ep"]
        "params" => params,      # e.g. ["TT", "CFMAX", ...]
        "states" => states,      # e.g. ["SP", "WC"]
        "outputs" => outputs,    # e.g. ["q", "Qt"]
        "description" => description_str # 完整的 ASCII 图表描述
    )
end


"""
    get_variables_detail(model_name::String)

获取模型中所有变量（Inputs, States, Fluxes）的详细元数据（描述、单位）。
"""
function get_variables_detail(model_name::String)
    valid_name = find_model(model_name)
    if isnothing(valid_name)
        throw(ArgumentError("模型 '$model_name' 未找到。"))
    end

    # 1. 加载模块 wrapper
    wrapper = HydroModelLibrary.load_model(Symbol(valid_name), reload=false)
    
    # 2. 安全读取 model_variables 列表 (利用 invokelatest 避免 world age 问题)
    # 对应你提到的: model_variables = [P, Ep, Sus, ...]
    vars_list = Base.invokelatest(m -> m.model_variables, wrapper)

    info_list = []
    
    for v in vars_list
        # 获取名称 (通常 Symbolics 变量转 string 即为名字)
        v_name = string(v)
        
        # 获取元数据 (假设 HydroModels 导出了这些 getter)
        # 如果是 Symbolics.jl 的变量，这些函数通常来自 metadata
        desc = try HydroModels.getdescription(v) catch; "" end
        unit = try string(HydroModels.getunit(v)) catch; "-" end
        
        push!(info_list, Dict(
            "name" => v_name,
            "description" => desc,
            "unit" => unit,
            "type" => "variable" # 标记类型
        ))
    end
    
    return info_list
end

"""
    get_parameters_detail(model_name::String)

获取模型参数的详细元数据，包括描述、单位和**取值范围(Bounds)**。
这对 Agent 进行参数率定至关重要。
"""
function get_parameters_detail(model_name::String)
    valid_name = find_model(model_name)
    if isnothing(valid_name)
        throw(ArgumentError("模型 '$model_name' 未找到。"))
    end

    wrapper = HydroModelLibrary.load_model(Symbol(valid_name), reload=false)
    
    # 安全读取 model_parameters 列表
    params_list = Base.invokelatest(m -> m.model_parameters, wrapper)

    info_list = []

    for p in params_list
        p_name = string(p)
        
        desc = try HydroModels.getdescription(p) catch; "" end
        unit = try string(HydroModels.getunit(p)) catch; "-" end
        
        # 获取边界 (Bounds)
        # 通常返回 (min, max) 元组，我们需要将其转换为 JSON 友好的数组
        bounds = try 
            b = HydroModels.getbounds(p)
            [b[1], b[2]] # 转为 Vector
        catch
            nothing
        end

        push!(info_list, Dict(
            "name" => p_name,
            "description" => desc,
            "unit" => unit,
            "bounds" => bounds, # JSON: [min, max] or null
            "type" => "parameter"
        ))
    end

    return info_list
end

export list_models, find_model, get_model_info, get_parameters_detail, get_variables_detail

end # module