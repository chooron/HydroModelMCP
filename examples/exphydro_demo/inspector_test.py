"""
ExpHydro 模型演示 - 使用 MCP Inspector 测试
这个脚本展示如何手动测试 MCP 工具
"""

# 测试步骤（在 MCP Inspector 中执行）

# 1. 启动 MCP Inspector
# npx @modelcontextprotocol/inspector julia --project=. start.jl

# 2. 获取模型信息
get_model_info_request = {
    "model_name": "exphydro"
}

# 3. 运行模拟
run_simulation_request = {
    "model_name": "exphydro",
    "data_source": "csv",
    "data_path": "data/03604000.csv",
    "input_mapping": {
        "prcp": "prcp(mm/day)",
        "tmean": "tmean(C)",
        "pet": "pet(mm)"
    },
    "solver": "ODE",
    "interpolation": "LINEAR"
}

# 4. 参数率定
calibrate_model_request = {
    "model_name": "exphydro",
    "data_source": "csv",
    "data_path": "data/03604000.csv",
    "input_mapping": {
        "prcp": "prcp(mm/day)",
        "tmean": "tmean(C)",
        "pet": "pet(mm)"
    },
    "observed_var": "flow(mm)",
    "simulated_var": "Q",
    "objective": "KGE",
    "algorithm": "BBO",
    "max_iterations": 50,
    "population_size": 30,
    "solver": "ODE",
    "interpolation": "LINEAR",
    "split_strategy": "split_sample",
    "split_ratio": 0.7
}

# 5. 计算性能指标
# 首先从模拟结果中获取 Q 值，然后：
compute_metrics_request = {
    "observed": [],  # 从 CSV 读取的 flow(mm) 列
    "simulated": [],  # 从模拟结果获取的 Q 值
    "metrics": ["NSE", "KGE", "RMSE", "PBIAS", "R2"]
}

print("请在 MCP Inspector 中依次执行上述请求")
print("访问: http://localhost:5173")
