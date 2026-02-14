# ExpHydro 模型演示

这个示例展示如何通过 MCP 服务器调用 exphydro 模型进行水文模拟和参数率定。

## 数据说明

使用数据文件: `data/03604000.csv`

数据列:
- `prcp(mm/day)` - 降水量
- `tmean(C)` - 平均温度
- `pet(mm)` - 潜在蒸散发
- `flow(mm)` - 观测流量

数据长度: 1461 天

## 文件说明

### 1. `demo_julia.jl` - Julia 直接调用演示（推荐）

直接调用 HydroModelMCP 核心功能，无需启动 MCP 服务器。包含完整的工作流程:
1. 获取模型信息和参数边界
2. 读取数据
3. 运行初始模拟（随机参数）
4. 数据分割（训练集/验证集）
5. 参数率定
6. 验证集测试
7. 全数据集评估
8. 保存结果

### 2. `demo_client.py` - Python MCP 客户端演示

通过 MCP 协议调用服务器，展示完整的客户端-服务器交互。

### 3. `simple_demo.py` - 快速开始

简化版 Python 客户端，只包含核心功能。

## 使用方法

### 方式 1: Julia 直接调用（最简单）

```bash
cd E:\JlCode\HydroModelMCP
julia examples/exphydro_demo/demo_julia.jl
```

这是最直接的方式，不需要启动 MCP 服务器，直接调用核心功能。

### 方式 2: 快速测试（命令行）

```bash
cd E:\JlCode\HydroModelMCP
julia -e '
using Pkg; Pkg.activate(".")
using HydroModelMCP, CSV, DataFrames

# 读取数据
df = CSV.read("data/03604000.csv", DataFrame)
forcing_nt = (P = df[!, "prcp(mm/day)"], T = df[!, "tmean(C)"], Ep = df[!, "pet(mm)"])
observed = df[!, "flow(mm)"]

# 分割数据
split_result = HydroModelMCP.DataSplitter.split_data(observed, forcing_nt; method="split_sample", ratio=0.7)

# 率定
calib_result = HydroModelMCP.Calibration.calibrate_model(
    "exphydro", split_result["cal_forcing"], split_result["cal_obs"];
    algorithm="BBO", maxiters=50, objective="KGE"
)

println("最优参数: ", calib_result["best_params"])
println("训练集 KGE: ", calib_result["best_objective"])
'
```

### 方式 3: Python MCP 客户端

前置要求:
```bash
pip install mcp
```

运行:
```bash
cd E:\JlCode\HydroModelMCP
python examples/exphydro_demo/demo_client.py
```

### 方式 4: MCP Inspector 手动测试

```bash
cd E:\JlCode\HydroModelMCP
npx @modelcontextprotocol/inspector julia --project=. start.jl
```

然后在浏览器中访问 http://localhost:5173

## 率定结果示例

运行演示后，结果会保存在 `calibration_results.json`:

```json
{
  "model_name": "exphydro",
  "best_params": {
    "f": 0.606,
    "Qmax": -2.564,
    "Df": 1330.121,
    "Tmin": 0.017,
    "Smax": 2.864,
    "Tmax": 21.449
  },
  "best_objective": 0.276,
  "objective_name": "KGE"
}
```

## 参数说明

### ExpHydro 模型参数

- `f` - 最大入渗率 (0-3)
- `Smax` - 最大土壤含水量 (0-5)
- `Qmax` - 最大基流系数 (-3-0)
- `Df` - 延迟因子 (100-2000)
- `Tmax` - 最大温度阈值 (10-50)
- `Tmin` - 最小温度阈值 (0-0.1)

### 率定算法选项

- **BBO** (Biogeography-Based Optimization) - 推荐，收敛快
- **PSO** (Particle Swarm Optimization) - 适合多峰问题
- **DE** (Differential Evolution) - 鲁棒性好
- **CMAES** - 适合高维问题
- **ECA** - 进化算法

### 性能指标

- **KGE** (Kling-Gupta Efficiency) - 综合指标，推荐
- **NSE** (Nash-Sutcliffe Efficiency) - 经典指标
- **RMSE** (Root Mean Square Error) - 误差指标
- **PBIAS** (Percent Bias) - 偏差百分比
- **R2** (Coefficient of Determination) - 相关系数

## 调整率定参数

修改 `maxiters` 和算法来调整率定过程:

```julia
calib_result = HydroModelMCP.Calibration.calibrate_model(
    "exphydro", train_forcing, train_obs;
    algorithm="BBO",      # 算法选择
    maxiters=100,         # 增加迭代次数提高精度
    objective="KGE",      # 目标函数
    solver_type="ODE",    # 求解器类型
    interp_type="LINEAR"  # 插值方法
)
```

## 故障排除

### 问题: Julia 版本警告
```
WARNING: Detected access to binding `HydroModelLibrary.exphydro` in a world prior to its definition world.
```
这是 Julia 1.12 的警告，不影响功能，可以忽略。

### 问题: 率定时间过长
减少 `maxiters` 参数:
```julia
maxiters=30  # 从 50 或 100 减少到 30
```

### 问题: 率定结果不理想
1. 增加迭代次数: `maxiters=100`
2. 尝试不同算法: `algorithm="PSO"` 或 `"DE"`
3. 检查数据质量和分割比例

## 扩展功能

### 1. 敏感性分析

```julia
sensitivity_result = HydroModelMCP.Sensitivity.run_sensitivity(
    "exphydro", forcing_nt, observed;
    method="Morris",
    n_samples=100
)
```

### 2. 多目标优化

```julia
multi_result = HydroModelMCP.Calibration.calibrate_multiobjective(
    "exphydro", train_forcing, train_obs;
    objectives=["KGE", "NSE"],
    algorithm="NSGA2"
)
```

### 3. 使用其他模型

查看可用模型:
```julia
models = HydroModelMCP.Discovery.list_models()
```

## 参考

- HydroModelMCP 文档: [README.md](../../README.md)
- MCP 协议: https://modelcontextprotocol.io
- ExpHydro 模型: Patil & Stieglitz (2015)
