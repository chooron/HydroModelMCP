"""
ExpHydro Model Demo - MCP Client
演示如何通过 MCP 服务器调用 exphydro 模型进行模拟和参数率定
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_exphydro_demo():
    """运行 exphydro 模型的完整演示流程"""

    # 配置服务器参数
    server_params = StdioServerParameters(
        command="julia",
        args=["--project=.", "start.jl"],
        env=None
    )

    print("=" * 60)
    print("ExpHydro Model Demo - MCP Client")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 步骤 1: 获取模型信息
            print("\n[步骤 1] 获取 exphydro 模型信息...")
            model_info = await session.call_tool(
                "get_model_info",
                arguments={"model_name": "exphydro"}
            )
            print(f"模型信息: {json.dumps(json.loads(model_info.content[0].text), indent=2, ensure_ascii=False)}")

            # 步骤 2: 获取参数边界
            print("\n[步骤 2] 获取模型参数边界...")
            params_info = await session.call_tool(
                "get_model_parameters",
                arguments={"model_name": "exphydro"}
            )
            print(f"参数信息: {json.dumps(json.loads(params_info.content[0].text), indent=2, ensure_ascii=False)}")

            # 步骤 3: 运行初始模拟（使用默认参数）
            print("\n[步骤 3] 运行初始模拟（使用默认参数）...")
            sim_result = await session.call_tool(
                "run_simulation",
                arguments={
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
            )
            sim_data = json.loads(sim_result.content[0].text)
            print(f"模拟完成! 输出变量: {list(sim_data['outputs'].keys())}")
            print(f"时间步数: {len(sim_data['outputs']['Q'])}")

            # 步骤 4: 数据分割（用于率定和验证）
            print("\n[步骤 4] 分割数据集...")
            split_result = await session.call_tool(
                "split_data",
                arguments={
                    "data_source": "csv",
                    "data_path": "data/03604000.csv",
                    "strategy": "split_sample",
                    "split_ratio": 0.7
                }
            )
            split_data = json.loads(split_result.content[0].text)
            print(f"训练集: {split_data['train_start']} 到 {split_data['train_end']}")
            print(f"验证集: {split_data['test_start']} 到 {split_data['test_end']}")

            # 步骤 5: 初始化率定配置
            print("\n[步骤 5] 初始化率定配置...")
            calib_config = await session.call_tool(
                "init_calibration_setup",
                arguments={
                    "model_name": "exphydro",
                    "data_source": "csv",
                    "data_path": "data/03604000.csv",
                    "observed_var": "flow(mm)",
                    "simulated_var": "Q"
                }
            )
            config_data = json.loads(calib_config.content[0].text)
            print(f"推荐算法: {config_data['recommended_algorithm']}")
            print(f"推荐目标函数: {config_data['recommended_objectives']}")

            # 步骤 6: 运行参数率定
            print("\n[步骤 6] 运行参数率定（这可能需要一些时间）...")
            calibration_result = await session.call_tool(
                "calibrate_model",
                arguments={
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
                    "max_iterations": 100,
                    "population_size": 50,
                    "solver": "ODE",
                    "interpolation": "LINEAR",
                    "split_strategy": "split_sample",
                    "split_ratio": 0.7
                }
            )
            calib_data = json.loads(calibration_result.content[0].text)
            print(f"\n率定完成!")
            print(f"最优参数: {json.dumps(calib_data['best_parameters'], indent=2)}")
            print(f"训练集性能: {calib_data['train_metrics']}")
            print(f"验证集性能: {calib_data['test_metrics']}")

            # 步骤 7: 使用率定后的参数运行模拟
            print("\n[步骤 7] 使用率定后的参数运行最终模拟...")
            final_sim = await session.call_tool(
                "run_simulation",
                arguments={
                    "model_name": "exphydro",
                    "data_source": "csv",
                    "data_path": "data/03604000.csv",
                    "input_mapping": {
                        "prcp": "prcp(mm/day)",
                        "tmean": "tmean(C)",
                        "pet": "pet(mm)"
                    },
                    "parameters": calib_data['best_parameters'],
                    "solver": "ODE",
                    "interpolation": "LINEAR"
                }
            )
            final_data = json.loads(final_sim.content[0].text)

            # 步骤 8: 计算最终性能指标
            print("\n[步骤 8] 计算最终性能指标...")
            metrics_result = await session.call_tool(
                "compute_metrics",
                arguments={
                    "observed": final_data['outputs']['Q'],
                    "simulated": final_data['outputs']['Q'],
                    "metrics": ["NSE", "KGE", "RMSE", "PBIAS", "R2"]
                }
            )
            metrics_data = json.loads(metrics_result.content[0].text)
            print(f"\n最终性能指标:")
            for metric, value in metrics_data.items():
                print(f"  {metric}: {value:.4f}")

            # 保存结果
            print("\n[步骤 9] 保存结果...")
            output_file = "examples/exphydro_demo/calibration_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": "exphydro",
                    "data_file": "data/03604000.csv",
                    "best_parameters": calib_data['best_parameters'],
                    "train_metrics": calib_data['train_metrics'],
                    "test_metrics": calib_data['test_metrics'],
                    "final_metrics": metrics_data
                }, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {output_file}")

            print("\n" + "=" * 60)
            print("演示完成!")
            print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_exphydro_demo())
