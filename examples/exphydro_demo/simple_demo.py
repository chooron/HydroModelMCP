"""
ExpHydro 简化演示 - 快速开始
只包含核心的模拟和率定功能
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def simple_demo():
    """简化的 exphydro 演示"""

    server_params = StdioServerParameters(
        command="julia",
        args=["--project=.", "start.jl"],
        env=None
    )

    print("ExpHydro 简化演示")
    print("-" * 40)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. 快速模拟
            print("\n1. 运行模拟...")
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
                    }
                }
            )
            sim_data = json.loads(sim_result.content[0].text)
            print(f"   模拟完成! 时间步数: {len(sim_data['outputs']['Q'])}")

            # 2. 参数率定
            print("\n2. 参数率定...")
            calib_result = await session.call_tool(
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
                    "max_iterations": 50,
                    "population_size": 30
                }
            )
            calib_data = json.loads(calib_result.content[0].text)

            print(f"\n率定结果:")
            print(f"   最优参数: {calib_data['best_parameters']}")
            print(f"   训练集 KGE: {calib_data['train_metrics']['KGE']:.4f}")
            print(f"   验证集 KGE: {calib_data['test_metrics']['KGE']:.4f}")

            print("\n完成!")

if __name__ == "__main__":
    asyncio.run(simple_demo())
