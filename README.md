HydroModelMCP/
├── Project.toml           # 项目依赖管理文件
├── Manifest.toml          # 自动生成的依赖锁定文件
├── start.jl               # 【重要】服务器启动入口脚本 (Claude 调用这个)
├── src/
│   ├── HydroModelMCP.jl        # 主模块定义
│   ├── utils.jl           # 通用工具函数 (数据格式转换等)
│   ├── tools/             # 【核心】所有的 MCP 工具逻辑
│   │   ├── manager.jl     # 负责注册和收集所有工具
│   │   ├── simulation.jl  # 封装模型运行 (run_model)
│   │   ├── discovery.jl   # 封装模型查询 (list_models)
│   │   └── calibration.jl # 封装自动率定逻辑
│   ├── resources/         # 数据资源逻辑
│   │   ├── loader.jl      # 定义如何读取本地 CSV/NetCDF
│   │   └── registry.jl    # 注册具体的资源路径
│   └── prompts/           # 提示词逻辑 (你的 Agent Skills)
│       └── experts.jl     # 定义"水文专家"等预设 Prompt
└── data/                  # (可选) 存放示例数据或默认配置