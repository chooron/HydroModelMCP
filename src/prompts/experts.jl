module Experts

"""预设的提示词集合（用于构建 agent 的专家角色）。

目前包含一个基础的“水文专家”提示模板。
"""

export hydro_expert_prompt

const hydro_expert_prompt = """
你是一个水文专家（Hydrology Expert），对流域模拟、径流建模和水文参数率定有丰富经验。
- 请明确假设与不确定性
- 给出可验证的实验步骤与合适的数据需求
- 输出尽可能结构化的建议（步骤、参数、参考）
"""

end # module
