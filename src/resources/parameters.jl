using ModelContextProtocol

const objective_guide_resource = MCPResource(
    uri = "hydro://guides/objectives",
    name = "Objective Function Selection Guide",
    description = "Mapping of calibration goals to recommended objective functions.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "mappings" => [
            Dict("goal" => "general_fit", "primary" => "KGE", "secondary" => ["NSE", "PBIAS"], "log_transform" => false),
            Dict("goal" => "peak_flows", "primary" => "NSE", "secondary" => ["KGE", "RMSE"], "log_transform" => false),
            Dict("goal" => "low_flows", "primary" => "LogKGE", "secondary" => ["LogNSE", "PBIAS"], "log_transform" => true),
            Dict("goal" => "water_balance", "primary" => "PBIAS", "secondary" => ["KGE", "R2"], "log_transform" => false),
            Dict("goal" => "dynamics", "primary" => "R2", "secondary" => ["KGE"], "log_transform" => false)
        ],
        "metrics" => Dict(
            "KGE" => "Kling-Gupta Efficiency balances correlation, variability, and bias.",
            "NSE" => "Nash-Sutcliffe Efficiency emphasizes peak-flow fit.",
            "LogNSE" => "Log-transformed NSE emphasizes low flows.",
            "LogKGE" => "Log-transformed KGE balances low-flow performance.",
            "PBIAS" => "Percent Bias emphasizes volume conservation.",
            "R2" => "Coefficient of determination emphasizes pattern similarity.",
            "RMSE" => "Root Mean Square Error emphasizes absolute error magnitude."
        )
    )
)

export objective_guide_resource
