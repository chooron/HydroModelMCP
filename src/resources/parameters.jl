using ModelContextProtocol
using JSON3
using URIs

# Resource: Objective function guide
objective_guide_resource = MCPResource(
    uri = "hydro://guides/objectives",
    name = "Objective Function Selection Guide",
    description = "Mapping of calibration goals to recommended objective functions",
    mime_type = "application/json",
    data_provider = () -> begin
        guide = Dict(
            "mappings" => [
                Dict("goal" => "general_fit", "primary" => "KGE",
                     "secondary" => ["NSE", "PBIAS"], "log_transform" => false),
                Dict("goal" => "peak_flows", "primary" => "NSE",
                     "secondary" => ["KGE", "RMSE"], "log_transform" => false),
                Dict("goal" => "low_flows", "primary" => "LogKGE",
                     "secondary" => ["LogNSE", "PBIAS"], "log_transform" => true),
                Dict("goal" => "water_balance", "primary" => "PBIAS",
                     "secondary" => ["KGE", "R2"], "log_transform" => false),
                Dict("goal" => "dynamics", "primary" => "R2",
                     "secondary" => ["KGE"], "log_transform" => false)
            ],
            "metrics" => Dict(
                "KGE" => "Kling-Gupta Efficiency - balanced performance",
                "NSE" => "Nash-Sutcliffe Efficiency - sensitive to peaks",
                "LogNSE" => "Log-transformed NSE - sensitive to low flows",
                "LogKGE" => "Log-transformed KGE - balanced for low flows",
                "PBIAS" => "Percent Bias - water balance accuracy",
                "R2" => "Coefficient of Determination - pattern matching",
                "RMSE" => "Root Mean Square Error - absolute error"
            )
        )
        return TextResourceContents(
            uri = URI("hydro://guides/objectives"),
            mime_type = "application/json",
            text = JSON3.write(guide)
        )
    end
)

export objective_guide_resource
