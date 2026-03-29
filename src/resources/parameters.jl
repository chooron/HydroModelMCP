using ModelContextProtocol

const objective_guide_resource = MCPResource(
    uri = "hydro://guides/objectives",
    name = "Objective Function Selection Guide",
    title = "Objective Selection Guide",
    description = "Mapping of calibration goals to recommended objective functions.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "mappings" => [
            Dict("goal" => "general_fit", "primary" => "KGE", "secondary" => ["NSE", "PBIAS"], "log_transform" => false),
            Dict("goal" => "peak_flows", "primary" => "NSE", "secondary" => ["KGE", "RMSE"], "log_transform" => false),
            Dict("goal" => "low_flows", "primary" => "LogKGE", "secondary" => ["LogNSE", "PBIAS"], "log_transform" => true),
            Dict("goal" => "water_balance", "primary" => "PBIAS", "secondary" => ["KGE", "R2"], "log_transform" => false),
            Dict("goal" => "dynamics", "primary" => "R2", "secondary" => ["KGE"], "log_transform" => false),
        ],
        "metrics" => Dict(
            "KGE" => "Kling-Gupta Efficiency balances correlation, variability, and bias.",
            "NSE" => "Nash-Sutcliffe Efficiency emphasizes peak-flow fit.",
            "LogNSE" => "Log-transformed NSE emphasizes low flows.",
            "LogKGE" => "Log-transformed KGE balances low-flow performance.",
            "PBIAS" => "Percent Bias emphasizes volume conservation.",
            "R2" => "Coefficient of determination emphasizes pattern similarity.",
            "RMSE" => "Root Mean Square Error emphasizes absolute error magnitude.",
        ),
    ),
)

const metrics_guide_resource = MCPResource(
    uri = "hydro://guides/metrics",
    name = "Metrics Guide",
    title = "Metrics Reference",
    description = "Supported hydrological metrics, optimization direction, and selection hints.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "metrics" => [
            Dict("name" => "NSE", "higher_is_better" => true, "best_for" => ["peak flows", "overall hydrograph fit"]),
            Dict("name" => "KGE", "higher_is_better" => true, "best_for" => ["balanced fit", "general calibration"]),
            Dict("name" => "LogNSE", "higher_is_better" => true, "best_for" => ["low flows", "wide magnitude ranges"]),
            Dict("name" => "LogKGE", "higher_is_better" => true, "best_for" => ["low flows", "balanced low-flow calibration"]),
            Dict("name" => "PBIAS", "higher_is_better" => false, "best_for" => ["water balance", "volume bias review"]),
            Dict("name" => "R2", "higher_is_better" => true, "best_for" => ["temporal dynamics", "pattern similarity"]),
            Dict("name" => "RMSE", "higher_is_better" => false, "best_for" => ["absolute error magnitude", "large-error penalty"]),
            Dict("name" => "MAE", "higher_is_better" => false, "best_for" => ["robust average error", "outlier-resistant summaries"]),
            Dict("name" => "Bias", "higher_is_better" => false, "best_for" => ["signed mean bias", "directional error review"]),
        ],
        "notes" => [
            "Log-domain metrics are usually appropriate when positive observed flows span more than two orders of magnitude.",
            "compute_metrics may include advisory fields such as _magnitude_ratio and _log_transform_recommended.",
        ],
    ),
)

const data_handle_guide_resource = MCPResource(
    uri = "hydro://guides/data-handles",
    name = "Data Handle Guide",
    title = "Data Handle Workflow",
    description = "Guidance for when to use load_hydro_csv data handles versus direct file paths.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "preferred_patterns" => [
            Dict(
                "workflow" => "single simulation",
                "recommended_entry" => "run_simulation",
                "why" => "Direct file paths keep the call compact and deterministic.",
            ),
            Dict(
                "workflow" => "calibration or repeated diagnostics",
                "recommended_entry" => "load_hydro_csv",
                "why" => "A reusable data_handle avoids resending large forcing and observation payloads.",
            ),
        ],
        "cleanup" => Dict(
            "tool" => "clear_session_cache",
            "when" => "After workflows that created transient handles.",
            "note" => "This clears session cache only and does not delete files under ./result.",
        ),
    ),
)

export data_handle_guide_resource, metrics_guide_resource, objective_guide_resource
