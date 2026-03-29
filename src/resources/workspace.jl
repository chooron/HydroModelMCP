using ModelContextProtocol

const runoff_workspace_guide_resource = MCPResource(
    uri = "hydro://guides/runoff-workspace",
    name = "Runoff Workspace Guide",
    title = "Workspace Runoff Workflow",
    description = "Default workspace conventions for single-run runoff simulation and optional evaluation.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "defaults" => Dict(
            "input_dir" => "./data",
            "output_dir" => "./result",
            "forcing_filenames" => ["forcing.csv", "meteo.csv", "meteorology.csv", "input.csv"],
            "observation_filenames" => ["obs.csv", "observed.csv", "discharge.csv", "streamflow.csv"],
        ),
        "recommended_sequence" => [
            "find_model",
            "get_model_info",
            "get_model_parameters",
            "list_workspace_files",
            "run_simulation",
            "compute_metrics (when observations exist)",
            "clear_session_cache",
        ],
        "notes" => [
            "When params are omitted, run_simulation generates a random valid parameter set and labels the run through params_source=random.",
            "Simulation outputs are written under ./result/simulation and summaries under ./result.",
        ],
    ),
)

const result_artifact_guide_resource = MCPResource(
    uri = "hydro://guides/result-artifacts",
    name = "Result Artifact Guide",
    title = "Result Artifact Layout",
    description = "Where HydroModelMCP writes simulation, metrics, and stored-result artifacts.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "workspace_outputs" => Dict(
            "simulation_csv" => "./result/simulation/<base>_result_<timestamp>.csv",
            "simulation_metadata" => "./result/simulation/<base>_result_<timestamp>.metadata.json",
            "run_summary" => "./result/run_summary_<timestamp>.md",
            "metrics_json" => "./result/metrics/<base>_metrics_<timestamp>.json",
        ),
        "stored_result_categories" => ["calibration", "sensitivity", "ensemble"],
        "stored_result_note" => "Stored result resources are registered from the storage backend when the server starts.",
    ),
)

export result_artifact_guide_resource, runoff_workspace_guide_resource
