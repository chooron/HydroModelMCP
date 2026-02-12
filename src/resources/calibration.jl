using ModelContextProtocol
using JSON3
using URIs

# Requires Storage module
function create_calibration_resources(storage_backend)
    # Resource 1: List all calibration results
    calibration_list_resource = MCPResource(
        uri = "hydro://calibration/results",
        name = "Calibration Results List",
        description = "List of all stored calibration results with metadata",
        mime_type = "application/json",
        data_provider = () -> begin
            results = Storage.list_results(storage_backend, "calibration")
            return TextResourceContents(
                uri = URI("hydro://calibration/results"),
                mime_type = "application/json",
                text = JSON3.write(Dict("results" => results, "count" => length(results)))
            )
        end
    )

    return [calibration_list_resource]
end

export create_calibration_resources
