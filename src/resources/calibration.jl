using ModelContextProtocol

function create_calibration_resources(storage_backend)
    calibration_list_resource = MCPResource(
        uri = "hydro://calibration/results",
        name = "Calibration Results List",
        description = "List of stored calibration result identifiers available in the current backend.",
        mime_type = "application/json",
        data_provider = () -> begin
            results = Storage.list_results(storage_backend, "calibration")
            Dict("results" => results, "count" => length(results))
        end
    )

    return MCPResource[calibration_list_resource]
end

export create_calibration_resources
