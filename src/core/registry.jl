module Registry

"""资源注册表：维护数据资源名称与路径的映射。"""

export register_resource, get_resource_path, list_resources

const RESOURCES = Dict{String,String}()

function register_resource(name::String, path::String)
    RESOURCES[name] = path
    return true
end

function get_resource_path(name::String)
    return get(RESOURCES, name, nothing)
end

function list_resources()
    return copy(RESOURCES)
end

end # module