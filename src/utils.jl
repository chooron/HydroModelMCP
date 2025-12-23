module HydroModelMCP.Utils

"""通用工具函数集（数据格式转换、快速读取等）。"""

export parse_csv, read_data

using CSV
using DataFrames

"""读取给定路径的 CSV 并返回 DataFrame。"""
function parse_csv(path::AbstractString)
    @info "Parsing CSV" path=path
    return CSV.read(path, DataFrame)
end

"""读取数据文件（目前简单代理到 parse_csv）。

TODO: 通过资源注册表支持相对资源名加载。
"""
function read_data(fname::AbstractString)
    return parse_csv(fname)
end

end # module
