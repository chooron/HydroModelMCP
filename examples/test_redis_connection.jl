using Redis
import UUIDs

"""
    test_redis_connection(; host = "127.0.0.1", port = 6379)

Connect to Redis and run a set/get/del probe.
Returns true when Redis is reachable and read/write works.
"""
function test_redis_connection(; host::String = "127.0.0.1", port::Int = 6379)
    conn = nothing
    probe_key = "hydromcp:test:probe:" * string(UUIDs.uuid4())

    try
        conn = Redis.RedisConnection(host = host, port = port)
        Redis.set(conn, probe_key, "ok")
        value = Redis.get(conn, probe_key)
        Redis.del(conn, probe_key)
        return value == "ok"
    catch err
        @error "Redis connection test failed" host = host port = port exception = (err, catch_backtrace())
        return false
    finally
        if !(conn === nothing)
            try
                isopen(conn) && close(conn)
            catch
            end
        end
    end
end

function parse_args(args::Vector{String})
    host = get(ENV, "REDIS_HOST", "127.0.0.1")
    env_port = tryparse(Int, get(ENV, "REDIS_PORT", "6379"))
    env_port === nothing && error("Invalid REDIS_PORT environment variable")
    port = Int(env_port)

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--host")
            i == length(args) && error("Missing value after $(arg)")
            host = args[i + 1]
            i += 2
        elseif arg in ("-p", "--port")
            i == length(args) && error("Missing value after $(arg)")
            parsed_port = tryparse(Int, args[i + 1])
            parsed_port === nothing && error("Invalid port: $(args[i + 1])")
            port = Int(parsed_port)
            i += 2
        elseif arg == "--help"
            println("Usage: julia --project=. examples/test_redis_connection.jl [--host HOST] [--port PORT]")
            println("Defaults: host from REDIS_HOST or 127.0.0.1, port from REDIS_PORT or 6379")
            exit(0)
        else
            error("Unknown argument: $(arg)")
        end
    end

    return host, port
end

function main()
    host, port = parse_args(ARGS)
    println("Testing Redis connectivity at $(host):$(port) ...")

    ok = test_redis_connection(host = host, port = port)
    if ok
        println("SUCCESS: Julia can connect to Redis and perform read/write.")
        return 0
    end

    println(stderr, "FAIL: Julia cannot connect to Redis. Check service status and connection settings.")
    return 1
end

exit(main())
