-- Engine-build monitor for vf_animejanai.
--
-- Inference engines are compiled on first play, per model and resolution:
-- TensorRT on Windows/NVIDIA, MIGraphX on Linux/AMD. Builds run on a
-- background thread, but both compilers benchmark GPU kernels while they
-- work and want an idle GPU for the best result - so by default this script
-- pauses playback while a build runs, narrates what is happening on the OSD,
-- and resumes automatically when the engine is ready. Unpausing manually
-- during a build is respected: the video keeps playing (unupscaled) and the
-- script stays hands-off.
--
-- The filter rewrites the stats log on every (re)configure; a
-- "Building <backend> engine for <model> for <res>" line means a build is in
-- flight (the OSD wording adapts per backend).
--
-- script-opts (prefix animejanai_engine_monitor-):
--   auto_pause=yes|no   pause playback during builds (default yes)
--   stats_path=...      override the stats log location

local mp = require 'mp'
local msg = require 'mp.msg'
local options = require 'mp.options'

local o = {
    auto_pause = true,
    poll_interval = 0.25,
    stats_path = "~~/../animejanai/currentanimejanai.log",
}
options.read_options(o, "animejanai_engine_monitor")

local stats_path = mp.command_native({'expand-path', o.stats_path})

local building = false
local we_paused = false
local started_at = 0
local build_name = "?"
local build_res = "?"
-- which backend is compiling: "TensorRT" (Windows) or "MIGraphX" (Linux/AMD).
-- The filter writes "Building <backend> engine for <model> for <res>"; the OSD
-- wording adapts per backend (TensorRT keeps its familiar text).
local build_backend = "?"

local function read_stats()
    local f = io.open(stats_path, "r")
    if not f then return nil end
    local s = f:read("*a")
    f:close()
    return s
end

mp.add_periodic_timer(o.poll_interval, function()
    local s = read_stats()
    -- matches both "Building TensorRT engine for ..." and "Building MIGraphX engine for ..."
    local b = s ~= nil and s:find("Building %S+ engine for") ~= nil

    -- the user taking over wins: if they unpause mid-build, stay hands-off
    if we_paused and not mp.get_property_bool("pause") then
        we_paused = false
    end

    if b and not building then
        building = true
        started_at = mp.get_time()
        if o.auto_pause and not mp.get_property_bool("pause") then
            mp.set_property_bool("pause", true)
            we_paused = true
            msg.info("engine build started; pausing playback")
        else
            msg.info("engine build started")
        end
    elseif not b and building then
        building = false
        local failed = s ~= nil and s:find("build FAILED", 1, true) ~= nil
        msg.info(string.format("engine build finished after %ds%s",
            math.floor(mp.get_time() - started_at),
            we_paused and "; resuming playback" or ""))
        if we_paused then
            mp.set_property_bool("pause", false)
            we_paused = false
        end
        if failed then
            if build_backend == "TensorRT" then
                local log_path = s and s:match("%(see ([^%)]+)%)") or
                                 "the .build.log file next to the model"
                mp.osd_message(string.format(
                    "AnimeJaNai: Building TensorRT engine for %s for %s " ..
                    "failed. Upscaling is disabled. (details: %s).",
                    build_name, build_res, log_path), 10)
            else
                mp.osd_message(string.format(
                    "AnimeJaNai: Optimizing the upscaler for %s at %s failed. " ..
                    "Upscaling is disabled.", build_name, build_res), 10)
            end
        else
            if build_backend == "TensorRT" then
                mp.osd_message(string.format(
                    "AnimeJaNai: Building TensorRT engine for %s for %s " ..
                    "completed successfully. Upscaling is active.",
                    build_name, build_res), 5)
            else
                mp.osd_message(string.format(
                    "AnimeJaNai: Upscaler optimized for %s at %s. " ..
                    "Upscaling is active.", build_name, build_res), 5)
            end
        end
    end

    if building then
        local bk, n, r = s:match(
            "Building (%S+) engine for (%S+) for (%S+)")
        if n then
            build_backend = bk
            build_name = n
            build_res = r
        end
        local elapsed = math.floor(mp.get_time() - started_at)
        local second = we_paused and "Playback will resume on completion."
                                  or "Upscaling will activate on completion."
        local text
        if build_backend == "TensorRT" then
            text = string.format(
                "AnimeJaNai: Building TensorRT engine for %s for %s " ..
                "(%ds, usually about a minute)\n%s",
                build_name, build_res, elapsed, second)
        else
            -- AMD/MIGraphX: a per-resolution engine, ~2 min the first time only
            text = string.format(
                "AnimeJaNai: Optimizing the upscaler for your GPU\n" ..
                "%s at %s (%ds — first play at this resolution, ~2 min)\n%s",
                build_name, build_res, elapsed, second)
        end
        mp.osd_message(text, o.poll_interval + 0.5)
        -- while paused no frames flow, so the filter would never notice the
        -- finished build; this no-op command wakes it so it polls
        mp.commandv("vf-command", "aji", "poll", "1")
    end
end)
