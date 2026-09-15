-- AnimeJaNai stats overlay (Ctrl+j): the engine's current chain (from the stats log the
-- filter rewrites on every reconfigure), the inference backend and video renderer in use,
-- and playback fps - the instantaneous output rate, a running average since the overlay
-- was opened (or since the last seek/file change), and dropped frames. Refreshes twice a
-- second while shown.

local mp = require 'mp'
local utils = require 'mp.utils'

local MAX_DURATION = 2147483
local REFRESH = 0.5

local showing = false
local timer = nil
local fps_sum, fps_n = 0, 0

local function read_file(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local content = f:read("*a")
    f:close()
    return content
end

-- [global] backend= from animejanai.conf, the value the inference shim dispatches on.
local function conf_backend()
    local path = mp.command_native({'expand-path', "~~/../animejanai/animejanai.conf"})
    local text = read_file(path) or ""
    local in_global = false
    for line in text:gmatch("[^\r\n]+") do
        local section = line:match("^%s*%[(.-)%]%s*$")
        if section then
            in_global = section:lower() == "global"
        elseif in_global then
            local k, v = line:match("^%s*([%w_]+)%s*=%s*(.-)%s*$")
            if k and k:lower() == "backend" then return v end
        end
    end
    return nil
end

local BACKEND_LABEL = {
    tensorrt = "TensorRT (aji_trt, CUDA)",
    directml = "DirectML (aji_dml, D3D12)",
    ncnn     = "DirectML (aji_dml, D3D12)",
    rocm     = "ROCm (aji_rocm, MIGraphX)",
    vulkan   = "Vulkan (aji_vk, ncnn-Vulkan)",
}

local function backend_line()
    local raw = conf_backend() or "TensorRT"
    return BACKEND_LABEL[raw:lower()] or raw
end

local function renderer_line()
    local vo = mp.get_property("current-vo") or "?"
    local ctx = mp.get_property("current-gpu-context")
    local hwdec = mp.get_property("hwdec-current")
    local s = vo
    if ctx and ctx ~= "" then s = s .. " / " .. ctx end
    if hwdec and hwdec ~= "" and hwdec ~= "no" then s = s .. ", hwdec " .. hwdec else s = s .. ", software decode" end
    return s
end

local function fmt(n) return n and string.format("%.1f", n) or "?" end

local function build_message()
    if (mp.get_property("vf") or "") == "" then
        return "Upscaling is disabled"
    end
    local log_path = mp.command_native({'expand-path', "~~/../animejanai/currentanimejanai.log"})
    local chain = read_file(log_path)
    if not chain or chain == "" then
        chain = "Error during upscale; press ~ to view error in console"
    end
    chain = chain:gsub("%s+$", "")

    -- fps: estimated-vf-fps is the rate frames leave the filter chain (the upscaled
    -- output); average it while playing so pauses don't drag it down.
    local now = mp.get_property_number("estimated-vf-fps")
    if now and now > 0 and not mp.get_property_bool("pause") then
        fps_sum = fps_sum + now; fps_n = fps_n + 1
    end
    local avg = fps_n > 0 and fps_sum / fps_n or nil
    local src = mp.get_property_number("container-fps")
    local dropped = mp.get_property_number("frame-drop-count") or 0
    local w, h = mp.get_property_number("width"), mp.get_property_number("height")
    local ow, oh = mp.get_property_number("dwidth"), mp.get_property_number("dheight")

    return table.concat({
        chain,
        "",
        "Backend:  " .. backend_line(),
        "Renderer: " .. renderer_line(),
        string.format("Source:   %sx%s @ %s fps  ->  output %sx%s",
            w or "?", h or "?", fmt(src), ow or "?", oh or "?"),
        string.format("FPS:      %s now  /  %s avg  /  %d dropped", fmt(now), fmt(avg), dropped),
    }, "\n")
end

local function refresh()
    if not showing then return end
    mp.osd_message(build_message(), MAX_DURATION)
end

local function reset_average()
    fps_sum, fps_n = 0, 0
end

local function toggle()
    if showing then
        showing = false
        if timer then timer:kill(); timer = nil end
        mp.osd_message("")
        return
    end
    showing = true
    reset_average()
    refresh()
    timer = mp.add_periodic_timer(REFRESH, refresh)
end

-- a seek or a new file starts a fresh average (the old numbers describe other content)
mp.register_event("seek", reset_average)
mp.register_event("file-loaded", reset_average)

mp.add_key_binding("Ctrl+j", "show_animejanai_stats", toggle)
