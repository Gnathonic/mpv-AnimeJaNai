// Package builder for the native-filter mpv-upscale-2x_animejanai.
//
// The package has no VapourSynth, Python, or vs-mlrt plugins: upscaling and
// RIFE run inside the mpv fork's vf_animejanai filter, which loads aji.dll
// (github.com/the-database/animejanai-inference). Everything NVIDIA lives in
// one self-contained animejanai/inference/ directory (the filter resolves
// the shim's dependencies from its own directory).
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.Zip;
using SevenZipExtractor;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using static Downloader;

// Third-party component versions. Bump these together when cutting a release.
// The inference runtime (TensorRT + trtexec) is reused from the vs-mlrt cuda
// release archives: publicly downloadable, license-precedented, and trtexec
// is version-matched to nvinfer by construction. aji_trt.dll must be built
// against the SAME TensorRT major.minor (v16.x == TensorRT 11.0).
// NOTE: v16.test1 is vs-mlrt's TRT 11 PRE-release - recheck for a stable
// v16 tag before cutting the package release.
const string VsMlrtCudaVersion    = "v16.test1";
const string AjiVersion           = "v0.6.0";       // github.com/the-database/animejanai-inference release tag (DML 4:4:4 input; op21 SD preset; missing-model passthrough; configurable RIFE/upscale order; benchmark slots 1012/1013)

const string SevenZipVersion      = "2501";         // 7-zip "extra" standalone console version
const string MpvNetVersion        = "v7.1.2.0";
const string ManagerVersion       = "0.4.0";        // github.com/the-database/AnimeJaNaiManager release tag (AnimeJaNai Manager)

// DirectML backend runtime (backend=DirectML in animejanai.conf). These are
// the last DirectML-flavored releases: Microsoft moved DML to sustained
// engineering, so 1.24.x is the ORT ceiling until the WinML migration.
const string OrtDmlVersion        = "1.24.4";       // Microsoft.ML.OnnxRuntime.DirectML (NuGet)
const string DirectMLVersion      = "1.15.4";       // Microsoft.AI.DirectML (NuGet)
const string RifeModelsVersion    = "models-rife-fp16-1"; // animejanai-inference release tag (fp16 conversions)

// Custom libmpv fork build (github.com/the-database/mpv-winbuild release).
// The release tag and the archive filename's date can differ (the tag is
// stamped at publish, the filename at build), so they are pinned separately.
const string MpvForkVersion       = "2026-06-18-9bb5fe9680"; // release tag
                                                    // the vf_animejanai filter lives inside this libmpv build; rebuild + bump alongside AjiVersion when the filter changes. This build carries DML 4:4:4 (x2bgr10) input + rife_before_upscale (RIFE-first) on both TRT and DML.
const string MpvForkBuildDate     = "20260618";     // build date in the dev archive filename
const string MpvForkGitHash       = "9bb5fe9680";   // git short hash in the dev archive filename

// TensorRT runtime files taken from the vs-mlrt cuda archive's vsmlrt-cuda/
// directory. Everything else in there (cuDNN, cuBLAS, onnxruntime, the lean
// and dispatch runtimes) serves backends/options the native filter does not
// use; engine builds run with --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT.
string[] inferenceRuntimeFiles = [
    "nvinfer_11.dll",
    "nvinfer_plugin_11.dll",
    "nvonnxparser_11.dll",
    "trtexec.exe",
];
string[] inferenceRuntimePrefixes = [
    "cudart64_",
    "nvinfer_builder_resource_",
];

if (args.Length < 1)
{
    throw new ArgumentException("Version is required.");
}

var assemblyDirectory = AppContext.BaseDirectory;
var animejanaiDirectory = Path.Combine(assemblyDirectory, "mpv-upscale-2x_animejanai");
var installDirectory = Path.Combine(assemblyDirectory, $"mpv-upscale-2x_animejanai-v{args[0]}");

// --packs-only [dir]: emit component packs from an already-built install tree
// (default: the version-derived directory above) and exit, skipping the build.
int packsOnlyIndex = Array.IndexOf(args, "--packs-only");
if (packsOnlyIndex >= 0 && packsOnlyIndex + 1 < args.Length &&
    Directory.Exists(args[packsOnlyIndex + 1]))
{
    installDirectory = Path.GetFullPath(args[packsOnlyIndex + 1]);
}

var inferencePath = Path.Combine(installDirectory, "animejanai", "inference");
var onnxPath = Path.Combine(installDirectory, "animejanai", "onnx");
var rifePath = Path.Combine(installDirectory, "animejanai", "rife");

// Standalone 7-Zip console (7za.exe): used here to extract the multi-part
// vs-mlrt archive, and shipped at the install root for the updater
// (manifest archive_tool).
async Task InstallSevenZip()
{
    Console.WriteLine("Downloading 7-Zip standalone console...");
    var downloadUrl = $"https://www.7-zip.org/a/7z{SevenZipVersion}-extra.7z";
    var targetPath = Path.GetFullPath("7z-extra.7z");
    await DownloadFileAsync(downloadUrl, targetPath, (progress) =>
    {
        Console.WriteLine($"Downloading 7-Zip ({progress}%)...");
    });

    var targetExtractPath = Path.GetFullPath("7z-extra-temp");
    Directory.CreateDirectory(targetExtractPath);
    using (ArchiveFile archiveFile = new(targetPath))
    {
        archiveFile.Extract(targetExtractPath);
    }
    File.Copy(Path.Combine(targetExtractPath, "x64", "7za.exe"),
              Path.Combine(installDirectory, "7za.exe"), true);
    Directory.Delete(targetExtractPath, true);
    File.Delete(targetPath);
}

async Task InstallInferenceRuntime()
{
    Console.WriteLine("Downloading TensorRT runtime (from the vs-mlrt cuda release)...");
    var baseDownloadUrl = $"https://github.com/AmusementClub/vs-mlrt/releases/download/{VsMlrtCudaVersion}/";
    var fileNames = new[]
    {
        $"vsmlrt-windows-x64-cuda.{VsMlrtCudaVersion}.7z.001",
        $"vsmlrt-windows-x64-cuda.{VsMlrtCudaVersion}.7z.002",
    };
    var targetPaths = fileNames.Select(f => Path.GetFullPath(f)).ToArray();

    double lastProgress = -1;
    int updateThreshold = 5;

    for (int i = 0; i < fileNames.Length; i++)
    {
        string downloadUrl = baseDownloadUrl + fileNames[i];
        string targetPath = targetPaths[i];

        await DownloadFileAsync(downloadUrl, targetPath, (progress) =>
        {
            if (progress >= lastProgress + updateThreshold)
            {
                Console.WriteLine($"Downloading {fileNames[i]} ({progress}%)...");
                lastProgress = progress;
            }
        });
    }

    Console.WriteLine("Extracting TensorRT runtime (this may take several minutes)...");
    var tempDirectory = Path.GetFullPath("vsmlrt-temp");
    Directory.CreateDirectory(tempDirectory);

    // Only vsmlrt-cuda/ is needed (a flat directory); extracting just that
    // subtree also skips the plugin DLLs (vstrt/vsort/...) entirely.
    await RunProcess(Path.Combine(installDirectory, "7za.exe"),
                     $"x \"{targetPaths[0]}\" -o\"{tempDirectory}\" \"vsmlrt-cuda\\*\" -r- -y");

    Directory.CreateDirectory(inferencePath);
    var cudaDirectory = Path.Combine(tempDirectory, "vsmlrt-cuda");
    foreach (var file in Directory.GetFiles(cudaDirectory))
    {
        var name = Path.GetFileName(file);
        bool keep = inferenceRuntimeFiles.Contains(name) ||
                    inferenceRuntimePrefixes.Any(p => name.StartsWith(p, StringComparison.OrdinalIgnoreCase)) ||
                    name.Contains("LICENSE", StringComparison.OrdinalIgnoreCase);
        if (keep)
        {
            File.Copy(file, Path.Combine(inferencePath, name), true);
        }
    }

    Directory.Delete(tempDirectory, true);
    foreach (var targetPath in targetPaths)
    {
        File.Delete(targetPath);
    }
}

async Task InstallAji()
{
    Directory.CreateDirectory(inferencePath);

    // Dev override: point AJI_LOCAL_ZIP at a locally built archive.
    var localZip = Environment.GetEnvironmentVariable("AJI_LOCAL_ZIP");
    string targetPath;
    if (!string.IsNullOrEmpty(localZip))
    {
        Console.WriteLine($"Using local aji build: {localZip}");
        targetPath = localZip;
    }
    else
    {
        Console.WriteLine("Downloading aji (native inference shim)...");
        var downloadUrl = $"https://github.com/the-database/animejanai-inference/releases/download/{AjiVersion}/aji-windows-x64.zip";
        targetPath = Path.GetFullPath("aji-windows-x64.zip");
        await DownloadFileAsync(downloadUrl, targetPath, (progress) =>
        {
            Console.WriteLine($"Downloading aji ({progress}%)...");
        });
    }

    ExtractZip(targetPath, inferencePath, (double progress) => { });

    if (string.IsNullOrEmpty(localZip))
    {
        File.Delete(targetPath);
    }
}

// ONNX Runtime + DirectML for the DirectML backend. The .nupkg files are
// plain zips; only the x64 runtime DLLs (and the DirectML license, which the
// redistribution terms require keeping intact) go into the package. Load
// order at runtime is handled by aji_dml.dll (DirectML.dll before
// onnxruntime.dll, both from this directory).
async Task InstallOrtDml()
{
    Directory.CreateDirectory(inferencePath);
    var packages = new (string Name, string Version, string[] CopyFromTo)[]
    {
        ("Microsoft.ML.OnnxRuntime.DirectML", OrtDmlVersion, new[]
        {
            "runtimes/win-x64/native/onnxruntime.dll", "onnxruntime.dll",
        }),
        ("Microsoft.AI.DirectML", DirectMLVersion, new[]
        {
            "bin/x64-win/DirectML.dll", "DirectML.dll",
            "LICENSE.txt", "DirectML_LICENSE.txt",
        }),
    };
    foreach (var (name, version, copies) in packages)
    {
        Console.WriteLine($"Downloading {name} {version}...");
        var downloadUrl = $"https://www.nuget.org/api/v2/package/{name}/{version}";
        var targetPath = Path.GetFullPath($"{name}.{version}.nupkg");
        await DownloadFileAsync(downloadUrl, targetPath, _ => { });

        var tempDirectory = Path.GetFullPath($"{name}-temp");
        ExtractZip(targetPath, tempDirectory, _ => { });
        for (var i = 0; i < copies.Length; i += 2)
        {
            var src = Path.Combine(tempDirectory,
                                   copies[i].Replace('/', Path.DirectorySeparatorChar));
            File.Copy(src, Path.Combine(inferencePath, copies[i + 1]), true);
        }
        Directory.Delete(tempDirectory, true);
        File.Delete(targetPath);
    }
}

async Task InstallRife()
{
    // fp16 conversions of vs-mlrt's rife v1 (video_player) models — one
    // model set for both backends (DirectML runs them faster than fp32
    // at reference-class quality; TensorRT 11's strong typing requires
    // fp16 onnx). Converted by animejanai-inference's
    // tools/convert_rife_fp16.py (GridSample grid math kept fp32) and
    // hosted as a single release asset. Lives outside onnx/ so the
    // heavy, deps-versioned models stay out of the overlay archive.
    Console.WriteLine("Downloading RIFE fp16 models...");
    var downloadUrl = "https://github.com/the-database/animejanai-inference/" +
                      $"releases/download/{RifeModelsVersion}/rife-fp16-1.7z";
    var targetPath = Path.GetFullPath("rife-fp16.7z");
    await DownloadFileAsync(downloadUrl, targetPath, (progress) =>
    {
        Console.WriteLine($"Downloading RIFE fp16 models ({progress}%)...");
    });

    Directory.CreateDirectory(rifePath);
    using (ArchiveFile archiveFile = new(targetPath))
    {
        archiveFile.Extract(rifePath);
    }
    File.Delete(targetPath);
}

async Task InstallMpvnet()
{
    var downloadUrl = $"https://github.com/mpvnet-player/mpv.net/releases/download/{MpvNetVersion}/mpv.net-{MpvNetVersion}-portable-x64.zip";
    var targetPath = Path.GetFullPath("mpvnet.zip");
    await DownloadFileAsync(downloadUrl, targetPath, (progress) =>
    {
        Console.WriteLine($"Downloading mpv.net ({progress}%)...");
    });

    Console.WriteLine("Extracting mpv.net...");
    ExtractZip(targetPath, installDirectory, (double progress) =>
    {
        Console.WriteLine($"Extracting mpv.net ({progress}%)...");
    });

    File.Delete(targetPath);
}

async Task InstallCustomLibmpv()
{
    Console.WriteLine("Downloading custom libmpv fork...");
    var downloadUrl = $"https://github.com/the-database/mpv-winbuild/releases/download/{MpvForkVersion}/mpv-dev-x86_64-{MpvForkBuildDate}-git-{MpvForkGitHash}.7z";
    var targetPath = Path.GetFullPath("mpv-dev.7z");
    await Downloader.DownloadFileAsync(downloadUrl, targetPath, (progress) =>
    {
        Console.WriteLine($"Downloading custom libmpv fork ({progress}%)...");
    });

    Console.WriteLine("Extracting custom libmpv fork...");
    var targetExtractPath = Path.Combine(installDirectory, "temp-libmpv");
    Directory.CreateDirectory(targetExtractPath);

    using (ArchiveFile archiveFile = new(targetPath))
    {
        archiveFile.Extract(targetExtractPath);

        File.Copy(
            Path.Combine(targetExtractPath, "libmpv-2.dll"),
            Path.Combine(installDirectory, "libmpv-2.dll"),
            true // overwrite the stock mpv.net libmpv-2.dll
        );
    }
    Directory.Delete(targetExtractPath, true);
    File.Delete(targetPath);
}

async Task InstallYtDlp()
{
    var downloadUrl = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe";
    var targetPath = Path.Combine(installDirectory, "yt-dlp.exe");
    await DownloadFileAsync(downloadUrl, targetPath, (progress) =>
    {
        Console.WriteLine($"Downloading yt-dlp.exe... ({progress})%");
    });
}

void InstallAnimeJaNaiCore()
{
    CopyDirectory(animejanaiDirectory, installDirectory);
}

// mpv's input.conf has no `include` (unlike mpv.conf) AND mpv.net builds its
// right-click menu by parsing input.conf's #menu: annotations - so the managed
// AnimeJaNai keybindings must physically live in input.conf. To still let users
// own input.conf (edit it, keep changes across updates) it is generated as a
// regenerable managed block (sourced from input-animejanai.conf, refreshed by
// the updater) followed by the user's own section. Bindings below the END
// marker override the managed ones above (mpv applies later bindings last).
void GenerateInputConf()
{
    var pc = Path.Combine(installDirectory, "portable_config");
    var managed = File.ReadAllText(Path.Combine(pc, "input-animejanai.conf"))
        .Replace("\r\n", "\n").TrimEnd('\n');
    // Markers must match AnimeJaNaiUpdater's regenerator exactly.
    var conf =
        "# Your keybindings. Safe to edit - updates never overwrite your section.\n" +
        "#\n" +
        "# The block between the markers below is managed by AnimeJaNai and refreshed\n" +
        "# on every update - do not edit inside it (changes there are replaced). Add\n" +
        "# your own keybindings UNDER the END marker; they survive updates and override\n" +
        "# the defaults above (mpv applies later bindings last). Syntax: one\n" +
        "# \"KEY  command\" per line, same as the managed block.\n" +
        "\n" +
        "#@ANIMEJANAI-MANAGED-BEGIN (do not edit this line or the block below)\n" +
        managed + "\n" +
        "#@ANIMEJANAI-MANAGED-END (add your keybindings below this line)\n" +
        "\n" +
        "# ===== Your keybindings below =====\n";
    File.WriteAllText(Path.Combine(pc, "input.conf"), conf);
}

async Task InstallAnimeJaNaiManager()
{
    Console.WriteLine("Downloading AnimeJaNai Manager...");
    var downloadUrl = $"https://github.com/the-database/AnimeJaNaiManager/releases/download/{ManagerVersion}/AnimeJaNaiManager-portable-x64.zip";
    var targetPath = Path.GetFullPath("AnimeJaNaiManager-portable-x64.zip");
    await DownloadFileAsync(downloadUrl, targetPath, (progress) =>
    {
        Console.WriteLine($"Downloading AnimeJaNai Manager ({progress}%)...");
    });

    Console.WriteLine("Extracting AnimeJaNai Manager...");
    // The zip is flat (AnimeJaNaiManager.exe + native DLLs) and lands at the install root,
    // next to mpvnet.exe, for discoverability. It finds its data in animejanai/.
    ExtractZip(targetPath, installDirectory, (double progress) =>
    {
        Console.WriteLine($"Extracting AnimeJaNai Manager ({progress}%)...");
    });

    File.Delete(targetPath);
}

// The TensorRT SLA requires this attribution when redistributing the
// runtime; keep it next to the redistributed files.
void WriteThirdPartyNotices()
{
    var notice = """
        Third-party components in this directory
        ========================================

        NVIDIA TensorRT runtime (nvinfer_11.dll, nvinfer_plugin_11.dll,
        nvonnxparser_11.dll, nvinfer_builder_resource_*.dll, trtexec.exe)
        and NVIDIA CUDA runtime (cudart64_*.dll), redistributed under the
        NVIDIA TensorRT Software License Agreement and CUDA Toolkit EULA:

            This software contains source code provided by NVIDIA Corporation.

        These files are obtained from the vs-mlrt project's release archives
        (https://github.com/AmusementClub/vs-mlrt), which redistributes them
        under the same terms.

        ONNX Runtime (onnxruntime.dll), (c) Microsoft Corporation,
        redistributed under the MIT license
        (https://github.com/microsoft/onnxruntime/blob/main/LICENSE).

        DirectML (DirectML.dll), (c) Microsoft Corporation, redistributed
        as the DirectML Redistributable Package under the Microsoft
        Software License Terms shipped alongside it as
        DirectML_LICENSE.txt (use on Windows and Xbox only).

        aji.dll / aji_trt.dll / aji_dml.dll / aji_harness.exe /
        aji_harness_dml.exe / aji_kernel_test.exe:
        https://github.com/the-database/animejanai-inference
        """;
    File.WriteAllText(Path.Combine(inferencePath, "THIRD_PARTY_NOTICES.txt"), notice);
}

// Writes version.txt + manifest.json into the install root. The updater (AnimeJaNaiUpdater) reads
// these to know the installed version, decide overlay-vs-full updates (by comparing deps), and know
// which paths to overwrite (overlay_paths) vs preserve (user_preserve). deploy.yml reads
// overlay_paths from manifest.json to build the lightweight overlay archive.
void WriteVersionAndManifest()
{
    var version = args[0];
    File.WriteAllText(Path.Combine(installDirectory, "version.txt"), version);

    var manifest = new
    {
        package_version = version,
        // Platform-specific names the updater needs. Each platform's builder emits its own values
        // (a future Linux builder would use e.g. "mpv" / "7zz") so the same updater code works
        // cross-platform without hardcoding Windows assumptions.
        player_executable = "mpvnet.exe",
        archive_tool = "7za.exe",
        // Heavy dependencies. If these are unchanged between releases the updater applies the small
        // overlay; if any differ it falls back to the full package. ManagerVersion is omitted on
        // purpose: the editor ships inside the overlay, so it updates without a full download.
        deps = new
        {
            mpvnet = MpvNetVersion,
            mpvfork = $"{MpvForkVersion}-{MpvForkGitHash}",
            inference_runtime = VsMlrtCudaVersion,
            ort_dml = $"{OrtDmlVersion}+{DirectMLVersion}",
            sevenzip = SevenZipVersion,
            rife = RifeModelsVersion,
        },
        // Managed program files (relative to install root) that make up the overlay update and are
        // overwritten on update. Extraction overlays these without deleting extras (e.g. user onnx).
        // aji.dll and its tools are small and update often, so they ride the overlay; the TensorRT
        // runtime files in the same directory are deps-versioned and only change on full updates.
        overlay_paths = new[]
        {
            "version.txt",
            "manifest.json",
            "AnimeJaNaiUpdater.exe",
            "animejanai/onnx",
            "animejanai/benchmarks",
            "animejanai/inference/aji.dll",
            "animejanai/inference/aji_trt.dll",
            "animejanai/inference/aji_dml.dll",
            "animejanai/inference/aji_harness.exe",
            "animejanai/inference/aji_harness_dml.exe",
            "animejanai/inference/aji_kernel_test.exe",
            "AnimeJaNaiManager.exe",
            "av_libglesv2.dll",
            "libHarfBuzzSharp.dll",
            "libSkiaSharp.dll",
            "portable_config/scripts",
            "portable_config/shaders",
            // Managed defaults files, overwritten on update. The user-facing
            // mpv.conf/input.conf (which carry these) are preserved (user_preserve);
            // for input.conf the updater refreshes its managed block from
            // input-animejanai.conf while keeping the user's keybindings.
            "portable_config/mpv-animejanai.conf",
            "portable_config/input-animejanai.conf",
        },
        // User data never overwritten by an update (full updates preserve these explicitly).
        user_preserve = new[]
        {
            "animejanai/animejanai.conf",
            "animejanai/currentanimejanai.log",
            // mpv.conf and input.conf are the user's own files now (they carry the
            // managed defaults - mpv.conf via include, input.conf via a managed block
            // the updater refreshes). Never overwrite them. Upgrades from <=3.3.x
            // still get the new versions because those releases list them under
            // overlay; the updater then folds the old mpv-user.conf / input-user.conf
            // into them and deletes those retired files.
            "portable_config/mpv.conf",
            "portable_config/input.conf",
            "portable_config/saved-props.json",
            "portable_config/settings.xml",
            "portable_config/screenshots",
        },
    };

    var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
    File.WriteAllText(Path.Combine(installDirectory, "manifest.json"), json);
}

void ExtractZip(string archivePath, string outFolder, ProgressChanged progressChanged)
{

    using (var fsInput = File.OpenRead(archivePath))
    using (var zf = new ZipFile(fsInput))
    {

        for (var i = 0; i < zf.Count; i++)
        {
            ZipEntry zipEntry = zf[i];

            if (!zipEntry.IsFile)
            {
                // Ignore directories
                continue;
            }
            String entryFileName = zipEntry.Name;

            var fullZipToPath = Path.Combine(outFolder, entryFileName);
            var directoryName = Path.GetDirectoryName(fullZipToPath);
            if (directoryName?.Length > 0)
            {
                Directory.CreateDirectory(directoryName);
            }

            var buffer = new byte[4096];

            using (var zipStream = zf.GetInputStream(zipEntry))
            using (Stream fsOutput = File.Create(fullZipToPath))
            {
                StreamUtils.Copy(zipStream, fsOutput, buffer);
            }

            var percentage = Math.Round((double)i / zf.Count * 100, 0);
            progressChanged?.Invoke(percentage);
        }
    }
}

async Task RunProcess(string fileName, string arguments)
{
    Debug.WriteLine($"{fileName} {arguments}");

    var process = new Process()
    {
        StartInfo = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        }
    };

    process.Start();
    string output = await process.StandardOutput.ReadToEndAsync();
    string error = await process.StandardError.ReadToEndAsync();
    await process.WaitForExitAsync();

    if (process.ExitCode != 0)
    {
        throw new Exception($"{fileName} failed (exit {process.ExitCode}): {error}");
    }
}

void CopyDirectory(string srcDir, string targetDir)
{
    Directory.CreateDirectory(targetDir);

    foreach (string file in Directory.GetFiles(srcDir))
    {
        // Never ship per-machine/per-GPU runtime cruft: MIGraphX engine caches
        // (.mxr, gfx-specific) and hipRTC color code objects (.co). Both are
        // regenerated on first run; this mirrors the engine repo's .gitignore so a
        // package staged from a directory that was played in stays pristine.
        var ext = Path.GetExtension(file);
        if (ext.Equals(".mxr", StringComparison.OrdinalIgnoreCase) ||
            ext.Equals(".co", StringComparison.OrdinalIgnoreCase))
            continue;
        string targetFilePath = Path.Combine(targetDir, Path.GetFileName(file));
        File.Copy(file, targetFilePath, true); // true to overwrite existing files
    }

    foreach (string subDir in Directory.GetDirectories(srcDir))
    {
        // Skip the runtime JIT code-object cache (animejanai/cache/).
        if (Path.GetFileName(subDir).Equals("cache", StringComparison.OrdinalIgnoreCase))
            continue;
        string newTargetDir = Path.Combine(targetDir, Path.GetFileName(subDir));
        CopyDirectory(subDir, newTargetDir);
    }
}

async Task Main()
{
    if (Directory.Exists(installDirectory))
    {
        Directory.Delete(installDirectory, true);
    }
    Directory.CreateDirectory(installDirectory);
    await InstallSevenZip();
    await InstallInferenceRuntime();
    await InstallAji();
    await InstallOrtDml();
    await InstallRife();
    await InstallMpvnet();
    await InstallCustomLibmpv();
    await InstallYtDlp();
    InstallAnimeJaNaiCore();
    GenerateInputConf();
    await InstallAnimeJaNaiManager();
    WriteThirdPartyNotices();
    WriteVersionAndManifest();
    if (args.Contains("--packs"))
    {
        var packFiles = await EmitComponentPacks();
        SlimInstallTree(packFiles);
    }
}

// Linux assembly: instead of downloading the Windows runtimes (TensorRT, DirectML,
// mpv.net), bundle the locally-built Vulkan stack — the mpv fork binary, the aji
// dispatcher + ncnn-Vulkan backend (libaji.so/libaji_vk.so) + ncnn, and the ncnn
// .param/.bin models — then rewrite the managed configs for backend=vulkan. The
// artifact source dirs are env-overridable (defaults are the dev build trees);
// once the engine + mpv fork publish Linux release assets these become downloads.
async Task MainLinux()
{
    Console.WriteLine("Assembling the Linux (ROCm/MIGraphX) package...");
    string home     = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    string ajiBuild = Environment.GetEnvironmentVariable("AJI_LINUX_BUILD_DIR") ?? Path.Combine(home, "Projects/animejanai-inference/build");
    string mpvBin   = Environment.GetEnvironmentVariable("MPV_FORK_BIN")        ?? "/tmp/mpvfork/build/mpv";

    if (Directory.Exists(installDirectory)) Directory.Delete(installDirectory, true);
    Directory.CreateDirectory(installDirectory);

    // 1. engine libs -> animejanai/inference/ (the dispatcher loads libaji_rocm.so from
    // its own dir). aji_rocm links MIGraphX + HIP + hipRTC from the SYSTEM ROCm install
    // (rpath /opt/rocm/lib) — ROCm must be installed; nothing ROCm is bundled (it's
    // gigabytes, like CUDA for the TensorRT backend). Runtime prereqs on the target:
    // libmigraphx_c.so.3, libamdhip64.so.7, libhiprtc.so (the color kernels are arch-
    // agnostic and JIT-compile per-GPU at first run, caching to animejanai/cache/*.co;
    // any standard ROCm 5.x+ install provides all three). Rebuild this .so from
    // animejanai-inference linux-vulkan-backend (a2692c4+): plain C++, no HIP arch list.
    var inference = Path.Combine(installDirectory, "animejanai", "inference");
    Directory.CreateDirectory(inference);
    foreach (var so in new[] { "libaji.so", "libaji_rocm.so" })
        File.Copy(Path.Combine(ajiBuild, so), Path.Combine(inference, so), true);

    // 2. mpv fork binary (standalone; libmpv is embedded)
    var mpvDst = Path.Combine(installDirectory, "mpv");
    File.Copy(mpvBin, mpvDst, true);
    SetExec(mpvDst);

    // 3. overlay (portable_config + animejanai/animejanai.conf + benchmarks)
    InstallAnimeJaNaiCore();

    // 4. models: aji_rocm runs the .onnx DIRECTLY (the same fp16 SPAN models the TRT/DML
    // backends ship: 3 standard + 2 sharp + SD op21), already placed by the overlay in
    // step 3 — no ncnn .param conversion. MIGraphX compiles a per-(model,resolution)
    // engine to a .mxr next to the .onnx on first use, then caches it.
    var modelDir = Path.Combine(installDirectory, "animejanai", "onnx");
    int onnxCount = Directory.Exists(modelDir) ? Directory.GetFiles(modelDir, "*.onnx").Length : 0;
    Console.WriteLine($"  {onnxCount} .onnx models shipped for MIGraphX");

    // 5. Linux conf rewrites (Windows source files untouched; only the assembled copies change)
    // Preserve the shipped default conf (the [slot_1..9] "New Profile" placeholders the
    // Manager edits) for parity; only swap the backend to the ROCm dispatcher. Built-in
    // slots 1001/1002/1003 (single-model presets) and any custom multi-model chains the
    // user defines via the Manager both work on aji_rocm; default_slot stays 1002.
    var ajiConf = Path.Combine(installDirectory, "animejanai", "animejanai.conf");
    if (File.Exists(ajiConf))
        File.WriteAllText(ajiConf, File.ReadAllText(ajiConf)
            .Replace("backend=TensorRT", "backend=rocm")
            .Replace("backend=DirectML", "backend=rocm"));
    else
        File.WriteAllText(ajiConf,
            "[global]\nconfig_version=3\nbackend=rocm\nlogging=yes\ndefault_slot=1002\n");
    var pc  = Path.Combine(installDirectory, "portable_config");
    var mac = Path.Combine(pc, "mpv-animejanai.conf");
    File.WriteAllText(mac, File.ReadAllText(mac)
        .Replace("lib=~~/../animejanai/inference/aji.dll", "lib=~~/../animejanai/inference/libaji.so")
        .Replace(":trtexec=~~/../animejanai/inference/trtexec.exe", "")
        .Replace("hwdec=nvdec", "hwdec=no")
        .Replace("gpu-api=vulkan,auto", "gpu-api=auto")
        // vulkan-queue-count is a Vulkan-VO option; the upscaling is on ncnn-Vulkan
        // and the VO renders via gpu-next/libplacebo, so comment it out (it errors on
        // an mpv built without the legacy Vulkan VO). Also turn off mpv's built-in OSC
        // and OSD bar here (global scope, above the profile sections): stock mpv has no
        // mpv.net UI, so the bundled uosc (scripts/uosc) draws the control bar instead,
        // and a live built-in OSC would double up with it.
        .Replace("vulkan-queue-count=3",
            "#vulkan-queue-count=3\nosc=no\nosd-bar=no"));
    // Port the mpv.net menu to stock mpv. mpv.net's `script-message-to mpvnet ...`
    // commands are dead without mpv.net, so remap the load-bearing ones to uosc (the
    // bundled UI script, which builds its right-click menu from the #menu: annotations
    // below) and to native mpv commands. uosc's control bar covers the rest; any
    // remaining mpvnet command is a harmless no-op (mpv just logs "no client mpvnet").
    // Order matters: longer patterns first, so a prefix match can't eat a longer line
    // (e.g. "playlist-add  1" is a prefix of "playlist-add  10").
    var inp = Path.Combine(pc, "input-animejanai.conf");
    File.WriteAllText(inp, File.ReadAllText(inp)
        // Ctrl+E "Launch Manager" -> the Linux ConfEditor binary (forward slashes, no .exe).
        // The source line has DOUBLE backslashes ("~~\\..\\AnimeJaNaiManager.exe", mpv.net's
        // escaped Windows path), so the C# pattern needs four backslashes to match two; a
        // single-backslash pattern silently no-ops and ships the broken Windows path.
        .Replace("~~\\\\..\\\\AnimeJaNaiManager.exe", "~~/../AnimeJaNaiManager")
        .Replace("~~\\..\\AnimeJaNaiManager.exe",     "~~/../AnimeJaNaiManager")
        .Replace("~~/../AnimeJaNaiManager.exe",       "~~/../AnimeJaNaiManager")
        // right-click menu -> uosc's menu (built from #menu: items); command palette
        // (F1) -> uosc/keybinds (its searchable command list), not a 2nd copy of the menu
        .Replace("script-message-to mpvnet show-menu",            "script-binding uosc/menu")
        .Replace("script-message-to mpvnet show-command-palette", "script-binding uosc/keybinds")
        // open files / external audio+subtitle loaders -> uosc's dedicated bindings
        // (uosc/load-audio and uosc/load-subtitles open a file browser; the track
        // SELECTORS below stay uosc/audio + uosc/subtitles)
        .Replace("script-message-to mpvnet open-files append",    "script-binding uosc/open-file")
        .Replace("script-message-to mpvnet open-files",           "script-binding uosc/open-file")
        .Replace("script-message-to mpvnet load-audio",           "script-binding uosc/load-audio")
        .Replace("script-message-to mpvnet load-sub",             "script-binding uosc/load-subtitles")
        .Replace("script-message-to mpvnet show-playlist",        "script-binding uosc/playlist")
        .Replace("script-message-to mpvnet show-audio-tracks",    "script-binding uosc/audio")
        .Replace("script-message-to mpvnet show-subtitle-tracks", "script-binding uosc/subtitles")
        .Replace("script-message-to mpvnet show-chapters",        "script-binding uosc/chapters")
        // file/media info -> the stats overlay
        .Replace("script-message-to mpvnet show-media-info osd",  "script-binding stats/display-stats-toggle")
        .Replace("script-message-to mpvnet show-media-info",      "script-binding stats/display-stats-toggle")
        .Replace("script-message-to mpvnet show-info",            "script-binding stats/display-stats-toggle")
        // playback + playlist navigation -> native commands
        .Replace("script-message-to mpvnet play-pause",           "cycle pause")
        .Replace("script-message-to mpvnet playlist-add -10",     "playlist-prev")
        .Replace("script-message-to mpvnet playlist-add  10",     "playlist-next")
        .Replace("script-message-to mpvnet playlist-add -1",      "playlist-prev")
        .Replace("script-message-to mpvnet playlist-add  1",      "playlist-next")
        .Replace("script-message-to mpvnet playlist-first",       "playlist-play-index 0")
        .Replace("script-message-to mpvnet playlist-last",        "playlist-play-index -1")
        .Replace("script-message-to mpvnet cycle-audio",          "cycle audio")
        .Replace("script-message-to mpvnet show-progress",        "show-progress")
        // external links -> the platform opener
        .Replace("script-message-to mpvnet shell-execute ",       "run xdg-open ")
        // osc=no killed the built-in OSC, so "Toggle OSC Visibility" -> uosc's UI toggle
        .Replace("script-binding osc/visibility",                 "script-binding uosc/toggle-ui")
        // no Linux auto-updater: make "AnimeJaNai > Install Update" a no-op instead of
        // firing a dead script-message (the animejanai_update.lua script is removed below)
        .Replace("Ctrl+u           script-message animejanai-update #menu: AnimeJaNai > Install Update",
                 "#Ctrl+u          script-message animejanai-update (no Linux updater)")
        // AnimeJaNai upscale slots, mouse-reachable from the menu (the !/@/SHARP/Ctrl+N
        // keys still switch slots; uosc nests these under AnimeJaNai > Upscale by path)
        + "\n\n# Linux: expose the upscale slots in the uosc menu (menu-only entries)\n"
        + "_  apply-profile upscale-on; show-text \"AnimeJaNai: Off\"; script-message aji-slot 0          #menu: AnimeJaNai > Upscale > Off\n"
        + "_  apply-profile upscale-on; show-text \"AnimeJaNai: Quality\"; script-message aji-slot 1001    #menu: AnimeJaNai > Upscale > Quality\n"
        + "_  apply-profile upscale-on; show-text \"AnimeJaNai: Balanced\"; script-message aji-slot 1002   #menu: AnimeJaNai > Upscale > Balanced\n"
        + "_  apply-profile upscale-on; show-text \"AnimeJaNai: Performance\"; script-message aji-slot 1003 #menu: AnimeJaNai > Upscale > Performance\n");
    // the auto-updater is Windows-only (no Linux AnimeJaNaiUpdater build); drop its
    // script so it doesn't error a failed subprocess on every launch.
    var updScript = Path.Combine(pc, "scripts", "animejanai_update.lua");
    if (File.Exists(updScript)) File.Delete(updScript);

    GenerateInputConf();

    // 5b. uosc: the player UI (control bar + context menu) for stock mpv, which has no
    // mpv.net interface. Reads the #menu: annotations from the input.conf written above.
    await InstallUoscLinux(pc);

    // 6. portable launcher (standalone mpv needs an explicit --config-dir)
    var launcher = Path.Combine(installDirectory, "run-animejanai");
    File.WriteAllText(launcher,
        "#!/bin/sh\nDIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\nexec \"$DIR/mpv\" --config-dir=\"$DIR/portable_config\" \"$@\"\n");
    SetExec(launcher);

    // 7. the ConfEditor / Manager GUI (published linux-x64 Avalonia binary + its native
    // libs) at the package root, where Ctrl+E (~~/../AnimeJaNaiManager) launches it and
    // it reads animejanai/animejanai.conf relative to itself.
    string confDir = Environment.GetEnvironmentVariable("AJI_CONFEDITOR_DIR") ?? "/tmp/confeditor-linux";
    if (File.Exists(Path.Combine(confDir, "AnimeJaNaiManager")))
    {
        foreach (var f in new[] { "AnimeJaNaiManager", "libHarfBuzzSharp.so", "libSkiaSharp.so" })
        {
            var src = Path.Combine(confDir, f);
            if (File.Exists(src)) File.Copy(src, Path.Combine(installDirectory, f), true);
        }
        SetExec(Path.Combine(installDirectory, "AnimeJaNaiManager"));
    }
    else
    {
        Console.WriteLine($"  (ConfEditor not found at {confDir}; package built without the Manager GUI)");
    }

    File.WriteAllText(Path.Combine(installDirectory, "version.txt"), args[0]);
    Console.WriteLine($"Linux package assembled at {installDirectory}");
}

void RecreateSymlink(string link, string target)
{
    if (File.Exists(link) || Directory.Exists(link)) File.Delete(link);
    File.CreateSymbolicLink(link, target);
}

void SetExec(string path) => File.SetUnixFileMode(path,
    UnixFileMode.UserRead  | UnixFileMode.UserWrite  | UnixFileMode.UserExecute |
    UnixFileMode.GroupRead | UnixFileMode.GroupExecute |
    UnixFileMode.OtherRead | UnixFileMode.OtherExecute);

// Bundle uosc (github.com/tomasklaen/uosc) into the Linux package's portable_config.
// Stock mpv has no mpv.net UI; uosc gives the control bar + the context menu (built
// from input.conf's #menu: items). The release zip lays down scripts/uosc/ (a script
// directory mpv auto-loads) and the icon fonts at its root, so it extracts straight
// into portable_config. Sourced from AJI_UOSC_DIR if set (a pre-extracted release, used
// for offline/local builds), otherwise downloaded from the latest GitHub release.
async Task InstallUoscLinux(string portableConfig)
{
    var fromDir = Environment.GetEnvironmentVariable("AJI_UOSC_DIR");
    if (fromDir != null && File.Exists(Path.Combine(fromDir, "scripts", "uosc", "main.lua")))
    {
        Console.WriteLine($"Bundling uosc from {fromDir}...");
        CopyDirectory(Path.Combine(fromDir, "scripts", "uosc"),
                      Path.Combine(portableConfig, "scripts", "uosc"));
        CopyDirectory(Path.Combine(fromDir, "fonts"), Path.Combine(portableConfig, "fonts"));
    }
    else
    {
        Console.WriteLine("Downloading uosc...");
        var zip = Path.GetFullPath("uosc.zip");
        await DownloadFileAsync(
            "https://github.com/tomasklaen/uosc/releases/latest/download/uosc.zip",
            zip, (progress) => Console.WriteLine($"Downloading uosc ({progress}%)..."));
        ExtractZip(zip, portableConfig, _ => { });
        File.Delete(zip);
    }

    // uosc shells out to its bundled ziggy helper (file browser / Open Files, clipboard
    // paste, updater) by path. The release zip / a plain copy doesn't carry a +x bit and
    // mpv can't exec a non-executable file, so without this those uosc features fail with
    // "Calling ziggy failed". (The Windows ziggy.exe needs no exec bit.)
    var ziggy = Path.Combine(portableConfig, "scripts", "uosc", "bin", "ziggy-linux");
    if (File.Exists(ziggy)) SetExec(ziggy);
}

// The released package is the slim core: everything hardware-specific
// (TensorRT runtime, per-GPU kernel packs, RIFE models) ships only as
// component packs, installed on demand by the AnimeJaNai Manager (the
// first-run dialog offers the hardware-matched set in one click). This
// keeps the one download people grab small; it is still named
// "full-package" because it is the complete release - and the 3.3.x
// updater downloads that asset name blindly, after which upgraders get
// the same Manager first-run flow as new users.
void SlimInstallTree(List<string> packFiles)
{
    Console.WriteLine("Slimming install tree (component packs ship separately)...");
    long removed = 0;
    foreach (var rel in packFiles)
    {
        var abs = Path.Combine(installDirectory, rel);
        if (!File.Exists(abs))
        {
            continue;
        }
        removed += new FileInfo(abs).Length;
        File.Delete(abs);
    }
    // prune directories the packs emptied (e.g. animejanai/rife)
    foreach (var dir in Directory.GetDirectories(installDirectory, "*", SearchOption.AllDirectories)
                                 .OrderByDescending(d => d.Length))
    {
        if (!Directory.EnumerateFileSystemEntries(dir).Any())
        {
            Directory.Delete(dir);
        }
    }
    Console.WriteLine($"Slimmed {removed / 1048576} MB out of the package tree.");
}

// Component packs: subsets of the freshly built install tree, emitted as
// rooted 7z archives + a packs.json index. The AnimeJaNai Manager (the
// updater's component modes) downloads these from the release so a slim
// install can add - and an older full install can shed - the heavy,
// hardware-specific pieces: the TensorRT runtime, per-GPU-generation
// builder resources, and the RIFE models. Archive paths are relative to
// the install root, so extraction over an install IS installation.
// Returns the expanded per-file list of everything packed; under --packs
// the caller then strips those files from the tree (SlimInstallTree).
async Task<List<string>> EmitComponentPacks()
{
    Console.WriteLine("Emitting component packs...");
    var version = args[0];
    var packsDir = Path.Combine(assemblyDirectory, $"packs-v{version}");
    Directory.CreateDirectory(packsDir);
    var sevenZa = Path.Combine(installDirectory, "7za.exe");
    if (!File.Exists(sevenZa))
    {
        sevenZa = Path.Combine(assemblyDirectory, "7za.exe"); // --packs-only on a partial tree
    }

    // Dep = the manifest.json deps key that governs a pack's content, emitted into packs.json so
    // the updater can skip re-downloading an already-present pack when that dep is unchanged across
    // releases (TensorRT runtime + builder resources are versioned by inference_runtime; RIFE by rife).
    var packs = new List<(string Name, string Dep, string[] Files)>
    {
        ("trt-runtime", "inference_runtime", Directory.GetFiles(inferencePath)
            .Where(f =>
            {
                var n = Path.GetFileName(f);
                return (n.StartsWith("nvinfer_") && !n.Contains("builder_resource")) ||
                       n.StartsWith("nvonnxparser_") || n.StartsWith("cudart64_") ||
                       n == "trtexec.exe" ||
                       n.Contains("LICENSE", StringComparison.OrdinalIgnoreCase);
            })
            .Select(f => Path.GetRelativePath(installDirectory, f)).ToArray()),
        ("rife", "rife", new[] { Path.GetRelativePath(installDirectory, rifePath) }),
    };
    foreach (var f in Directory.GetFiles(inferencePath, "nvinfer_builder_resource_*"))
    {
        // nvinfer_builder_resource_sm120_11.dll -> trt-sm120
        var m = System.Text.RegularExpressions.Regex.Match(
            Path.GetFileName(f), @"builder_resource_([a-z0-9]+)_");
        if (m.Success)
        {
            packs.Add(($"trt-{m.Groups[1].Value}", "inference_runtime",
                       new[] { Path.GetRelativePath(installDirectory, f) }));
        }
    }

    var index = new List<object>();
    var packedFiles = new List<string>();
    foreach (var (name, dep, files) in packs)
    {
        if (files.Length == 0 ||
            !files.Any(f => File.Exists(Path.Combine(installDirectory, f)) ||
                            Directory.Exists(Path.Combine(installDirectory, f))))
        {
            Console.WriteLine($"  component-{name}: nothing to pack in this tree, skipped");
            continue;
        }
        var archive = Path.Combine(packsDir, $"component-{name}.7z");
        File.Delete(archive);
        // -spf2: store the relative paths as given (rooted at install dir)
        var fileArgs = string.Join(' ', files.Select(f => $"\"{f}\""));
        var psi = new ProcessStartInfo
        {
            FileName = sevenZa,
            Arguments = $"a -spf2 -mx=3 \"{archive}\" {fileArgs}",
            WorkingDirectory = installDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        using var proc = Process.Start(psi)!;
        await proc.WaitForExitAsync();
        if (proc.ExitCode != 0)
        {
            throw new InvalidOperationException($"7za failed for pack {name}");
        }
        // expand directories to the concrete file list for clean uninstall
        var allFiles = files.SelectMany(f =>
        {
            var abs = Path.Combine(installDirectory, f);
            return Directory.Exists(abs)
                ? Directory.GetFiles(abs, "*", SearchOption.AllDirectories)
                    .Select(x => Path.GetRelativePath(installDirectory, x))
                : new[] { f };
        }).Select(f => f.Replace('\\', '/')).ToArray();
        index.Add(new
        {
            name,
            dep,
            asset = Path.GetFileName(archive),
            bytes = new FileInfo(archive).Length,
            files = allFiles,
        });
        packedFiles.AddRange(allFiles);
        Console.WriteLine($"  component-{name}.7z ({new FileInfo(archive).Length / 1048576} MB, {allFiles.Length} files)");
    }
    File.WriteAllText(Path.Combine(packsDir, "packs.json"),
        JsonSerializer.Serialize(new { package_version = version, packs = index },
            new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"Packs written to {packsDir}");
    return packedFiles;
}

if (packsOnlyIndex >= 0)
{
    await EmitComponentPacks();
}
else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
{
    await MainLinux();
}
else
{
    await Main();
}
