<#
.SYNOPSIS
Selects a local Visual Studio/CMake pair and runs a Windows development command.
.PARAMETER Action
configure generates the build; fetch-deps builds dependencies; build builds the
application; run builds and launches it; nocompile launches an existing executable
without configuring or building; clean removes compiled build outputs.
.PARAMETER Configuration
RelWithDebInfo (optimized with PDBs) or Debug. Defaults to RelWithDebInfo.
.PARAMETER ProgramArguments
Arguments forwarded for run or nocompile, including an optional data directory.
.NOTES
Build actions require VS 2022 or 2026 with Desktop development with C++ and
CMake >= 3.25. nocompile only requires an existing executable and its runtime DLLs.
Throws on missing tools, invalid arguments, or failed child processes. Downloads
stay in thirdparty; configuration-specific builds and installations stay in build.
#>
[CmdletBinding()]
param(
    [ValidateSet('configure', 'fetch-deps', 'build', 'run', 'nocompile', 'clean')]
    [string] $Action = 'configure',
    [ValidateSet('Debug', 'RelWithDebInfo')]
    [string] $Configuration = 'RelWithDebInfo',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ProgramArguments = @()
)

$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($ProgramArguments.Count -gt 0 -and $Action -notin @('run', 'nocompile')) {
    throw 'Program arguments are accepted only with -Action run or nocompile.'
}

# Launch directly from an existing build, preferring the same toolchain order as
# the build driver. Do not probe tools, configure CMake, or download dependencies.
if ($Action -eq 'nocompile') {
    $executable = $null
    foreach ($preset in @('windows-vs2026', 'windows-vs2022')) {
        $candidate = Join-Path $repository "build/$preset/$Configuration/bin/$Configuration/tensor_compression.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $executable = $candidate
            break
        }
    }
    if (-not $executable) {
        throw "No existing $Configuration executable found. Run just build $Configuration first."
    }
    Push-Location $repository
    try {
        & $executable @ProgramArguments
        if ($LASTEXITCODE -ne 0) {
            throw "tensor_compression exited with code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }
    return
}

# Match a C++ workload with a CMake that supports its generator. Prefer the
# bundled CMake over unrelated MinGW tools that may appear earlier on PATH.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Install Visual Studio 2022 or 2026 with Desktop development with C++.'
}
$toolchain = $null
foreach ($spec in @(
    @{ Preset = 'windows-vs2026'; Generator = 'Visual Studio 18 2026'; Range = '[18.0,19.0)' },
    @{ Preset = 'windows-vs2022'; Generator = 'Visual Studio 17 2022'; Range = '[17.0,18.0)' }
)) {
    $installation = & $vswhere -latest -products '*' -version $spec.Range `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath |
        Select-Object -First 1
    if (-not $installation) {
        continue
    }
    $bundled = Join-Path $installation 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
    $candidates = @($bundled) + @(Get-Command cmake.exe -All -ErrorAction SilentlyContinue | ForEach-Object Source)
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate)) {
            continue
        }
        $capabilities = & $candidate -E capabilities | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        $version = [version]::new($capabilities.version.major, $capabilities.version.minor, $capabilities.version.patch)
        if ($version -lt [version]'3.25.0' -or $capabilities.generators.name -notcontains $spec.Generator) {
            continue
        }
        $toolchain = [PSCustomObject]@{ CMake = $candidate; Preset = $spec.Preset }
        break
    }
    if ($toolchain) {
        break
    }
}
if (-not $toolchain) {
    throw 'No compatible VS 2022/2026 C++ workload and CMake >= 3.25 were found.'
}

# MSBuild rejects environments containing duplicate PATH/Path keys in some hosts.
$normalizedPath = $env:Path
Remove-Item Env:PATH -ErrorAction SilentlyContinue
$env:Path = $normalizedPath
$buildDirectory = Join-Path $repository "build/$($toolchain.Preset)/$Configuration"
Push-Location $repository
try {
    if ($Action -eq 'configure' -or -not (Test-Path -LiteralPath "$buildDirectory/CMakeCache.txt")) {
        & $toolchain.CMake --preset $toolchain.Preset -B $buildDirectory "-DCMAKE_BUILD_TYPE=$Configuration"
        if ($LASTEXITCODE -ne 0) {
            throw 'CMake configuration failed; see the diagnostic above.'
        }
    }
    if ($Action -eq 'configure') {
        return
    }
    if ($Action -eq 'clean') {
        # ExternalProject owns independent build systems. Clean the application's
        # actual build tree while preserving the already-built numerical libraries.
        $applicationBuild = Join-Path $buildDirectory 'application'
        if (Test-Path -LiteralPath "$applicationBuild/CMakeCache.txt") {
            & $toolchain.CMake --build $applicationBuild --config $Configuration --target clean
            if ($LASTEXITCODE -ne 0) {
                throw "Cleaning $Configuration failed."
            }
        }
        return
    }
    $buildArguments = @('--build', $buildDirectory, '--config', $Configuration, '--parallel')
    if ($Action -eq 'fetch-deps') {
        $buildArguments += @('--target', 'dependencies')
    }
    & $toolchain.CMake @buildArguments
    if ($LASTEXITCODE -ne 0) {
        throw "CMake action '$Action' failed for $Configuration."
    }
    if ($Action -eq 'run') {
        & "$buildDirectory/bin/$Configuration/tensor_compression.exe" @ProgramArguments
        if ($LASTEXITCODE -ne 0) {
            throw "tensor_compression exited with code $LASTEXITCODE."
        }
    }
} finally {
    Pop-Location
}
