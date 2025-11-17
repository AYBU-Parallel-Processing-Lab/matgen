<#
.SYNOPSIS
  Configures the Visual Studio x64 (amd64) environment and then runs CMake.
  This is the recommended, modern way to do this in PowerShell.
#>

param (
  [string]$CMakeArgs = ".." # Example: "-S . -B build"
)

Write-Host "[INFO] Finding Visual Studio installation..."

# Try to find vswhere.exe
$vswherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswherePath)) {
  Write-Error "vswhere.exe not found at $vswherePath."
  Write-Error "Please ensure Visual Studio 2017 or newer is installed."
  exit 1
}

# Find the latest VS installation path
$vsInstallPath = & $vswherePath -latest -property installationPath
if (-not $vsInstallPath) {
  Write-Error "Could not find a Visual Studio installation."
  exit 1
}

Write-Host "[INFO] Found Visual Studio at: $vsInstallPath"

# Path to the Developer Shell module
$devShellModule = Join-Path $vsInstallPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
if (-not (Test-Path $devShellModule)) {
  Write-Error "Could not find the VS Developer Shell module at $devShellModule."
  exit 1
}

# --- 1. Import the VS Environment ---
Write-Host "[INFO] Importing VS amd64 environment..."
Import-Module $devShellModule

Enter-VsDevShell -VsInstallPath $vsInstallPath -Arch 'amd64' -SkipAutomaticLocation

if ($LASTEXITCODE -ne 0) {
  Write-Error "Failed to initialize the Visual Studio environment."
  exit 1
}

Write-Host "[INFO] Environment ready. Running CMake..."

# --- 2. Run CMake ---
# We use Invoke-Expression to ensure CMakeArgs are parsed correctly
try {
  Invoke-Expression "cmake $CMakeArgs"
  Write-Host "[INFO] CMake configuration finished."
}
catch {
  Write-Error "CMake failed to run."
  Write-Error $_
  exit 1
}
