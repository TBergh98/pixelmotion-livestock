param(
    [string]$OpenCvVersion = "4.10.0",
    [string]$CudaArchBin = "8.9",
    [string]$BuildRoot = (Join-Path $PSScriptRoot "..\build\opencv-cuda")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

function Ensure-PythonPackages {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string[]]$PackageNames
    )

    Write-Host "Installing missing tools via pip: $($PackageNames -join ', ')"
    & $PythonExe -m pip install --upgrade @PackageNames
    if ($LASTEXITCODE -ne 0) {
        throw "pip bootstrap failed while installing: $($PackageNames -join ', ')"
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $false)]
        [string[]]$ArgumentList = @()
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-PythonValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string]$Code
    )

    $result = & $PythonExe -c $Code
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed while evaluating: $Code"
    }

    return ($result | Select-Object -First 1).ToString().Trim()
}

Assert-Command -Name "git"
Assert-Command -Name "python"

$missingTools = @()
if (-not (Get-Command "cmake" -ErrorAction SilentlyContinue)) {
    $missingTools += "cmake"
}

$ninjaCommand = Get-Command "ninja" -ErrorAction SilentlyContinue
if (-not $ninjaCommand) {
    $missingTools += "ninja"
}

if ($missingTools.Count -gt 0) {
    Ensure-PythonPackages -PythonExe (Get-Command python).Path -PackageNames $missingTools
}

Assert-Command -Name "cmake"
$ninjaCommand = Get-Command "ninja" -ErrorAction SilentlyContinue
if (-not $ninjaCommand) {
    throw "ninja was still not found after the pip bootstrap."
}

$nvccCommand = Get-Command "nvcc" -ErrorAction SilentlyContinue
if (-not $nvccCommand) {
    throw "nvcc was not found on PATH. Install the NVIDIA CUDA Toolkit and make sure nvcc is available before building OpenCV with CUDA."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([System.IO.Path]::IsPathRooted($BuildRoot)) {
    $buildRootPath = [System.IO.Path]::GetFullPath($BuildRoot)
}
else {
    $buildRootPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildRoot))
}
$sourceRoot = Join-Path $buildRootPath "source"
$buildDir = Join-Path $buildRootPath "build"
$installDir = Join-Path $buildRootPath "install"
$opencvSrc = Join-Path $sourceRoot "opencv"
$opencvContribSrc = Join-Path $sourceRoot "opencv_contrib"

New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

$pythonExe = (Get-Command python).Path
$pythonSitePackages = Get-PythonValue -PythonExe $pythonExe -Code "import site; print(site.getsitepackages()[0])"
$pythonInclude = Get-PythonValue -PythonExe $pythonExe -Code "import sysconfig; print(sysconfig.get_paths()['include'])"
$numpyInclude = Get-PythonValue -PythonExe $pythonExe -Code "import numpy; print(numpy.get_include())"

if (-not (Test-Path $opencvSrc)) {
    Invoke-Checked -FilePath "git" -ArgumentList @(
        "clone",
        "--branch", $OpenCvVersion,
        "--depth", "1",
        "https://github.com/opencv/opencv.git",
        $opencvSrc
    )
}

if (-not (Test-Path $opencvContribSrc)) {
    Invoke-Checked -FilePath "git" -ArgumentList @(
        "clone",
        "--branch", $OpenCvVersion,
        "--depth", "1",
        "https://github.com/opencv/opencv_contrib.git",
        $opencvContribSrc
    )
}

Write-Host "Using Python: $pythonExe"
Write-Host "Python site-packages: $pythonSitePackages"
Write-Host "CUDA arch bin: $CudaArchBin"
Write-Host "Install path: $installDir"

$cmakeArgs = @(
    "-S", $opencvSrc,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$installDir",
    "-DOPENCV_EXTRA_MODULES_PATH=$opencvContribSrc\modules",
    "-DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,python3,cudaarithm,cudaimgproc,cudawarping,cudaobjdetect,cudabgsegm,cudaoptflow",
    "-DWITH_CUDA=ON",
    "-DWITH_CUDNN=OFF",
    "-DOPENCV_DNN_CUDA=ON",
    "-DCUDA_ARCH_BIN=$CudaArchBin",
    "-DCUDA_FAST_MATH=ON",
    "-DWITH_OPENCL=OFF",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_DOCS=OFF",
    "-DBUILD_opencv_java=OFF",
    "-DBUILD_opencv_python2=OFF",
    "-DBUILD_opencv_python3=ON",
    "-DPYTHON3_EXECUTABLE=$pythonExe",
    "-DPYTHON3_INCLUDE_DIR=$pythonInclude",
    "-DPYTHON3_NUMPY_INCLUDE_DIRS=$numpyInclude",
    "-DOPENCV_PYTHON3_INSTALL_PATH=$pythonSitePackages"
)

Invoke-Checked -FilePath "cmake" -ArgumentList $cmakeArgs
Invoke-Checked -FilePath "cmake" -ArgumentList @("--build", $buildDir, "--config", "Release", "--target", "install")

Write-Host "Verifying CUDA-enabled OpenCV..."
Invoke-Checked -FilePath $pythonExe -ArgumentList @(
    "-c",
    "import cv2; info = cv2.getBuildInformation(); print('cv2 file:', cv2.__file__); print('cv2 version:', cv2.__version__); print('CUDA present in build:', 'CUDA' in info); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0)"
)

Write-Host "OpenCV CUDA build completed. Re-open the terminal, activate the environment, and run the pipeline again."
