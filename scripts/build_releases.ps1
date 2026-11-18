param(
    [string]$Version = "1.0.3",
    [string]$Configuration = "Release",
    [string[]]$Rids = @("win-x64", "win-arm64"),
    [switch]$SkipSelfTest
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectPath = Join-Path $RepoRoot "VoicierVoice.csproj"
$DistRoot = Join-Path $RepoRoot "dist"
$CleanTag = $Version.TrimStart("v")

if (Test-Path $DistRoot) {
    Remove-Item -Path $DistRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $DistRoot | Out-Null

$ZipPaths = @()

foreach ($Rid in $Rids) {
    $StageName = "VoicierVoice-v$CleanTag-$Rid"
    $StageDir = Join-Path $DistRoot $StageName
    $ZipPath = Join-Path $DistRoot "$StageName.zip"

    Write-Host "=== Publishing $StageName (self-contained, single-file) ==="
    dotnet publish $ProjectPath `
        -c $Configuration `
        -r $Rid `
        --self-contained true `
        -p:Version=$CleanTag `
        -p:PublishSingleFile=true `
        -p:IncludeNativeLibrariesForSelfExtract=true `
        -p:EnableCompressionInSingleFile=true `
        -p:DebugType=none `
        -p:DebugSymbols=false `
        -o $StageDir

    if ($LASTEXITCODE -ne 0) {
        throw "dotnet publish failed for $Rid with exit code $LASTEXITCODE"
    }

    # Remove static import .lib files copied by ONNX Runtime native targets; not needed at runtime.
    Get-ChildItem -Path $StageDir -Filter "*.lib" -File | Remove-Item -Force

    if (-not $SkipSelfTest -and $Rid -eq "win-x64") {
        Write-Host "=== Running self-test on published $StageName ==="
        $ExePath = Join-Path $StageDir "VoicierVoice.exe"
        $Proc = Start-Process -FilePath $ExePath -ArgumentList "--self-test" -Wait -PassThru -NoNewWindow
        if ($Proc.ExitCode -ne 0) {
            throw "Self-test failed for $StageName with exit code $($Proc.ExitCode)"
        }
    }

    if (Test-Path $ZipPath) {
        Remove-Item -Path $ZipPath -Force
    }
    Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ZipPath -CompressionLevel Optimal
    $ZipPaths += $ZipPath
}

$ChecksumFile = Join-Path $DistRoot "SHA256SUMS.txt"
$ChecksumLines = foreach ($Zip in $ZipPaths) {
    $Hash = (Get-FileHash -Path $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash  $(Split-Path $Zip -Leaf)"
}
$ChecksumLines | Set-Content -Path $ChecksumFile -Encoding UTF8

Write-Host "=== Release artifacts ready in $DistRoot ==="
Get-ChildItem -Path $DistRoot -File | Select-Object Name, @{Name="SizeMB";Expression={[math]::Round($_.Length / 1MB, 2)}} | Format-Table -AutoSize
