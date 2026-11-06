param([string]$Message = "")

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "Git failed (exit $LASTEXITCODE). Stopped without reset or force-push." }
}

$remoteUrl = & git remote get-url personal
if ($LASTEXITCODE -ne 0 -or $remoteUrl -ne 'https://github.com/thuhang012/cnn-vit-interpretability-personal.git') {
    throw 'The personal remote does not match the configured personal repository.'
}
$branch = & git branch --show-current
if ($LASTEXITCODE -ne 0 -or $branch -ne 'feat/gradcam') {
    throw 'Switch to feat/gradcam before syncing to the Colab notebooks.'
}

# Includes new files and intentional deletions; .gitignore remains in effect.
Invoke-Git add -A -- .
& git diff --cached --quiet
$diffExit = $LASTEXITCODE
if ($diffExit -eq 1) {
    Invoke-Git diff --cached --stat
    if (-not $Message) { $Message = 'Experiment snapshot ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') }
    Invoke-Git commit -m $Message
} elseif ($diffExit -ne 0) {
    throw 'Could not inspect staged changes.'
}
Invoke-Git push personal HEAD:refs/heads/feat/gradcam
Write-Host 'Synced. On Colab: stop the current run, rerun setup, then rerun the experiment cells.'
