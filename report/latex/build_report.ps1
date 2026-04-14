$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$python = Join-Path $repo ".venv\\Scripts\\python.exe"
if (-not (Test-Path $python)) {
  $python = "python"
}

Push-Location $repo
try {
  & $python -u "scripts\\build_latex_report_assets.py"
} finally {
  Pop-Location
}

Push-Location $PSScriptRoot
try {
  pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex
  pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex
} finally {
  Pop-Location
}
