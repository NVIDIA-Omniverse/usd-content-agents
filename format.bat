@echo off
REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
setlocal EnableExtensions EnableDelayedExpansion

REM format.bat - Windows-native formatter/linter runner (no bash/sh)
REM Mirrors the behavior of ./format.sh:
REM   - Ruff lint (check or fix)
REM   - Ruff format (check or format)
REM   - Optional mypy (via --mypy)
REM   - Custom checks: trailing whitespace (world_understanding/tests), print() in world_understanding
REM
REM Usage:
REM   format.bat          -> fix mode
REM   format.bat fix      -> fix mode
REM   format.bat check    -> check mode
REM   format.bat --mypy
REM   format.bat check --mypy

cd /d "%~dp0"

set "MODE=fix"
set "RUN_MYPY=0"

for %%A in (%*) do (
  if /I "%%~A"=="check" set "MODE=check"
  if /I "%%~A"=="fix" set "MODE=fix"
  if /I "%%~A"=="--mypy" set "RUN_MYPY=1"
)

echo World Understanding Code Formatter
echo Mode: %MODE%
echo.

REM Activate venv if available (recommended by project rules)
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo WARNING: .venv not found. Continuing without a virtual environment.
  echo.
)

REM Helper: install package(s) using uv if available, else pip
set "INSTALL_CMD="
where uv >NUL 2>&1
if %ERRORLEVEL% EQU 0 (
  set "INSTALL_CMD=uv pip install -q"
) else (
  set "INSTALL_CMD=python -m pip install -q"
)

REM Ensure ruff exists
python -m ruff --version >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Installing missing tool: ruff
  %INSTALL_CMD% ruff
)

REM Ensure mypy exists if requested
if "%RUN_MYPY%"=="1" (
  python -m mypy --version >NUL 2>&1
  if %ERRORLEVEL% NEQ 0 (
    echo Installing missing tool: mypy
    %INSTALL_CMD% mypy
  )
)

set "FAILED=0"

echo === Running Ruff Linter ===
if /I "%MODE%"=="check" (
  python -m ruff check .
  if %ERRORLEVEL% NEQ 0 set "FAILED=1"
) else (
  python -m ruff check --fix --unsafe-fixes .
  REM In fix mode we warn but don't hard fail if some issues remain
)
echo.

echo === Running Ruff Formatter ===
if /I "%MODE%"=="check" (
  python -m ruff format --check .
  if %ERRORLEVEL% NEQ 0 set "FAILED=1"
) else (
  python -m ruff format .
  if %ERRORLEVEL% NEQ 0 set "FAILED=1"
)
echo.

if "%RUN_MYPY%"=="1" (
  echo === Running MyPy Type Checker ===
  python -m mypy world_understanding
  if %ERRORLEVEL% NEQ 0 (
    if /I "%MODE%"=="check" set "FAILED=1"
    echo NOTE: Type errors must be fixed manually.
  )
  echo.
)

echo === Running Custom Checks ===

REM Trailing whitespace in world_understanding/ and tests/
echo Checking for trailing whitespace...
powershell -NoProfile -Command ^
  "$paths=@('world_understanding','tests');" ^
  "$files=Get-ChildItem -Recurse -File -Filter '*.py' -Path $paths -ErrorAction SilentlyContinue;" ^
  "$hits=@(); foreach($f in $files){ $m=Select-String -Path $f.FullName -Pattern '[ \t]+$' -AllMatches -ErrorAction SilentlyContinue; if($m){ $hits += $m } }" ^
  "if($hits.Count -gt 0){ $hits | ForEach-Object { Write-Host ($_.Path + ':' + $_.LineNumber + ':' + $_.Line) }; exit 1 } else { exit 0 }"
if %ERRORLEVEL% NEQ 0 (
  echo WARNING: Found trailing whitespace (see above)
  if /I "%MODE%"=="check" (
    set "FAILED=1"
  ) else (
    echo Removing trailing whitespace...
    powershell -NoProfile -Command ^
      "$paths=@('world_understanding','tests');" ^
      "$files=Get-ChildItem -Recurse -File -Filter '*.py' -Path $paths -ErrorAction SilentlyContinue;" ^
      "foreach($f in $files){ $txt=Get-Content -Raw -LiteralPath $f.FullName; $new=$txt -replace '[ \t]+(?=\r?\n)','' -replace '[ \t]+$',''; if($new -ne $txt){ Set-Content -NoNewline -LiteralPath $f.FullName -Value $new } }"
  )
) else (
  echo OK: No trailing whitespace found
)
echo.

REM print(...) in world_understanding (warn only), ignoring lines with '# noqa'
echo Checking for print statements...
powershell -NoProfile -Command ^
  "$files=Get-ChildItem -Recurse -File -Filter '*.py' -Path 'world_understanding' -ErrorAction SilentlyContinue;" ^
  "$hits=@(); foreach($f in $files){ $m=Select-String -Path $f.FullName -Pattern '^[ \t]*print\(' -AllMatches -ErrorAction SilentlyContinue; if($m){ $hits += ($m | Where-Object { $_.Line -notmatch '#\s*noqa' }) } }" ^
  "if($hits.Count -gt 0){ $hits | ForEach-Object { Write-Host ($_.Path + ':' + $_.LineNumber + ':' + $_.Line) }; exit 0 } else { exit 0 }"
echo.

if "%FAILED%"=="0" (
  echo All checks passed.
  exit /b 0
) else (
  echo Some checks failed.
  if /I "%MODE%"=="check" echo Tip: run "format.bat fix" to auto-fix issues.
  exit /b 1
)

