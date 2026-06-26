@echo off
rem ============================================================================
rem  scripts\build_nvcc.bat   (classic Windows cmd.exe batch)
rem  ---------------------------------------------------------------------------
rem  Same idea as build_nvcc.ps1, for people who prefer a plain .bat. It calls
rem  nvcc directly to build the demo and the verifier into the build\ folder.
rem
rem  USAGE (from the project root, in a terminal where nvcc is on PATH):
rem      scripts\build_nvcc.bat            (defaults to sm_75)
rem      scripts\build_nvcc.bat 86         (target a different architecture)
rem
rem  PREREQUISITE: nvcc on PATH. Verify with:  nvcc --version
rem ============================================================================

setlocal

rem -- First argument is the GPU architecture number; default to 75 (Turing). --
set ARCH=%1
if "%ARCH%"=="" set ARCH=75

rem -- %~dp0 is the folder this .bat lives in (scripts\, with a trailing "\").  --
rem -- ".." is the project root. The for-loop with %%~fI normalizes the path,   --
rem -- turning "...\scripts\.." into a clean absolute root so the paths we echo  --
rem -- (and any nvcc error messages) do not contain a confusing "scripts\.."     --
rem -- segment. We use absolute paths so the current directory does not matter.  --
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set INCLUDE=%ROOT%\include
set SRC=%ROOT%\src
set BUILD=%ROOT%\build

if not exist "%BUILD%" mkdir "%BUILD%"

rem -- The shared core sources, reused by both executables. --
set CORE="%SRC%\alignment_common.cpp" "%SRC%\cpu_reference.cpp" "%SRC%\gpu_wavefront.cu" "%SRC%\gpu_batched.cu"

echo Building for GPU architecture sm_%ARCH% ...

echo   -^> build\demo.exe
nvcc -O2 -std=c++14 -arch=sm_%ARCH% -I"%INCLUDE%" "%ROOT%\demo\demo_main.cpp" %CORE% -o "%BUILD%\demo.exe"
if errorlevel 1 goto :error

echo   -^> build\verify.exe
nvcc -O2 -std=c++14 -arch=sm_%ARCH% -I"%INCLUDE%" "%ROOT%\tests\verify_main.cpp" %CORE% -o "%BUILD%\verify.exe"
if errorlevel 1 goto :error

echo Done. Run:
echo     build\demo.exe
echo     build\verify.exe
endlocal
exit /b 0

:error
echo BUILD FAILED.
endlocal
exit /b 1
