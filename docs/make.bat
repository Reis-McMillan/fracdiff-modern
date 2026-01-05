@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
if "%SPHINXAUTOBUILD%" == "" (
    set SPHINXAUTOBUILD=sphinx-autobuild
)

set SOURCEDIR=source
set BUILDDIR=build
set SPHINXPROJ=Fracdiff
REM Added -WT and --keep-going to match the Makefile
set SPHINXOPTS=-j auto -WT --keep-going

if "%1" == "" goto help
if "%1" == "livehtml" goto livehtml

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
    echo.installed.
    exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:livehtml
%SPHINXAUTOBUILD% %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd