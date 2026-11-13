; Inno Setup script for dom. Built by scripts/build_release.py, which passes:
;   /DAppVersion=0.2.0 /DSourceDir=<PyInstaller output folder> /DArch=x64|arm64 /DOutputDir=<dir>
; Installs per user (no administrator rights needed). Chats and models live in
; %LOCALAPPDATA%\dom and are kept on uninstall.

#ifndef AppVersion
  #error AppVersion must be defined
#endif

[Setup]
AppId={{6F0B2E47-5C7A-4D0B-9C5E-2D3F8B1E4A10}
AppName=dom
AppVersion={#AppVersion}
AppPublisher=dom
DefaultDirName={localappdata}\Programs\dom
DefaultGroupName=dom
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#OutputDir}
OutputBaseFilename=dom-{#AppVersion}-windows-{#Arch}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\dom.exe
#if Arch == "arm64"
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
#else
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
#endif

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\dom"; Filename: "{app}\dom.exe"
Name: "{autodesktop}\dom"; Filename: "{app}\dom.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\dom.exe"; Description: "Start dom"; Flags: nowait postinstall skipifsilent
