#requires -Version 5.1
# Stop hook -> append session log entry to Obsidian Vault\Sessions\YYYY-MM-DD.md
# Receives Claude Code hook JSON on stdin.

$ErrorActionPreference = 'Stop'

try {
    $stdin = [Console]::In.ReadToEnd()
    $payload = if ($stdin) { try { $stdin | ConvertFrom-Json } catch { $null } } else { $null }

    $sessionsDir = 'C:\Users\ANAKIN\Documents\Obsidian Vault\Sessions'
    if (-not (Test-Path $sessionsDir)) { New-Item -ItemType Directory -Path $sessionsDir -Force | Out-Null }

    $now = Get-Date
    $logFile = Join-Path $sessionsDir ($now.ToString('yyyy-MM-dd') + '.md')

    $cwd = if ($payload.cwd) { $payload.cwd } elseif ($env:CLAUDE_PROJECT_DIR) { $env:CLAUDE_PROJECT_DIR } else { (Get-Location).Path }
    $sessionId = if ($payload.session_id) { $payload.session_id.Substring(0, [Math]::Min(8, $payload.session_id.Length)) } else { 'n/a' }

    $lastUser = ''; $lastAssistant = ''
    if ($payload.transcript_path -and (Test-Path $payload.transcript_path)) {
        try {
            $turns = Get-Content -LiteralPath $payload.transcript_path | Where-Object { $_ } | ForEach-Object { try { $_ | ConvertFrom-Json } catch { } }
            $userTurns = $turns | Where-Object { $_.type -eq 'user' -and $_.message.content }
            $asstTurns = $turns | Where-Object { $_.type -eq 'assistant' -and $_.message.content }
            if ($userTurns) {
                $u = $userTurns[-1].message.content
                $lastUser = if ($u -is [string]) { $u } else { ($u | Where-Object { $_.type -eq 'text' } | Select-Object -First 1 -ExpandProperty text -ErrorAction SilentlyContinue) }
            }
            if ($asstTurns) {
                $a = $asstTurns[-1].message.content
                $lastAssistant = if ($a -is [string]) { $a } else { ($a | Where-Object { $_.type -eq 'text' } | Select-Object -First 1 -ExpandProperty text -ErrorAction SilentlyContinue) }
            }
        } catch { }
    }

    $truncate = { param($s, $n) if ($null -eq $s) { '' } elseif ($s.Length -le $n) { $s } else { $s.Substring(0, $n) + '...' } }
    $userExcerpt = & $truncate $lastUser 400
    $asstExcerpt = & $truncate $lastAssistant 400

    if (-not (Test-Path $logFile)) {
        $header = "# Sessions Claude - $($now.ToString('yyyy-MM-dd'))`n`n"
        Set-Content -LiteralPath $logFile -Value $header -Encoding UTF8
    }

    $entry = "## $($now.ToString('HH:mm')) - $cwd`n`n" +
             "- **Session :** $sessionId`n"
    if ($userExcerpt) { $entry += "- **Dernier prompt :** $userExcerpt`n" }
    if ($asstExcerpt) { $entry += "- **Derniere reponse :** $asstExcerpt`n" }
    $entry += "`n---`n`n"

    Add-Content -LiteralPath $logFile -Value $entry -Encoding UTF8

    @{ systemMessage = "Session loggee : Sessions\$($now.ToString('yyyy-MM-dd')).md" } | ConvertTo-Json -Compress
} catch {
    @{ systemMessage = "Hook Obsidian erreur : $($_.Exception.Message)" } | ConvertTo-Json -Compress
}
