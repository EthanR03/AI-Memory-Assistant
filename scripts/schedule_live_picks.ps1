<#
Register (or remove) the Windows scheduled task that runs
scripts\run_live_picks.py every hour through the 2026 regular season.

    powershell -ExecutionPolicy Bypass -File scripts\schedule_live_picks.ps1
    powershell -ExecutionPolicy Bypass -File scripts\schedule_live_picks.ps1 -Remove

The task runs as you and only while you are logged on, so git can push
with your stored credentials. It wakes the machine from sleep, runs on
battery, and catches up as soon as it can after a missed hour. It expires
after week 18. Re-running this replaces the task.
#>
param([switch]$Remove)

$TaskName = "NFL wind picks"
$Root = Split-Path -Parent $PSScriptRoot

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed '$TaskName'."
    return
}

$action = New-ScheduledTaskAction `
    -Execute (Join-Path $Root "venv\Scripts\pythonw.exe") `
    -Argument "scripts\run_live_picks.py" `
    -WorkingDirectory $Root

# Hourly at five past, from the next hour until two days after week 18.
$start = (Get-Date).Date.AddHours((Get-Date).Hour + 1).AddMinutes(5)
$trigger = New-ScheduledTaskTrigger -Once -At $start `
    -RepetitionInterval (New-TimeSpan -Hours 1)
$trigger.EndBoundary = "2027-01-05T00:00:00"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable -WakeToRun `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action `
    -Trigger $trigger -Settings $settings -Principal $principal -Force `
    -Description ("Logs wind-rule picks about 26h before each outdoor " +
                  "NFL kickoff, then commits and pushes " +
                  "data/live/wind_picks.csv. See src/nfl/WIND_RULE.md.") |
    Out-Null

Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo |
    Select-Object TaskName, NextRunTime
