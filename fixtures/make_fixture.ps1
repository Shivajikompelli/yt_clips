param(
    [string]$OutFile = "$PSScriptRoot\narration.wav"
)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.SetOutputToWaveFile($OutFile)

$text = @(
    "Here is a question almost nobody asks. Why do most side projects die in the first month? "
    "The answer is brutally simple. People optimize the wrong thing from day one. "
    "They pick frameworks, they pick colors, they argue about tabs versus spaces. "
    "But they never talk to a single user. Let me give you a number. "
    "In my experience, ninety percent of features you ship in the first year are never used twice. "
    "Ninety percent. That is not a small mistake, that is an entire roadmap thrown away. "
    "So here is the system I use instead. Step one, write down the one job your product does. "
    "If you need more than one sentence, you do not have a product yet. "
    "Step two, find five people with that problem this week. Not next month, this week. "
    "Step three, ship the ugliest thing that solves the job, and charge money for it immediately. "
    "Charging money is the only honest feedback you will ever get. "
    "I sold a spreadsheet for twenty dollars before I wrote a single line of code. "
    "That purchase told me more than six months of surveys ever did. "
    "And here is the part that surprised me. The customers who paid told their friends, "
    "and those friends told other friends, and suddenly distribution was not my problem anymore. "
    "The lesson? Do things that do not scale, and do them embarrassingly early. "
    "If you take one idea from this video, take this one. Talk to users before you build. "
    "Everything else in software is decoration."
)

foreach ($line in $text) { $synth.Speak($line) }
$synth.Dispose()
Write-Output "wrote $OutFile"
