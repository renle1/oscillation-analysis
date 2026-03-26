# Workspace Setup Notes

This file defines the Git/Python environment baseline for this repository.

## Git

- Repository path: `C:\Users\danB\Desktop\oscillation-analysis`
- Default branch: `main` (tracks `origin/main`)
- Remote `origin`:
  - `https://github.com/renle1/oscillation-analysis.git`

### Common Git Flow

```powershell
cd "C:\Users\danB\Desktop\oscillation-analysis"
git pull
git status
git add .
git commit -m "message"
git push
```

## Python

- Preferred interpreter:
  - `C:\Users\danB\AppData\Local\Programs\Python\Python312\python.exe`
- Verified version:
  - `Python 3.12.7`

### Compile Check Example

```powershell
& "C:\Users\danB\AppData\Local\Programs\Python\Python312\python.exe" -m py_compile "osc_modul\osc_alert_burst_policy_modul.py" "osc_modul\osc_config_modul.py" "osc_modul\osc_runtime_modul.py"
```

### Semantic Boundary Validation Example

```powershell
& "C:\Users\danB\AppData\Local\Programs\Python\Python312\python.exe" "tools\check_semantic_boundaries.py"
```

## Notes

- `py -3.12` launcher alias may be unavailable in this environment.
  Use the absolute Python path above.
- To avoid GitHub email privacy push errors, use a noreply email for Git.
  - `205188399+renle1@users.noreply.github.com`
