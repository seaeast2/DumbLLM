# DumbLLM

## 가상환경(venv) 설정

Windows PowerShell에서 프로젝트 루트로 이동한 뒤 아래 명령을 실행합니다.

```powershell
# 가상환경 생성
py -m venv .venv

# 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# pip 업데이트 및 의존성 설치
python -m pip install --upgrade pip
pip install -r requirements.txt
```

활성화에 성공하면 명령 프롬프트 앞에 `(.venv)`가 표시됩니다. 프로젝트는 다음과 같이 실행합니다.

```powershell
python run.py
```

작업을 마치고 가상환경을 종료하려면 다음을 실행합니다.

```powershell
deactivate
```

PowerShell 실행 정책 때문에 활성화가 차단되면, 현재 터미널에서만 아래 설정을 적용한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

VS Code에서는 `Ctrl+Shift+P` → `Python: Select Interpreter`에서 `.venv` 인터프리터를 선택합니다. `.venv` 디렉터리는 Git에 커밋하지 않습니다.
책보고 공부하면서 만드는 멍청한 LLM 프로젝트
