from pathlib import Path
import urllib.request

url = ("https://raw.githubusercontent.com/rickiepark/"
       "llm-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data/raw"
data_dir.mkdir(parents=True, exist_ok=True)

file_path = data_dir / "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# 코드 2-1 파이썬으로 단편 소설을 테스트 샘플로 읽기
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("총 문자 개수: ", len(raw_text))
print("처음 500자: ", raw_text[:500])  # 처음 500자만 출력
