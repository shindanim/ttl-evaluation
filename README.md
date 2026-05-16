# KG Turtle Evaluation

Turtle(`.ttl`) 형식의 `true.ttl`과 `pred.ttl` 지식 그래프를 비교하고, 평가 결과와 HTML 시각화를 제공합니다.

## 설치

Anaconda를 사용한다면 새 환경을 만드는 방식을 권장합니다.

```powershell
conda env create -f environment.yml
conda activate kg-eval
```

이미 사용하는 conda 환경이 있다면 해당 환경을 activate한 뒤 아래처럼 설치해도 됩니다.

```powershell
pip install -r requirements.txt
```

## 평가 실행

Strict 모드는 subject/predicate/object가 모두 같은 triple만 match로 봅니다.

```powershell
python -m kg_eval.evaluator --true examples/true.ttl --pred examples/pred.ttl --mode strict
```

Cosine 모드는 triple의 subject, predicate, object 라벨을 각각 임베딩한 뒤 평균 cosine similarity가 threshold 이상인 best match를 찾습니다.

```powershell
python -m kg_eval.evaluator --true examples/true.ttl --pred examples/pred.ttl --mode cosine --threshold 0.75
```

## HTML 시각화 파일 생성

```powershell
python -m kg_eval.visualize --true examples/true.ttl --pred examples/pred.ttl --output kg_visualization.html --mode strict
```

생성된 HTML에는 `All`, `true.ttl`, `pred.ttl` 보기 버튼과 `Highlight matched nodes` 토글 버튼이 있습니다. 켜져 있으면 매칭된 노드가 빨간색으로 표시되고, 끄면 true/pred 그래프 기본 색상으로 돌아갑니다.

서버로 볼 때는 상단의 `Strict`, `Cosine` 버튼으로 평가 모드를 전환할 수 있습니다. `Matched triples` 버튼을 누르면 매칭된 triple 목록을 텍스트로 확인할 수 있습니다.

## 서버로 띄우기

```powershell
python -m kg_eval.server --true examples/true.ttl --pred examples/pred.ttl --port 5000
```

브라우저에서 아래 주소를 열면 됩니다.

```text
http://127.0.0.1:5000/
```

평가 JSON만 보고 싶으면:

```text
http://127.0.0.1:5000/api/evaluate?mode=strict
```

Cosine 시각화는 처음 실행 시 SentenceTransformer 모델을 다운로드할 수 있습니다.

```text
http://127.0.0.1:5000/?mode=cosine&threshold=0.75
```
