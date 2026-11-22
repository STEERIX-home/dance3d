# 댄스 3D 기술 문서 (2026-11-16)

홍보 영상에 들어간 핵심 기술을 정리한 문서다. 춤 영상 → 관절 → 3D 몸 → 영상 합성의 방법만 다룬다.

- **결과**: `댄스3D_기술문서_20261116.pdf` (A4 4쪽, 한국어).
- **차례**: 한눈에 보기 → 1 영상 → 관절 → 2 관절 → 3D 몸 → 3 3D 몸 → 영상 합성 → 한계와 조건.
- **원본**:
  - `work/doc.html`: 본문
  - `work/fig/`: 홍보 영상에서 뽑은 장면과 머리글 로고
  - `work/build.py`: PDF와 쪽 그림을 만든다
  - `work/pairs.py`: 두 쪽씩 나란히 놓아 본다
- **다시 만들기**: WeasyPrint 70.0과 글꼴 Noto Sans CJK KR이 필요하다. 서버에서는 작업 폴더 `/srv/training/jobs/20261116-110453-dance-tech-document-v2-7ba8`의 `cache/venv`에 있다. `work` 폴더에서 다음을 실행한다.
  `python -B build.py ../댄스3D_기술문서_20261116.pdf preview && python -B pairs.py preview`
- **주의**: `doc.html`에 `<meta charset="utf-8">`가 없으면 한글이 깨진다.
- **이전 판**: 처음 판(16쪽, 만든 날짜 순서와 홍보 영상 제작까지 적은 것)은 커밋 `a94feb7`에 있다.
