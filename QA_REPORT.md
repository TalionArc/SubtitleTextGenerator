# QA report — LectureSubtitleBatcher 1.2.1 portable

최종 검증일: 2026-09-07 (Asia/Seoul)

## 환경

- Windows 11 x64
- NVIDIA GeForce RTX 3080 12GB, 드라이버 591.86
- Ryzen 7 5800X, RAM 64GB
- 빌드 Python 3.14.3 / PyInstaller 6.22.0

## 자동 검사

- Ruff: 통과
- Pytest: 48개 통과
- 검사 범위: 재귀 영상·오디오 스캔, 한국어·괄호 경로, 중복 파일명, SRT/TXT 결과 판정, 지원 오디오 확장자 11종, TXT 타임스탬프 포함·제외와 설정 유지, 용어집 제한, Whisper JSON 파싱, 불확실도 판정, 구간 병합, hybrid 교체, SRT/TXT 형식과 단일 백업, 이전 앱 버전 Turbo 캐시 재개, OOM batch 재시도, 원본 변경 감지, 순차 대기열, 실패 후 다음 작업 계속, 부분 다운로드 재개, CUDA 실패 시 CPU 우회 금지

## 실제 설치·GPU 검사

- 공개 Faster-Whisper-XXL r245.4 아카이브 다운로드 및 SHA-256 검증: 통과
- Windows 기본 `tar.exe`를 통한 BCJ2 7z 해제: 통과
- 고정 커밋의 large-v3-turbo와 large-v3 전체 다운로드·파일별 SHA-256 검증: 통과
- 1.2.1 배포판 설치 위치: EXE 옆 `LectureSubtitleBatcher-data` (현재 PC의 기존 `dist` 1.0.0 설치본은 `%LOCALAPPDATA%` 유지)
- 실제 설치 용량: 약 10.1GB(복구용 엔진 아카이브 포함), 설치 여유 공간 기준은 12GB
- CUDA/FP16 Turbo 및 large-v3 실행: 통과

## 실제 강의 스모크 테스트

원본 `강의\13-1\DocZoomScreenCapture.mp4`를 읽기만 하고, 중간 오디오는 앱 데이터의 테스트 사본으로 처리했다. 원본 폴더에는 SRT나 기타 파일을 만들지 않았다.

- 5분 샘플 Turbo 적응형 처리: 7.6초, 엔진 처리량 약 119 audio-sec/s
- 실제 large-v3 구간 인식: 성공, 반환 단어 타임스탬프가 원본 절대시간임을 확인
- 강제로 만든 불확실 구간: Turbo 캐시 재사용 → large-v3 교체 → hybrid SRT 생성 성공
- 28.2분 샘플 전체 적응형 처리: 66.2초
- 1시간 환산: 약 141초(2분 21초)로 목표인 5분 이내 충족
- 불확실 구간 2개를 한 `clip_timestamps` 호출로 처리하고 양쪽 구간 모두 결과 단어가 존재함을 확인
- 공개 엔진이 완전한 JSON을 저장한 뒤 비정상 종료 코드를 반환한 사례를 재현했다. 앱은 JSON 발화 세그먼트, `100%`, 저장 완료 문구, 작업 완료 문구가 모두 있을 때만 결과를 회수하도록 보강했으며 재시험에서 hybrid 교체에 성공했다. 그 외 비정상 종료는 Turbo fallback 경고로 처리한다.
- 실제 인식 중 취소: 1.8초 안에 엔진 프로세스 트리 종료, JSON/SRT 미생성, 잔여 Faster-Whisper·FFmpeg 프로세스 없음 확인

## 출력·UI·패키징 검사

- 실제 SRT: UTF-8 BOM, CRLF, 최대 2줄, 줄당 최대 22자, 시간 순서 검증 통과
- 기존 SRT 재생성: `.srt.bak` 한 개만 유지함을 확인
- 실제 20초 M4A: CUDA/FP16 Turbo 인식 후 `[시작 - 종료]` 타임스탬프가 붙은 TXT 2줄 생성, UTF-8 BOM·CRLF 확인
- 같은 M4A에서 옵션을 끄고 재생성: Turbo 캐시 재사용으로 0.225초에 일반 TXT 생성, 직전 타임스탬프 TXT와 동일한 `.txt.bak` 한 개 보존 확인
- M4A 기존 TXT 감지로 기본 작업을 건너뛰는 동작 확인
- 지원 오디오: M4A, MP3, WAV, FLAC, AAC, OGG, OPUS, WMA, AIF, AIFF, MKA
- 강의 폴더 검색: 영상 20개, 자막 없음 20개, 전체 경로 20개 모두 고유
- 서로 다른 폴더의 `DocZoomScreenCapture.mp4` 8개를 별도 항목으로 구분
- Tk GUI 시작: 강의 영상 20개 목록 표시와 실제 M4A 검사 폴더의 오디오 1개·기존 TXT 상태 표시 확인
- 단일 EXE를 Python 경로가 제거된 환경에서 실행: 통과
- 최종 배포 경로: `output\LectureSubtitleBatcher.exe`
- `backup\1.0.0`, `backup\1.1.0`, `backup\1.2.0`, `backup\1.2.1`에 EXE를 버전별로 보존하고 출력본과 1.2.1 백업의 SHA-256이 같음을 확인
- Python을 PATH에서 제거한 환경에서 패키징된 1.2.1 EXE 기동 및 종료 후 잔여 프로세스 없음 확인
- 최종 EXE: 12,777,062 bytes, 파일/제품 버전 1.2.1, 제품명 `강의 자막·텍스트 생성기`
- 최종 EXE SHA-256: `FF860F02AE7BDEBFDD349B7E26D7B93A9B105BE27C2F21262E2DFFFF242002AE`
- PyInstaller 아카이브 내부에 Python 3.14 DLL, Tk/Tcl DLL, 앱 PYZ와 표준 라이브러리가 포함됨을 확인했다. 외부 PE import는 Windows 시스템 DLL뿐이다.

## 확인 범위의 한계

자동 품질 신호와 짧은 실제 구간의 결과 구조는 확인했지만, 20개 강의 전체를 처리하거나 사람이 모든 문장을 듣고 의미 정확도를 채점하지는 않았다. 사용자가 필요한 영상을 개별 선택해 처리하도록 한 제품 요구사항에 맞춰 전체 강의 자동 처리는 수행하지 않았다.
