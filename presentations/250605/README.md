# Liquid 모델 설정 가이드

이 가이드는 Liquid 모델을 로컬에서 실행하기 위한 완전한 설정 방법을 제공합니다.

## 필요한 파일들

Liquid 모델의 이미지 생성 및 평가 기능을 사용하기 위해서는 다음 파일들이 필요합니다:

### 1. Liquid 메인 모델
- `models/Liquid_V1_7B/` - 7B 파라미터 Liquid 모델 (약 17GB)

### 2. VQGAN 파일들 (이 디렉토리)
- `vqgan.ckpt` - VQGAN 모델의 체크포인트 파일
- `vqgan.yaml` - VQGAN 모델의 설정 파일

## 설정 방법

### 1단계: Liquid 메인 모델 다운로드

프로젝트 루트 디렉토리에서 실행:

```bash
# Hugging Face CLI 설치 (없는 경우)
pip install huggingface_hub

# Liquid 7B 모델 다운로드 (약 17GB)
huggingface-cli download --resume-download Junfeng5/Liquid_V1_7B --local-dir models/Liquid_V1_7B
```

### 2단계: VQGAN 파일들 다운로드

```bash
# chameleon 디렉토리로 이동
cd evaluation/chameleon

# VQGAN 파일들 다운로드
wget -P . https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.ckpt 
wget -P . https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.yaml
```

### 3단계: 로컬 실행 방법

모든 파일 다운로드 완료 후, evaluation 폴더에서 실행할 수 있습니다:

```bash
# evaluation 디렉토리로 이동
cd ../

# 텍스트 대화
python inference_t2t.py --model_path ../models/Liquid_V1_7B --prompt "Write me a poem about Machine Learning."

# 이미지 이해
python inference_i2t.py --model_path ../models/Liquid_V1_7B --image_path samples/baklava.png --prompt 'How to make this pastry?'

# 이미지 생성 (30GB 미만 GPU는 --load_8bit 추가)
python inference_t2i.py --model_path ../models/Liquid_V1_7B --prompt "young blue dragon with horn lightning in the style of dd fantasy full body"
```


## 파일 확인

모든 다운로드가 완료되면 다음 구조가 되어야 합니다:
```
프로젝트루트/
├── models/
│   └── Liquid_V1_7B/              # 메인 모델 (약 17GB)
│       ├── config.json
│       ├── model-00001-of-00004.safetensors
│       └── ... (기타 모델 파일들)
└── evaluation/
    └── chameleon/
        ├── vqgan.ckpt             # VQGAN 체크포인트
        └── vqgan.yaml             # VQGAN 설정
```

## 주의사항

- **디스크 공간**: Liquid_V1_7B 모델은 약 17GB, VQGAN 파일들도 추가 용량이 필요합니다
- **GPU 메모리**: 이미지 생성 시 30GB 미만 GPU는 `--load_8bit` 옵션을 사용하세요
- **네트워크**: 안정적인 연결에서 다운로드하세요 (파일 크기가 큼)
- **재시작**: `--resume-download` 옵션으로 중단된 다운로드를 재개할 수 있습니다

## 문제 해결

### 모델 다운로드 문제
- `huggingface-cli` 명령어가 없다면: `pip install huggingface_hub`
- 다운로드 중단됨: `--resume-download` 옵션으로 재시작
- 디스크 공간 부족: 최소 20GB 이상 여유 공간 확보

### 실행 문제  
- GPU 메모리 부족: `--load_8bit` 옵션 추가
- 모듈 찾을 수 없음: 프로젝트 루트에서 실행하는지 확인
- VQGAN 파일 없음: `evaluation/chameleon/` 디렉토리에 파일 존재 확인

### 기타
- 네트워크 문제: 안정적인 연결에서 재시도

설정이 완료되면 Liquid 모델의 텍스트 대화, 이미지 이해, 이미지 생성 기능을 모두 사용할 수 있습니다!

## 참조

이 가이드는 [Liquid 프로젝트](https://github.com/FoundationVision/Liquid/tree/main)를 참조하여 작성되었습니다. 