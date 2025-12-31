import os
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from huggingface_hub import snapshot_download

def download_qwen_model():
    """Qwen 모델을 로컬에 다운로드"""

    model_id = "Qwen/Qwen3-VL-32B-Instruct-FP8"
    local_dir = "./models/Qwen3-VL-32B-Instruct-FP8"

    print(f"모델 다운로드 시작: {model_id}")
    print(f"저장 위치: {local_dir}")

    # 디렉토리 생성
    os.makedirs(local_dir, exist_ok=True)

    try:
        # 전체 모델 파일 다운로드
        snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 심볼릭 링크 대신 실제 파일 복사
        resume_download=True,          # 중단된 다운로드 재개
        )

        print("✅ 모델 다운로드 완료!")

# 다운로드된 파일 확인
        print("다운로드된 파일:")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024**2)  # MB
                print(f"  {file}: {file_size:.1f}MB")

    # 모델 로드 테스트
        print("모델 로드 테스트...")
        tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
        print("✅ 토크나이저 및 프로세서 로드 성공!")

    # GPU가 있으면 모델도 테스트
        import torch
        if torch.cuda.is_available():
            print("GPU 모델 로드 테스트...")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
            local_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
            )
            print("✅ 모델 로드 성공!")

    except Exception as e:
        print(f"❌ 다운로드 실패: {str(e)}")
        raise

if __name__ == "__main__":
    download_qwen_model()
