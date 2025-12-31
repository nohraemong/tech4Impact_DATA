import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, cache_dir: str = "/tmp/model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_model(
        self,
        model_id: str,
        force_download: bool = False
    ) -> str:
        """모델을 다운로드하고 로컬 경로 반환"""
        
        model_cache_dir = self.cache_dir / model_id.replace("/", "--")
        
        # 이미 다운로드된 경우 확인
        if model_cache_dir.exists() and not force_download:
            config_file = model_cache_dir / "config.json"
            if config_file.exists():
                logger.info(f"✅ 캐시된 모델 사용: {model_cache_dir}")
                return str(model_cache_dir)
        
        logger.info(f"📥 모델 다운로드 시작: {model_id}")
        logger.info(f"💾 저장 위치: {model_cache_dir}")
        
        # 사용 가능한 디스크 공간 확인
        free_space = self._get_free_space()
        logger.info(f"💿 사용 가능한 공간: {free_space:.1f}GB")
        
        if free_space < 70:  # 32B 모델용 최소 공간
            logger.warning(f"⚠️ 디스크 공간 부족: {free_space:.1f}GB < 70GB")
        
        try:
            # 다운로드 실행
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(model_cache_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                # 불필요한 파일 제외
                ignore_patterns=[
                    "*.bin",  # safetensors만 사용
                    "pytorch_model*.bin",
                    "optimizer.pt",
                    "scheduler.pt",
                    "training_args.bin",
                    "*.msgpack",
                    "*.h5"
                ]
            )
            
            # 다운로드 완료 확인
            model_size = self._calculate_directory_size(model_cache_dir)
            logger.info(f"✅ 다운로드 완료: {model_size:.1f}GB")
            
            return str(model_cache_dir)
            
        except Exception as e:
            logger.error(f"❌ 다운로드 실패: {str(e)}")
            # 실패 시 부분 다운로드 정리
            if model_cache_dir.exists():
                import shutil
                shutil.rmtree(model_cache_dir)
            raise

    def _get_free_space(self) -> float:
        """사용 가능한 디스크 공간 (GB)"""
        statvfs = os.statvfs(self.cache_dir)
        return (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)

    def _calculate_directory_size(self, directory: Path) -> float:
        """디렉토리 크기 계산 (GB)"""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024**3)

    def cleanup_old_models(self, keep_latest: int = 2):
        """오래된 모델 캐시 정리"""
        if not self.cache_dir.exists():
            return
            
        model_dirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]
        if len(model_dirs) <= keep_latest:
            return
            
        # 수정 시간 기준 정렬
        model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 오래된 모델 제거
        for old_dir in model_dirs[keep_latest:]:
            logger.info(f"🗑️ 오래된 모델 캐시 제거: {old_dir}")
            import shutil
            shutil.rmtree(old_dir)

