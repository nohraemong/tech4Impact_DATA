import os
import bentoml
from typing import List, Dict, Any

# vllm 모듈 임포트
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# 환경 변수 설정
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
TENSOR_PARALLEL_SIZE = 4  # g4dn.12xlarge (GPU 4장)

@bentoml.service(
    name="qwen3_vllm_service",  # 서비스 이름 (소문자 권장)
    resources={"gpu": 4},
    traffic={"timeout": 300}
)
class Qwen3VLLMService:  # <--- [중요] bentofile.yaml에 적은 이름과 똑같아야 함!
    def __init__(self):
        print(f"Loading model from: {MODEL_PATH}...")
        
        # ---------------------------------------------------------------------
        # vLLM 엔진 설정 (T4 GPU 호환성 최적화)
        # ---------------------------------------------------------------------
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
            gpu_memory_utilization=0.90, # OOM 방지를 위해 90%만 사용
            max_model_len=4096,      # T4 VRAM 절약을 위해 컨텍스트 길이 제한
            dtype="float16",         # [중요] T4는 bf16/fp8 미지원 -> float16 강제
            enforce_eager=True       # T4 호환성 및 메모리 안정성 확보
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Model loaded successfully.")

    @bentoml.api
    async def generate(
        self,
        messages: List[Dict[str, Any]] = [{"role": "user", "content": "Hello!"}],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        
        # 샘플링 파라미터
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )

        request_id = bentoml.utils.generate_request_id()

        # 프롬프트 구성 (채팅 템플릿 적용)
        # Qwen3-VL은 이미지 처리가 복잡하므로 여기서는 텍스트 전용 예시입니다.
        # 실제 이미지는 content 안에 리스트 형태로 들어옵니다.
        prompt = ""
        for msg in messages:
            content = msg.get("content", "")
            # 간단한 텍스트 연결 (실제 서비스에선 tokenizer.apply_chat_template 권장)
            prompt += f"<|im_start|>{msg['role']}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        # 추론 실행
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id
        )

        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text

        return final_output