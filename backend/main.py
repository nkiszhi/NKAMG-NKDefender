import logging
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.codefender_client import DEFAULT_MODEL_NAMES, scan_file
from app.config import settings
from app.file_info import save_upload


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("codefender")

app = FastAPI(
    title="Codefender 多模型恶意文件检测",
    version="1.0.0",
    description="独立的 Codefender 文件检测后端",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _response_payload(info: dict, exe_result: dict, detection_mode: str) -> dict:
    return {
        "success": True,
        "original_filename": info["filename"],
        "filename": info["filename"],
        "sha256": info["sha256"],
        "md5": info["md5"],
        "file_size": info["file_size"],
        "file_type": info["file_type"],
        "is_pe": info["is_pe"],
        "pe_info": info["pe_info"],
        "query_result": info["query_result"],
        "exe_result": exe_result,
        "results": exe_result,
        "detection_mode": detection_mode,
        "detection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/")
async def root() -> dict:
    return {"message": "Codefender backend is running", "docs": "/docs"}


@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy"}


@app.get("/models")
async def models() -> dict:
    return {"count": len(DEFAULT_MODEL_NAMES), "models": DEFAULT_MODEL_NAMES}


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    try:
        file_path, info = await save_upload(file)
    except ValueError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except Exception as error:
        logger.exception("保存上传文件失败")
        raise HTTPException(status_code=500, detail=f"保存上传文件失败: {error}") from error

    logger.info("开始检测上传文件: filename=%s sha256=%s", info["filename"], info["sha256"])
    if not info["is_pe"]:
        file_path.unlink(missing_ok=True)
        reason = info["pe_info"].get("reason") or "不是有效PE样本"
        raise HTTPException(status_code=400, detail=f"不是有效PE样本: {reason}")

    exe_result = scan_file(file_path)
    return _response_payload(info, exe_result, "file")


if __name__ == "__main__":
    import uvicorn

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
