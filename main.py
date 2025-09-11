#!/usr/bin/env python3
"""
General Student Agent API 启动脚本

使用方法:
    python start_api.py [options]

环境变量配置:
    HOST: 服务器主机地址 (默认: 0.0.0.0)
    PORT: 服务器端口 (默认: 8001)
    RELOAD: 是否开启热重载 (默认: true)
    LOG_LEVEL: 日志级别 (默认: info)
    WORKERS: 工作进程数 (默认: 1)
    ENVIRONMENT: 运行环境 (dev/staging/prod, 默认: development)

示例:
    # 开发模式启动
    python start_api.py
    
    # 生产模式启动
    ENVIRONMENT=prod PORT=8001 WORKERS=4 python start_api.py
    
    # 指定配置启动
    python start_api.py --host 127.0.0.1 --port 8002 --no-reload
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import uvicorn


def setup_logging():
    """配置日志"""
    log_level = os.getenv("LOG_LEVEL", "info").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            # 可以添加文件日志处理器
            # logging.FileHandler("api.log")
        ],
    )

    # 设置uvicorn日志级别
    logging.getLogger("uvicorn.access").setLevel(
        getattr(logging, log_level, logging.INFO)
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="General Student Agent API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind the server to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8001)),
        help="Port to bind the server to (default: 8001)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", 1)),
        help="Number of worker processes (default: 1)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("RELOAD", "true").lower() == "true",
        help="Enable auto-reload on code changes (default: true)",
    )

    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")

    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug"],
        help="Log level (default: info)",
    )

    parser.add_argument(
        "--app",
        type=str,
        default="src.service.general_student.api.main:app",
        help="FastAPI app import string",
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 处理reload参数
    if args.no_reload:
        args.reload = False

    # 设置日志
    os.environ["LOG_LEVEL"] = args.log_level
    setup_logging()

    logger = logging.getLogger(__name__)

    # 打印启动信息
    environment = os.getenv("ENVIRONMENT", "development")
    logger.info("=" * 60)
    logger.info("General Student Agent API Server")
    logger.info("=" * 60)
    logger.info(f"Environment: {environment}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"App: {args.app}")
    logger.info("=" * 60)

    # 配置uvicorn
    config = {
        "app": "src.service.general_student.api.main:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level.lower(),
        "access_log": True,
    }
    try:
        # 启动服务器
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
