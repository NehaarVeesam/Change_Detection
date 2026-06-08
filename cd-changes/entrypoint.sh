#!/usr/bin/env bash
# Legacy entrypoint — prefer Dockerfile ENTRYPOINT ["python", "qwen_family.py"].
exec python /app/qwen_family.py "$@"
