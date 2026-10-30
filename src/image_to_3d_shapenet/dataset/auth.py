"""Credential discovery for gated ShapeNetCore access."""

from __future__ import annotations

import os


ACCESS_INSTRUCTIONS = """Không tìm thấy HF_TOKEN hợp lệ.
1. Đăng nhập Hugging Face: https://huggingface.co/login
2. Mở https://huggingface.co/datasets/ShapeNet/ShapeNetCore
3. Điền họ tên, giảng viên và trường theo biểu mẫu truy cập.
4. Chấp nhận điều khoản ShapeNetCore và chờ cấp quyền nếu cần.
5. Tạo token loại Read tại https://huggingface.co/settings/tokens
6. Lưu token dưới tên HF_TOKEN trong biến môi trường hoặc Colab Secrets.
Token không được ghi vào config, notebook hoặc source code."""


class MissingTokenError(RuntimeError):
    """Raised when neither the environment nor Colab provides HF_TOKEN."""


def _read_colab_token() -> str | None:
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        value = userdata.get("HF_TOKEN")
    except Exception:
        return None
    return str(value).strip() if value else None


def get_hf_token() -> str:
    """Read HF_TOKEN from the environment first, then Colab userdata."""
    environment_value = os.getenv("HF_TOKEN")
    if environment_value and environment_value.strip():
        return environment_value.strip()
    colab_value = _read_colab_token()
    if colab_value:
        return colab_value
    raise MissingTokenError(ACCESS_INSTRUCTIONS)

