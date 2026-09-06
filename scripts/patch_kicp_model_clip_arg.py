# scripts/patch_kicp_model_clip_arg.py
# -*- coding: utf-8 -*-

"""
修复 models/kicp_model.py 中 KICPCLIPEncoder 参数名不匹配的问题。

报错：
    TypeError: __init__() got an unexpected keyword argument 'model_name'

原因：
    当前项目里的 KICPCLIPEncoder 不接收 model_name 参数，
    而是接收 clip_model_name 参数。

作用：
    自动把 models/kicp_model.py 里的：
        model_name=clip_model_name
    替换为：
        clip_model_name=clip_model_name
"""

from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    kicp_model_path = project_root / "models" / "kicp_model.py"

    if not kicp_model_path.exists():
        raise FileNotFoundError(f"找不到文件：{kicp_model_path}")

    text = kicp_model_path.read_text(encoding="utf-8")

    old = "model_name=clip_model_name"
    new = "clip_model_name=clip_model_name"

    if old not in text and new in text:
        print("已经修复过了，不需要重复修改。")
        return

    if old not in text:
        print("没有找到需要替换的内容。")
        print("请检查 models/kicp_model.py 中 KICPCLIPEncoder(...) 的参数名。")
        return

    text = text.replace(old, new)

    backup_path = kicp_model_path.with_suffix(".py.bak")
    backup_path.write_text(kicp_model_path.read_text(encoding="utf-8"), encoding="utf-8")
    kicp_model_path.write_text(text, encoding="utf-8")

    print("修复完成。")
    print(f"已备份原文件：{backup_path}")
    print(f"已修改文件：{kicp_model_path}")


if __name__ == "__main__":
    main()
