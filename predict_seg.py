import os
from pathlib import Path

from ultralytics import YOLO


def read_files_in_folder(folder_path):
    """
    读取指定文件夹下所有文件的内容
    """
    folder = Path(folder_path)
    image_paths = []

    # 递归遍历所有文件
    for file_path in folder.rglob('*'):
        if file_path.is_file():
            try:
                # 输出文件路径
                print(f"\n{'=' * 50}")
                print(f"文件路径：{file_path}")
                image_paths.append(file_path)

                # # 读取文件内容（文本模式）
                # with open(file_path, 'r', encoding='utf-8') as file:
                #     content = file.read()
                #     print(f"\n文件内容：\n{content}")

            except UnicodeDecodeError:
                # 处理二进制文件
                print("\n[注意] 该文件是二进制文件或使用了不同的编码，无法用文本模式读取")
            except Exception as e:
                print(f"\n[错误] 读取文件时出错：{str(e)}")

    return image_paths


def predict_seg(paths):
    # Load a model
    model = YOLO("runs/segment/train4/weights/best.pt") # load a custom model

    # 执行推理
    for image_path in paths:
        model.predict(
            task="segment",             # 指定任务类型为实例分割
            source=image_path,          # 输入源：图片、视频、文件夹或摄像头（如 0 表示摄像头）
            conf=0.25,                  # 置信度阈值，过滤低置信度目标
            iou=0.45,                   # IOU 阈值，控制目标框的重叠过滤
            save=True,                  # 是否保存预测结果，默认保存到指定目录
            save_txt=True,              # 是否保存预测结果为文本
            save_conf=True,             # 是否保存预测框的置信度值
            show=False,                 # 是否实时显示预测结果
            device=0                    # 使用设备（0 表示第 0 块 GPU，或者 'cpu' 表示使用 CPU）
        )


if __name__ == "__main__":
    # target_folder = input("请输入要读取的文件夹路径：").strip()
    target_folder = r"data/silkworm-seg/cut"

    if not Path(target_folder).is_dir():
        print("错误：输入的路径不是有效的文件夹")
    else:
        images_path = read_files_in_folder(target_folder)

    predict_seg(images_path)