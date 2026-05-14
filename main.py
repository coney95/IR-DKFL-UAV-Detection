# ================== 3. 导入必要的库 ==================
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPETrainer
from ultralytics.utils import yaml_load
import os
import torch
import cv2
import math
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
from PIL import Image
from tqdm.auto import tqdm
from infrared_knowledge import build_infrared_prompts

# 👇 👇 👇 关键修复：所有代码放进 main 里
if __name__ == '__main__':
    # ============ 论文实验开关 ============
    USE_TDSA = True       # False = mode='simple', True = mode='dual_branch'
    USE_HFBB = True       # False = 用原版 yoloe-v8l.yaml, True = 用我们改的 yaml
    USE_STAR = True       # 对应 v8DetectionLoss 里的 self.use_star
    EXP_NAME = "full"     # 用于命名 run, 方便区分
    # ================== 4. 数据集配置 ==================
    ROOT_DIR = r'E:\project\project_root\datasets\hit-uav'
    train_imgs_dir = 'images/train'
    train_labels_dir = 'labels/train'
    val_imgs_dir = 'images/val'
    val_labels_dir = 'labels/val'
    test_imgs_dir = 'images/test'
    test_labels_dir = 'labels/test'

    classes = ['Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare']
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    print(f"数据集类别: {classes}")

    # ================== 5. 数据可视化函数 ==================
    def yolo2bbox(bboxes):
        xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
        xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
        return xmin, ymin, xmax, ymax

    def plot_box(image, bboxes, labels, classes=classes, colors=colors, pos='above'):
        height, width, _ = image.shape
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        tf = max(lw - 1, 1)
        
        for box_num, box in enumerate(bboxes):
            x1, y1, x2, y2 = yolo2bbox(box)
            xmin = int(x1*width)
            ymin = int(y1*height)
            xmax = int(x2*width)
            ymax = int(y2*height)

            p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
            class_name = classes[int(labels[box_num])]
            color = colors[classes.index(class_name)]

            cv2.rectangle(image, p1, p2, color=color, thickness=lw, lineType=cv2.LINE_AA)

            w, h = cv2.getTextSize(class_name, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = p1[1] - h >= 3

            if pos == 'above':
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(image, p1, p2, color=color, thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(image, class_name, (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                           cv2.FONT_HERSHEY_SIMPLEX, fontScale=lw/3.5, color=(255, 255, 255),
                           thickness=tf, lineType=cv2.LINE_AA)
            else:
                new_p2 = p1[0] + w, p2[1] + h + 3 if outside else p2[1] - h - 3
                cv2.rectangle(image, (p1[0], p2[1]), new_p2, color=color, thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(image, class_name, (p1[0], p2[1] + h + 2 if outside else p2[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, fontScale=lw/3, color=(255, 255, 255),
                           thickness=tf, lineType=cv2.LINE_AA)
        return image

    def plot_dataset_samples(image_path, label_path, num_samples, classes=classes, colors=colors, pos='above'):
        all_images = glob.glob(image_path+'/*')
        all_labels = glob.glob(label_path+'/*')
        all_images.sort()
        all_labels.sort()

        temp = list(zip(all_images, all_labels))
        random.shuffle(temp)
        all_images, all_labels = zip(*temp)
        all_images, all_labels = list(all_images), list(all_labels)

        num_images = len(all_images)
        if num_samples == -1:
            num_samples = num_images

        num_cols = 2
        num_rows = int(math.ceil(num_samples / num_cols))

        plt.figure(figsize=(10 * num_cols, 6 * num_rows))
        for i in range(num_samples):
            image_name = all_images[i].split(os.path.sep)[-1]
            image = cv2.imread(all_images[i])
            
            with open(all_labels[i], 'r') as f:
                bboxes = []
                labels = []
                label_lines = f.readlines()
                for label_line in label_lines:
                    label, x_c, y_c, w, h = label_line.split(' ')
                    x_c = float(x_c)
                    y_c = float(y_c)
                    w = float(w)
                    h = float(h)
                    bboxes.append([x_c, y_c, w, h])
                    labels.append(label)
            
            result_image = plot_box(image, bboxes, labels, classes, colors, pos)
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(image[:, :, ::-1])
            plt.axis('off')
            plt.title(f'Sample {i+1}: {image_name}')
        
        plt.tight_layout()
        plt.show()

    print("可视化训练数据样本...")
    plot_dataset_samples(
        image_path=os.path.join(ROOT_DIR, train_imgs_dir),
        label_path=os.path.join(ROOT_DIR, train_labels_dir),
        num_samples=8
    )

    # ================== 6. 创建数据集配置文件 ==================
    dataset_config = f"""# YOLOE数据集配置文件
path: {ROOT_DIR}
train: images/train
val: images/val
test: images/test
names:
  0: Person
  1: Car
  2: Bicycle
  3: OtherVehicle
nc: 4
"""

    with open('dataset.yaml', 'w') as f:
        f.write(dataset_config)

    print("数据集配置文件已创建: dataset.yaml")

    # ================== 7. YOLOE模型初始化 ==================
    print("正在初始化YOLOE模型...")

    os.environ["PYTHONHASHSEED"] = "0"

    model = YOLOE("yoloe-v8l.yaml")
    del model.model.model[-1].savpe
    model.load("yoloe-v8l-seg.pt")
    model.eval()

    print("YOLOE模型加载完成！")

        # ================== 8. 生成文本提示嵌入 (TDSA / IR-DKFL) ==================
    print("正在生成文本提示嵌入 (集成红外领域专家知识库 IR-DKFL)...")

    names = yaml_load('dataset.yaml')['names']
    class_names = list(names.values())
    print(f"检测类别: {class_names}")

    # ============ TDSA 核心: 红外领域专家描述 ============
    # mode 选项 (用于做消融对比, 对应论文表 5):
    #   'simple'      - 仅类名 (baseline)
    #   'tg_only'     - 仅温度梯度分支
    #   'ri_only'     - 仅热辐射强度分支
    #   'dual_branch' - 完整 TDSA (推荐, 默认)
    TDSA_MODE = 'dual_branch'   # 改这里做消融
    
    ir_prompts = build_infrared_prompts(class_names, mode=TDSA_MODE)
    print(f"\nTDSA 模式: {TDSA_MODE}")
    print(f"红外专家描述示例:")
    for n, p in zip(class_names[:2], ir_prompts[:2]):  # 只打前两个看看
        print(f"  [{n}]: {p[:120]}{'...' if len(p) > 120 else ''}")
    print()

    # 用增强后的红外语义生成 text PE
    tpe = model.get_text_pe(ir_prompts)
    
    # ⚠️ 注意: pe_state 里 names 必须保留原始类名 (不是 ir_prompts),
    # 因为后续 set_classes() 用 names 做类别识别, 但 pe 用增强后的 embedding
    pe_path = f"hit-uav-pe-{TDSA_MODE}.pt"   # 不同模式存不同文件, 方便消融对比
    torch.save({"names": class_names, "pe": tpe}, pe_path)

    print(f"文本提示嵌入已保存: {pe_path}")
    print(f"  - 类别名 (短): {class_names}")
    print(f"  - 实际 CLIP 输入: 详细红外描述 (见上面示例)")

    # ================== 9. 全量微调训练配置 ==================
    print("配置全量微调训练参数...")

    USE_FULL_TUNING = True
    project_name = "runs/full_tuning"
    lr0 = 1e-3
    freeze = []

    print(f"冻结参数数量: {len(freeze)}")
    print("全量微调配置完成！")

    # ================== 10. 开始训练 ==================
    print("开始YOLOE全量微调训练...")

    os.environ['WANDB_DISABLED'] = 'true'

    results = model.train(
        data='dataset.yaml',
        epochs=100,
        close_mosaic=10,
        batch=12,
        optimizer='AdamW',
        lr0=lr0,
        warmup_bias_lr=0.0,
        weight_decay=0.025,
        momentum=0.9,
        workers=2,  # 👈 我帮你调低了，Windows 更稳定
        device="0" if torch.cuda.is_available() else "cpu",
        val_interval=1,
        project=project_name,
        trainer=YOLOEPETrainer,
        freeze=freeze,
        train_pe_path=pe_path,
        name='yoloe_hit_uav_full_tuning'
    )

    print("训练完成！")

    # ================== 11. 训练结果可视化 ==================
    RUNS = str(results.save_dir)
    print("可视化训练结果...",RUNS)

    from imutils import paths

    for image_path in sorted(paths.list_images(RUNS)):
        if os.path.basename(image_path) in ['confusion_matrix.png', 'results.png', 'val_batch0_pred.jpg']:
            image = Image.open(image_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.title(os.path.basename(image_path))
            plt.axis('off')
            plt.show()

    # ================== 12. 模型推理测试 ==================
    print("开始模型推理测试...")

    best_model = YOLOE(f'{RUNS}/weights/best.pt')
    class_names = list(names.values())
    ir_prompts_for_inference = build_infrared_prompts(class_names, mode=TDSA_MODE)
    best_model.set_classes(class_names, best_model.get_text_pe(ir_prompts_for_inference))

    print(f"设置检测类别: {class_names}")

    results = best_model.predict(
        source=os.path.join(ROOT_DIR, test_imgs_dir),
        conf=0.5,
        iou=0.5,
        save=True,
        project=f'{RUNS}',
        name='inference'
    )

    print("推理完成！")

    # ================== 13. 随机展示推理结果 ==================
    print("展示推理结果...")

    indices = list(range(len(results)))
    random_indices = random.sample(indices, min(10, len(results)))
    num_cols = 2
    num_rows = min(5, int(math.ceil(len(random_indices) / num_cols)))

    plt.figure(figsize=(12 * num_cols, 6 * num_rows))

    for i, idx in enumerate(random_indices[:num_rows*num_cols]):
        if i < len(results):
            image = results[i].plot()
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Detection Result {i+1}')

    plt.tight_layout()
    plt.show()

    # ================== 14. 预测与真实标签对比 ==================
    print("对比预测结果与真实标签...")

    ground_colors = [(255, 0, 0) for _ in range(len(classes))]

    plot_dataset_samples(
        image_path=f'{RUNS}/inference',
        label_path=os.path.join(ROOT_DIR, test_labels_dir),
        num_samples=10,
        classes=classes,
        colors=ground_colors,
        pos='below'
    )

    print("全量微调训练完成！")