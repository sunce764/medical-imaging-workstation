# organs.onnx 标签映射（由 TotalSegmentator 真值实测恢复）

数据: 单例 TotalSegmentator-CT-Lite（1.5mm 各向同性，thorax-abdomen-pelvis）。
方法: GUI 同款滑窗推理 → 与真值逐标签 Dice，取每个输出标签重叠最大的真值器官。

| our label | 实测器官 | Dice |
|---|---|---|
| 1 | spleen | 0.966 |
| 2 | kidney_right | 0.985 |
| 3 | kidney_left | 0.977 |
| 4 | gallbladder | 0.818 |
| 5 | liver | 0.945 |
| 6 | stomach | 0.840 |
| 7 | pancreas | 0.859 |
| 8 | adrenal_R | 0.938 |
| 9 | adrenal_L | 0.924 |
| 10 | lung_upper_L | 0.991 |
| 11 | lung_lower_L | 0.991 |
| 12 | lung_upper_R | 0.967 |
| 13 | lung_middle_R | 0.956 |
| 14 | lung_lower_R | 0.990 |
| 15 | esophagus | 0.933 |
| 16 | trachea | 0.955 |
| 17 | thyroid | 0.794 |
| 18 | small_bowel | 0.910 |
| 19 | duodenum | 0.895 |
| 20 | colon | 0.855 |
| 21 | urinary_bladder | 0.871 |
| 22 | (absent in GT)  ·低置信 | 0.000 |
| 23 | (absent in GT)  ·低置信 | 0.000 |

可信匹配(Dice≥0.2): 21/23。
平均 Dice: 0.842。
