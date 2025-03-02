import matplotlib.pyplot as plt
import numpy as np

# 模拟数据：表1和表2
# 表1的数据
data_mean_privacyfree_2 = [0.65512 ,0.70552 ,0.72756 ,0.74254 ,0.74882 ,0.7488  ,0.74644 ,0.74724 ,0.74488,0.7441  ,0.74486 ,0.74408 ,0.7433  ,0.74252 ,0.74252]
data_mean_ours_2 = [0.65512 ,0.70552 ,0.72756 ,0.74254 ,0.74882 ,0.7488  ,0.74644 ,0.74724 ,0.74488,0.7441  ,0.74486 ,0.74408 ,0.7433  ,0.74252 ,0.74252]
data_mean_dropout_2= [0.51182 ,0.53624 ,0.54252 ,0.54962 ,0.55984 ,0.56458 ,0.60708 ,0.62836 ,0.6323,0.64408 ,0.66218 ,0.67088 ,0.67482 ,0.68188 ,0.69132]
data_mean_strongforall_2 = [0.49448 ,0.50944 ,0.50786 ,0.50788 ,0.51494 ,0.51104 ,0.52206 ,0.52678 ,0.53622,0.52834 ,0.54804 ,0.563   ,0.59216 ,0.6063  ,0.60392]


#mnist-iid
# data_mean_ours_2 = [0.1695 ,0.6455 ,0.8116 ,0.8151 ,0.863  ,0.8694 ,0.8742 ,0.8797 ,0.8788 ,0.862,0.8826 ,0.8813 ,0.8846 ,0.8849 ,0.8944]
# data_mean_dropout_2= [0.1642 ,0.3243 ,0.478  ,0.4693 ,0.6238 ,0.6987 ,0.7472 ,0.764  ,0.801  ,0.7929 ,0.8044 ,0.7945 ,0.8205 ,0.8216 ,0.8259]
# data_mean_strongforall_2 = [0.13752 ,0.14822 ,0.24338 ,0.27244 ,0.35852 ,0.44158 ,0.48588 ,0.48726 ,0.55488,0.566   ,0.61662 ,0.61834 ,0.6692  ,0.665   ,0.68862]

#mnist-niid
# data_mean_ours_2 = [0.103  ,0.2725 ,0.4929 ,0.385  ,0.533  ,0.5382 ,0.5938 ,0.6637 ,0.7243 ,0.7132,0.8069 ,0.7324 ,0.7485 ,0.8148 ,0.8027]
# data_mean_dropout_2= [0.103  ,0.2167 ,0.19   ,0.124  ,0.4287 ,0.3761 ,0.4009 ,0.5147 ,0.4795 ,0.5778,0.6803 ,0.4472 ,0.6188 ,0.7121 ,0.7005]
# data_mean_strongforall_2 = [.1029 ,0.1798 ,0.1714 ,0.1104 ,0.2499 ,0.1625 ,0.2939 ,0.3105 ,0.3284 ,0.3516 ,0.4693 ,0.3668 ,0.4643 ,0.5143 ,0.6281]



# 表2的数据
data_mean_privacyfree_1 = [0.6535, 0.67718, 0.70472, 0.71182, 0.72284, 0.73464, 0.74722, 0.748, 0.748, 0.748, 0.748, 0.74724, 0.752, 0.74962, 0.7496]
data_mean_ours_1 = [0.64252, 0.66378, 0.68032, 0.69212, 0.70392, 0.71102, 0.7118, 0.71572, 0.71812, 0.72202, 0.72204, 0.72362, 0.72756, 0.73148, 0.72994]
data_mean_dropout_1 = [0.62676, 0.6244, 0.66772, 0.6945, 0.6811, 0.68032, 0.6929, 0.70708, 0.70866, 0.71496, 0.7126, 0.71968, 0.70316, 0.71024, 0.70708]
data_mean_strongforall_1 = [0.6252, 0.65432, 0.65906, 0.65752, 0.67874, 0.65514, 0.66848, 0.674, 0.68348, 0.66064, 0.68188, 0.68188, 0.67794, 0.66774, 0.66696]

#mnist-iid
# data_mean_ours_1 = [0.14556 ,0.22382 ,0.43762 ,0.5954  ,0.6616  ,0.70878 ,0.75664 ,0.79152 ,0.81426,0.83232 ,0.84314 ,0.85366 ,0.8617  ,0.86982 ,0.87516]
# data_mean_dropout_1 = [0.14916 ,0.14714 ,0.23138 ,0.28928 ,0.38248 ,0.40906 ,0.4973  ,0.52632 ,0.5828,0.62934 ,0.64704 ,0.69902 ,0.73586 ,0.73308 ,0.7498]
# data_mean_strongforall_1 = [0.11278 ,0.1551  ,0.16028 ,0.1778  ,0.21324 ,0.25906 ,0.30624 ,0.3406  ,0.3651,0.41062 ,0.42838 ,0.45244 ,0.47722 ,0.52958 ,0.54824]

#mnist-niid
# data_mean_ours_1 = [0.1005  ,0.1065  ,0.2292  ,0.27996 ,0.43296 ,0.47836 ,0.5316  ,0.59196 ,0.58924,0.58774 ,0.63578 ,0.67278 ,0.66708 ,0.6988  ,0.68928]
# data_mean_dropout_1 = [0.10902 ,0.14636 ,0.1508  ,0.16104 ,0.19248 ,0.22762 ,0.25834 ,0.31172 ,0.3649,0.37512 ,0.41486 ,0.45388 ,0.50888 ,0.54956 ,0.5364]
# data_mean_strongforall_1 = [0.10824 ,0.13086 ,0.1327  ,0.13954 ,0.16938 ,0.20718 ,0.24936 ,0.27424 ,0.2725,0.31594 ,0.3246  ,0.3379  ,0.39836 ,0.44474 ,0.42018]

# X轴的位置
x = np.arange(1, len(data_mean_ours_1) + 1)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制四条曲线
plt.plot(x, data_mean_privacyfree_1, label='PrivacyFree (ori)', color='blue', marker='o', linestyle='-', markersize=6)
plt.plot(x, data_mean_ours_1, label='Ours (ori)', color='red', marker='x', linestyle='-', markersize=6)
plt.plot(x, data_mean_dropout_1, label='Dropout (ori)', color='green', marker='s', linestyle='-', markersize=6)
plt.plot(x, data_mean_strongforall_1, label='StrongForAll (ori)', color='purple', marker='d', linestyle='-', markersize=6)

# 绘制表2的数据
plt.plot(x, data_mean_privacyfree_2, label='PrivacyFree (test)', color='blue', marker='o', linestyle='--', markersize=6)
plt.plot(x, data_mean_ours_2, label='Ours (test)', color='red', marker='x', linestyle='--', markersize=6)
plt.plot(x, data_mean_dropout_2, label='Dropout (test)', color='green', marker='s', linestyle='--', markersize=6)
plt.plot(x, data_mean_strongforall_2, label='StrongForAll (test)', color='purple', marker='d', linestyle='--', markersize=6)

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Comparison of Fed-Heart-Disease:PrivacyFree, Ours, Dropout, StrongForAll')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# 显示图形
plt.show()