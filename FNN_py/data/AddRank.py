import os
import pandas as pd

datasets = ['gsdtsr', 'paintcontrol', 'iofrol', 'rails']
# 最大Cycle，需要调试以使放大后的Cycle接近整数 gsdtsr 没问题 336 312 320 3263
maxCycle = {'gsdtsr': 335, 'paintcontrol': 351, 'iofrol': 319, 'rails': 3259}
# 数据集全部大小
sizeDic = {'gsdtsr': 12000, 'paintcontrol': 25595, 'iofrol': 32261, 'rails': 7814}
activations = {'sigmoid', 'guass1'}


def add_rank(filename, dataset, activation):
    # 读取原始数据
    df = pd.read_csv('./' + dataset + '_normal.csv', delim_whitespace=True)
    # 读取训练结果rank
    # rank = pd.read_csv('./' + dataset + '/rank/' + filename, header=None, squeeze=True)
    rank = pd.read_csv('/Users/sc/Desktop/untitled/' + activation + '/' + dataset + '/' + filename, header=None, squeeze=True)
    size = df.shape[0]
    print(size)
    # 调节index至训练集对应位置
    rank.index += size // 2 + 1
    # 取出训练集
    df = df[size // 2 + 1: size + 1]
    # 合并数据
    df.insert(5, 'Rank', rank)
    df['CycleNormal'] = df['CycleNormal'] * maxCycle[dataset]
    # 命令行输出
    print(rank)
    print(df)
    # 文本输出res
    df.to_csv('./' + activation + '/' + dataset + '/' + filename, sep=' ')


def contact_data(dataset, activation):
    # path = './' + dataset + '/rank'
    path = '/Users/sc/Desktop/untitled/' + activation + '/' + dataset
    save_folder = './' + activation + '/' + dataset
    files = os.listdir(path)
    print(files)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for file in files:
        add_rank(file, dataset, activation)


if __name__ == "__main__":
    for activation in activations:
        for dataset in datasets:
            contact_data(dataset, activation)

# contact_data('iofrol')
# contact_data('paintcontrol')
