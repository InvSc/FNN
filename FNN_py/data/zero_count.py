import matplotlib as mlp
import pandas as pd
import numpy as np
dataset = 'rails'
df = pd.read_csv('./' + dataset + '.csv', sep=',', parse_dates=['LastRun'])
print(df)

df_normal = pd.read_csv('./' + dataset + '_normal.csv', sep=' ')
print(df_normal)

# Genarate ZeroCount. 统计倒数前n个 默认n=4
df['LastResults'] = df['LastResults'].astype(str)
n = 12
# dfZeCo = df[df['LastResults'].str.len() > 9]
def f(x):
    x = x[::-1]
    count = 0
    for i in range(min(len(x), 3 * n - 1)):
      if x[i] == '0':
        count += 1
    return count


df['ZeroCount'] = df['LastResults'].apply(f)
df_normal['ZeroCountNormal'] = df['ZeroCount'].apply(lambda x : x / n)
print(df_normal)

# 输出
df_normal.to_csv('./' + dataset + '_normal' + str(n) + '.csv', index=False, sep=' ')

