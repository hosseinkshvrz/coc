import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../data/prs/contributors/activegraph_contributors.csv')
df['sum'] = df['n_pr_before_coc'] + df['n_pr_after_coc']
# df = df.sort_values('sum', ascending=False)

data = np.array(df[['n_pr_before_coc', 'n_pr_after_coc']])
# Create a figure
fig = plt.figure(figsize=(16, 6))

# Set the y position
x_pos = np.arange(data.shape[0])
x_pos = [x for x in x_pos]
# plt.xticks(x_pos, names, fontsize=10)

# Create a horizontal bar in the position y_pos
plt.bar(x_pos,
        # using x1 data
        data[:, 0],
        # that is centered
        align='center',
        # with alpha 0.4
        alpha=0.7,
        label='Before CoC')

# Create a horizontal bar in the position y_pos
plt.bar(x_pos,
        # using NEGATIVE x2 data
        -data[:, 1],
        # that is centered
        align='center',
        # with alpha 0.4
        alpha=0.7,
        label='After CoC')

# annotation and labels
plt.xlabel('Distinct Users')
plt.ylabel('# of Contributions')
t = plt.title('Contributions Before and After the CoC Adoption')
# plt.ylim([-1,len(x1)+0.1])
# plt.xlim([-max(x2)-10, max(x1)+10])
# plt.grid()
locs, labels = plt.yticks()
plt.yticks(locs, [str(abs(l)) for l in locs])

plt.legend(loc='upper right')

plt.savefig('../cont.png')
