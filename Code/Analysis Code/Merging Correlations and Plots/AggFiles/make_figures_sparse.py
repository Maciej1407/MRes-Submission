import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to input files (adjust if needed)
SPARE_RUN_FILE = 'collectedData/old_t50.csv'
AGG_FILE = 'collectedData/aggregated_intersection.csv'

# Output directory
SAVE_DIR = 'SparseFiguresColored'
os.makedirs(SAVE_DIR, exist_ok=True)


sp = pd.read_csv(SPARE_RUN_FILE)


def parse_params(params):
    fr_str, sr_str = params.split('+')
    fr_val = fr_str.split(':')[1]
    sr_val = sr_str.split(':')[1] if 'sR:' in sr_str else sr_str
    def to_float(x):
        try: return float(x)
        except: return np.nan
    return to_float(fr_val), to_float(sr_val)

sp[['fR_val','sR_val']] = sp['Params'].apply(lambda x: pd.Series(parse_params(x)))

# Classify rows into categories
def classify(row):
    fr = row['fR_val']; sr = row['sR_val']
    if pd.isna(sr):
        return 'Homogeneous'
    if fr == 0 or sr == 0:
        return 'Pair containing sensor'
    return 'Homogeneous' if fr == sr else 'Heterogeneous'

sp['category'] = sp.apply(classify, axis=1)

# Colour palette for Stage 1 categories
palette_stage1 = {
    'Homogeneous': 'blue',
    'Heterogeneous': 'orange',
    'Pair containing sensor': 'green'
}


plt.figure(figsize=(10,4))
cat_order = ['Homogeneous','Heterogeneous','Pair containing sensor']
plt.subplot(1,2,1)
sns.boxplot(x='category', y='BFS_Composite', data=sp, order=cat_order, palette=palette_stage1)
plt.title('Sparse sweep: BFS Composite by category'); plt.xlabel('Category'); plt.ylabel('BFS Composite'); plt.xticks(rotation=20)
plt.subplot(1,2,2)
sns.boxplot(x='category', y='Midpoint_Composite', data=sp, order=cat_order, palette=palette_stage1)
plt.title('Sparse sweep: Midpoint Composite by category'); plt.xlabel('Category'); plt.ylabel('Midpoint Composite'); plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'fig1_stage1_category_boxplot.png'), dpi=300)
plt.close()


agg = pd.read_csv(AGG_FILE)
agg['Diffusion_float'] = agg['Diffusion'].astype(float)

plt.figure(figsize=(10,4))

agg['D_str'] = agg['Diffusion_float'].map(lambda v: f"{v:g}")
order_D = sorted(agg['Diffusion_float'].unique())
order_D_str = [f"{d:g}" for d in order_D]


palette_D = dict(zip(order_D_str, sns.color_palette("tab10", n_colors=len(order_D_str))))

plt.subplot(1,2,1)
sns.boxplot(
    x='D_str', y='BFS',
    data=agg,
    order=order_D_str,
    palette=palette_D
)
plt.title('Focused sweep: BFS Composite by diffusion')
plt.xlabel('Diffusion'); plt.ylabel('BFS Composite')

plt.subplot(1,2,2)
sns.boxplot(
    x='D_str', y='Mid',
    data=agg,
    order=order_D_str,
    palette=palette_D
)
plt.title('Focused sweep: Midpoint Composite by diffusion')
plt.xlabel('Diffusion'); plt.ylabel('Midpoint Composite')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig2_stage2_diffusion_boxplot.png'), dpi=300)
plt.close()


agg['category'] = agg.apply(lambda row: 'Homogeneous' if row['fR_sorted'] == row['sR_sorted'] else 'Heterogeneous', axis=1)
palette_stage2 = {'Homogeneous': 'blue', 'Heterogeneous': 'orange'}
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
sns.boxplot(x='category', y='BFS', data=agg, order=['Homogeneous','Heterogeneous'], palette=palette_stage2)
plt.title('Focused sweep: BFS Composite by category'); plt.xlabel('Category'); plt.ylabel('BFS Composite')
plt.subplot(1,2,2)
sns.boxplot(x='category', y='Mid', data=agg, order=['Homogeneous','Heterogeneous'], palette=palette_stage2)
plt.title('Focused sweep: Midpoint Composite by category'); plt.xlabel('Category'); plt.ylabel('Midpoint Composite')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'fig3_stage2_category_boxplot.png'), dpi=300)
plt.close()

print(f"Figures saved to {SAVE_DIR}")


##Time 50 midpoint
plt.figure(figsize=(10,4))
cat_order = ['Homogeneous','Heterogeneous','Pair containing sensor']
plt.subplot(1,2,1)
sns.boxplot(x='category', y='BFS_Composite', data=sp, order=cat_order, palette=palette_stage1)
plt.title('Sparse sweep: BFS Composite by category'); plt.xlabel('Category'); plt.ylabel('BFS Composite'); plt.xticks(rotation=20)
plt.subplot(1,2,2)
sns.boxplot(x='category', y='Midpoint_Composite', data=sp, order=cat_order, palette=palette_stage1)
plt.title('Sparse sweep: Midpoint Composite by category'); plt.xlabel('Category'); plt.ylabel('Midpoint Composite'); plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'fig1_stage1_category_boxplot.png'), dpi=300)
plt.close()