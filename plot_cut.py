##
import pandas as pd 



csv_path = f'/home/florian/data/These/Simulation/Sph3D_CME/z10_grad_012/'
df_along = pd.read_csv(f'{csv_path}/along_200.csv')

df_perp = pd.read_csv(f'{csv_path}/perp_200.csv')

# df.columns
# 'B:0', 'B:1', 'B:2', 'B_cart:0', 'B_cart:1', 'B_cart:2', 'prs', 'rho',
# 'v:0', 'v:1', 'v:2', 'vtkValidPointMask', 'arc_length', 'Points:0','Points:1', 'Points:2'


y = df_along['Points:1']
rho_a = df_along['rho']

# this axis is inversed
rho_p = df_perp['rho'][::-1]

fig,ax = plt.subplots(2)

ax[0].plot(y,rho_a,color='green',label='along perpendicular direction')
ax[0].set_yscale('log')
ax[0].set_xlim(2.5,20)
ax[0].legend()
# ax[0].set_ylim(6e-6,3e-4)
ax[1].plot(y,rho_p,color='red',label='perpendicular')
ax[1].set_yscale('log')
ax[1].set_xlim(2.5,20)
# ax[1].set_ylim(2e-4,2e-6)
ax[1].legend()
fig.suptitle('Density cross along 2 trajectories')
# fig.tight_layout()
##
