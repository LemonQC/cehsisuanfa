"""Plot actual simulated cube trajectories; never draw predicted successes."""
from pathlib import Path
import json
import numpy as np

def plot_gate_a(result_dir, out_file=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    result_dir=Path(result_dir)
    paths=sorted(result_dir.glob('bc_seed*_side*.json'))
    if not paths:raise FileNotFoundError(f'No actual gate-A physical trajectory traces in {result_dir}')
    fig,ax=plt.subplots(figsize=(8,6))
    for p in paths:
        data=json.loads(p.read_text())
        if not data:continue
        tr=np.asarray([x['cube'][:2] for x in data],dtype=float)
        side=1 if 'side1.json' in p.name else -1
        color='tab:blue' if side>0 else 'tab:orange'
        ax.plot(tr[:,0],tr[:,1],lw=1.5,color=color,alpha=.55,
                label=f'{p.stem} ({"check" if any(x["visited"] for x in data) else "no-check"})')
    ax.add_patch(Rectangle((-.05,-.077),.10,.154,color='gray',alpha=.38))
    for side,color in [(1,'tab:blue'),(-1,'tab:orange')]:
        ax.add_patch(Circle((0,side*.155),.065,fill=False,ls='--',color=color))
    ax.scatter([-.145,.145],[0,0],c=['black','green'],s=70,zorder=4)
    ax.set_aspect('equal');ax.set_xlim(-.24,.24);ax.set_ylim(-.27,.27)
    ax.set_xlabel('cube x (m)');ax.set_ylabel('cube y (m)')
    ax.set_title('Measured physical cube trajectories in ManiSkill')
    ax.legend(loc='upper left',fontsize=5,ncol=2)
    fig.tight_layout()
    out_file=Path(out_file or result_dir/'gateA_physics_trajectories.png')
    fig.savefig(out_file,dpi=160);plt.close(fig)
    return out_file


def trajectory_video(cube_xyz, output_path, title='Measured physical cube XY (not RGB video)', fps=12):
    """Visualize recorded physical trajectory without any GPU/GLX/Vulkan API.

    This creates an illustrative 2D animation from *actual simulated cube poses*,
    NOT a robot camera recording and not evidence of success by itself.
    """
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    import imageio.v2 as imageio
    import numpy as np
    points=np.asarray(cube_xyz, dtype=float)
    if points.ndim!=2 or points.shape[0]<1 or points.shape[1]<2:
        raise ValueError('cube_xyz must be nonempty array [steps,>=2]')
    dest=Path(output_path);dest.parent.mkdir(parents=True,exist_ok=True)
    step=max(1,int(np.ceil(len(points)/250)))
    sample=np.arange(0,len(points),step,dtype=int)
    if sample[-1]!=len(points)-1:sample=np.r_[sample,len(points)-1]
    fig,ax=plt.subplots(figsize=(6,5))
    ax.set_xlim(-.30,.30); ax.set_ylim(-.28,.28);ax.set_aspect('equal')
    ax.add_patch(Rectangle((-.05,-.077),.10,.154,facecolor='gray',alpha=.5))
    for sign in (-1,1):
        ax.add_patch(Circle((0,sign*.155),.065,fill=False,ls='--',alpha=.5))
    ax.plot([-.145],[0.],marker='s',markersize=8,color='tab:blue')
    ax.plot([.145],[0.],marker='o',markersize=8,color='tab:green')
    ax.set_title(title,fontsize=9)
    ax.set_xlabel('x / m'); ax.set_ylabel('y / m')
    line,=ax.plot([],[],color='tab:orange',lw=2)
    tip,=ax.plot([],[],marker='s',color='tab:red')
    fig.tight_layout()
    with imageio.get_writer(dest,fps=fps,codec='libx264',macro_block_size=1) as writer:
        for idx in sample:
            line.set_data(points[:idx+1,0],points[:idx+1,1])
            tip.set_data([points[idx,0]],[points[idx,1]])
            fig.canvas.draw()
            rgba=np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:,:,:3].copy())
    plt.close(fig)
    return dest
