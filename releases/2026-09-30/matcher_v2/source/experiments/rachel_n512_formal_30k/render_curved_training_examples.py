"""Static scientific mask panels: original versus real-outline partial deletion.

Figure contract: inspect shape realism, not accuracy. First accepted TRAIN
examples in source order; two side-by-side GT placements use identical axes.
Blue A / gold B plus direct labels; absent material white, old outline dashed.
800px model-frame geometry, B offset=-translation_rc. No predicted layout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

BLUE, GOLD = "#3584BC", "#D5A031"


def unpack(archive, key):
    return np.unpackbits(archive[key], axis=1).astype(bool)


def draw(ax, mask, offset, color, *, outline=False):
    rows, cols = mask.shape
    r,c = offset
    if not outline:
        rgba = np.zeros((*mask.shape,4))
        rgba[mask] = to_rgba(color)
        ax.imshow(rgba, extent=(c-.5,c+cols-.5,r+rows-.5,r-.5), interpolation="nearest")
    ax.contour(np.arange(cols)+c,np.arange(rows)+r,mask.astype(float),[.5],
        colors=["#555555" if outline else color],linewidths=.7,
        linestyles="--" if outline else "-",zorder=4)


def render_positive(path, output, *, original_keys=("original_a","original_b"),
                    partial_keys=("curve_a","curve_b")):
    with np.load(path,allow_pickle=False) as data:
        old=[unpack(data,k) for k in original_keys]
        new=[unpack(data,k) for k in partial_keys]
        translation=np.asarray(data["original_translation_rc"],float)
    offsets=[np.zeros(2),-translation]
    points=np.concatenate([np.argwhere(mask)+offset for mask,offset in zip(old,offsets)])
    low,high=points.min(0)-15,points.max(0)+15
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.2))
    for ax,masks,title in zip(axes,(old,new),("Original pair · GT placement","Curved partial pair · same GT")):
        for mask,offset,color in zip(masks,offsets,(BLUE,GOLD)):
            draw(ax,mask,offset,color)
        if masks is new:
            for original,current,offset in zip(old,new,offsets):
                if not np.array_equal(original,current):
                    draw(ax,original,offset,"#555555",outline=True)
        ax.set_xlim(low[1],high[1]); ax.set_ylim(high[0],low[0]); ax.set_aspect("equal")
        ax.set_title(title,fontsize=11); ax.set_xlabel("x (model-frame px)")
        ax.set_ylabel("y (model-frame px)")
        for name in ("top","right"):
            ax.spines[name].set_visible(False)
    fig.suptitle("Real-outline partial-seam training example",fontsize=14,x=.5,y=.99)
    fig.text(.5,.02,"Blue = A   |   Gold = B   |   Dashed = pre-cut outline   |   White = absent material\nTRAIN augmentation only; no model prediction. Weathering is applied separately during training.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.18,1,.94))
    fig.savefig(output,dpi=160,facecolor="white"); plt.close(fig)


def render_negative(path, output):
    """Two separately positioned canvases; solid dark line marks mined arcs."""
    metadata=json.loads(Path(path).with_suffix(".json").read_text())
    evidence=metadata["overlay_entry"]["geometry"]
    with np.load(path,allow_pickle=False) as data:
        masks=[unpack(data,"replacement_original_"+s) for s in "ab"]
        points=[data["replacement_original_points_"+s] for s in "ab"]
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.6))
    for ax,mask,contour,side,color in zip(axes,masks,points,"ab",(BLUE,GOLD)):
        draw(ax,mask,np.zeros(2),color)
        arc=evidence["fragment_"+side+"_arc"]
        ids=(arc["start_token"]+np.arange(arc["token_count"])) % len(contour)
        ax.plot(contour[ids,1],contour[ids,0],color="#222222",linewidth=2.4)
        ax.set_xlim(0,800); ax.set_ylim(800,0); ax.set_aspect("equal")
        ax.set_title("Fragment "+side.upper()+" · independent canvas",fontsize=11)
        ax.set_xlabel("x (model-frame px)"); ax.set_ylabel("y (model-frame px)")
    fig.suptitle("Nonadjacent TRAIN pair with opposing straight edges",fontsize=14,y=.99)
    fig.text(.5,.02,"Label = negative (original nonadjacency annotation)   |   Black = mined straight arc\nSeparate canvases; no joining transform or model prediction is shown.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.94))
    fig.savefig(output,dpi=160,facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--negative",action="store_true")
    args=parser.parse_args()
    (render_negative if args.negative else render_positive)(Path(args.input),Path(args.output))
