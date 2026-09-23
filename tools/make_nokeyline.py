# make_nokeyline.py — second sticker version: TRANSPARENT tight cut, no white keyline (owner 2026-09-23: "white border and transparent border,
# cutting close to the edge, no white"). Derived from pets/<Pet>_sticker.png so the art is pixel-identical:
#   keyline = near-white opaque pixels connected to the OUTSIDE (flood fill through a transparent pad), limited to a band of 6% of the size
#   (so white fur that touches the keyline keeps its body), art alpha = alpha minus keyline, 1 px clean-up, trim to bounds + 4 px.
import os, glob, sys, numpy as np, cv2
from PIL import Image
S=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SRC=f"{S}/pets"; OUT=f"{S}/pets_nokeyline"; os.makedirs(OUT,exist_ok=True)
names=sys.argv[1:]
for f in sorted(glob.glob(f"{SRC}/*_sticker.png")):
    n=os.path.basename(f).replace("_sticker.png","")
    if names and n not in names: continue
    im=np.array(Image.open(f).convert("RGBA")); rgb=im[:,:,:3].astype(np.int32); a=im[:,:,3]
    h,w=a.shape; pad=2
    op=np.pad(a>10,pad,constant_values=False); solid=np.pad(a>=128,pad,constant_values=False)
    wh=np.pad((rgb.min(2)>=230),pad,constant_values=False)&solid | (op&~solid)      # near-white solid pixels, plus the feathered fringe (low alpha, often dark) = walkable
    # flood the white band from outside: seed = transparent border, walkable = transparent OR (opaque & near-white)
    walk=((~op)|wh).astype(np.uint8); ff=np.zeros((walk.shape[0]+2,walk.shape[1]+2),np.uint8)
    cv2.floodFill(walk.copy(),ff,(0,0),2)                                     # ff marks reached pixels (offset by 1)
    reached=ff[1:-1,1:-1].astype(bool)
    keyline=reached&op                                                       # opaque near-white pixels reachable from outside = the die-cut band
    dist=cv2.distanceTransform(op.astype(np.uint8),cv2.DIST_L2,5)            # depth from the outer edge
    keyline&=dist<=0.06*max(h,w)                                              # never eat deeper than a plausible keyline
    art=op&~keyline; art=art[pad:-pad,pad:-pad].astype(np.uint8)*255
    art=cv2.morphologyEx(art,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n_,lab,st,_=cv2.connectedComponentsWithStats(art); keep=np.zeros_like(art)
    for i in range(1,n_):
        if st[i,cv2.CC_STAT_AREA]>=0.001*art.size: keep[lab==i]=255
    alpha=cv2.GaussianBlur(keep,(0,0),0.6)
    out=np.dstack([im[:,:,:3],alpha]); ys,xs=np.where(alpha>8)
    if len(xs)==0: print("EMPTY",n); continue
    m=4; y0,y1,x0,x1=max(0,ys.min()-m),min(h,ys.max()+m+1),max(0,xs.min()-m),min(w,xs.max()+m+1)
    Image.fromarray(out[y0:y1,x0:x1]).save(f"{OUT}/{n}_sticker.png",optimize=True)
    print(n,"keyline %.1f%% of opaque"%(100*keyline.sum()/max(1,op.sum())),(x1-x0,y1-y0),flush=True)
