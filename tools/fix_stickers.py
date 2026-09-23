# fix_stickers.py [--gen] [--cut] [names...] — rebuild clipped stickers from the COMPLETE canon dex art:
#   --gen : NBP recreates the character full-body with generous margin on a flat white background (2K), canon PNG as the ref
#   --cut : rembg alpha (single cartoon character on white = reliable) -> hole fill -> white die-cut keyline (2.5% of size) -> trim + margin
# Owner 2026-09-23: "a bunch are cut off" — 21 stickers had straight cut edges; the pipeline/allstickers copies of the same 21 are truncated downloads.
import sys, os, subprocess, numpy as np, cv2
from PIL import Image
S=os.path.dirname(os.path.abspath(__file__)); C="C:/Users/ADMIN/Desktop/EtherFantasy/etherfamily-pipeline/assets/canon"
FAL="C:/Users/ADMIN/Desktop/EtherFantasy/etherfamily-pipeline/scripts/fal_img.py"
GEN=f"{S}/_regen"; OUT=f"{S}/_fixed"; os.makedirs(GEN,exist_ok=True); os.makedirs(OUT,exist_ok=True)
BAD=["Alligwamp","Aphroxid","Arblizen","Armordigoal","Aromerita","Emperazor","Fantasnut","Felistar","Flairon","Fuenago","Gremin","Grubgas","Mytier","Occlusk","Opsidien","Pangrass","Pangrove","Vaudequin","Vernirox","Windora","Wrecktile"]
names=[a for a in sys.argv[1:] if not a.startswith("--")] or BAD
PROMPT=("Recreate this exact character as a clean sticker illustration: the WHOLE body fully visible with a generous empty margin on every side "
        "(nothing cropped at the edges, including wings, tails, horns, weapons and flames), centered, on a flat pure white background, crisp clean edges, "
        "the same design, colors, proportions, expression and pose as the reference, same illustration style, no shadow on the ground, no text, no watermark")
if "--gen" in sys.argv:
    env=dict(os.environ,EF_FAL_ENV="E:/ef-lora/.fal.env",FAL_AR="1:1",FAL_RES="2K",PYTHONIOENCODING="utf-8")
    for n in names:
        out=f"{GEN}/{n}.png"; ref=f"{C}/{n}.png"
        if os.path.exists(out): continue
        if not os.path.exists(ref): print("no canon for",n); continue
        for k in range(3):
            subprocess.run(["python",FAL,out,PROMPT,ref],env=env,capture_output=True,text=True)
            try: Image.open(out).load(); break
            except Exception:
                if os.path.exists(out): os.remove(out)
        print(n,"gen","OK" if os.path.exists(out) else "FAIL",flush=True)
if "--cut" in sys.argv:
    from rembg import remove, new_session
    sess=new_session("isnet-general-use") if False else new_session("u2net")
    for n in names:
        src=f"{GEN}/{n}.png"
        if not os.path.exists(src): print("skip",n); continue
        im0=Image.open(src).convert("RGBA"); P=int(0.05*max(im0.size))+16                                            # pad first: art touching the 2K canvas must still get a full keyline
        im=Image.new("RGBA",(im0.width+2*P,im0.height+2*P),(255,255,255,255)); im.paste(im0,(P,P))
        cut=remove(im,session=sess,alpha_matting=False)
        a=np.array(cut)[:,:,3]
        a=(a>60).astype(np.uint8)*255
        cnts,_=cv2.findContours(a,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(a,cnts,-1,255,cv2.FILLED)   # fill interior holes (eyes, white patches)
        n_,lab,st,_=cv2.connectedComponentsWithStats(a); keep=np.zeros_like(a)
        big=[i for i in range(1,n_) if st[i,cv2.CC_STAT_AREA]>=0.002*a.size]                                          # drop specks, keep detached parts (flames, floating bits)
        for i in big: keep[lab==i]=255
        a=keep
        h,w=a.shape; r=max(14,int(0.025*max(h,w))); ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)); ad=cv2.dilate(a,ker)
        ad=cv2.GaussianBlur(ad,(0,0),1.2)
        rgb=np.array(im)[:,:,:3].astype(np.float32); af=a.astype(np.float32)/255.0
        col=rgb*af[:,:,None]+255.0*(1-af[:,:,None])                                                                    # white keyline under the art
        out=np.dstack([np.clip(col,0,255).astype(np.uint8),ad])
        ys,xs=np.where(ad>8); m=12; y0,y1,x0,x1=max(0,ys.min()-m),min(h,ys.max()+m),max(0,xs.min()-m),min(w,xs.max()+m)
        Image.fromarray(out[y0:y1,x0:x1]).save(f"{OUT}/{n}_sticker.png",optimize=True); print(n,"cut",(x1-x0,y1-y0),"alpha %.1f%%"%(100*af.mean()),flush=True)
