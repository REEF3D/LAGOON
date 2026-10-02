import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Circle, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.colors import LinearSegmentedColormap

DARK='#2C4E7C'
GRAD=LinearSegmentedColormap.from_list('g',['#7ABBFB','#5B8EC5','#2F5282'])
FB=FontProperties(fname='/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf')

def text_path(s,x,y,size):
    p=TextPath((0,0),s,size=size,prop=FB)
    return p.transformed(Affine2D().translate(x,y))

def band_path(x0,x1,yc,A,t,lam,phase=0.0,n=400):
    x=np.linspace(x0,x1,n); s=A*np.sin(2*np.pi*(x-x0)/lam+phase)
    up=np.c_[x,yc+s+t/2]; lo=np.c_[x[::-1],(yc+s-t/2)[::-1]]
    v=np.r_[up,lo,up[:1]]; c=[Path.MOVETO]+[Path.LINETO]*(len(v)-2)+[Path.CLOSEPOLY]
    return Path(v,c)

def waves(ax,x0,x1,ytop,A,t,gap,lam,clip=None,phase=0.0,nb=3):
    paths=[band_path(x0,x1,ytop-i*gap,A,t,lam,phase) for i in range(nb)]
    comp=Path.make_compound_path(*paths)
    ylo=ytop-(nb-1)*gap-A-t; yhi=ytop+A+t
    im=ax.imshow(np.linspace(0,1,256)[:,None],cmap=GRAD,extent=(x0,x1,ylo,yhi),aspect='auto',origin='upper',zorder=2)
    pp=PathPatch(comp,transform=ax.transData,fc='none',ec='none'); ax.add_patch(pp); im.set_clip_path(pp)
    if clip is not None:
        # second clip: intersect by drawing the image inside clip only
        im.remove()
        im=ax.imshow(np.linspace(0,1,256)[:,None],cmap=GRAD,extent=(x0,x1,ylo,yhi),aspect='auto',origin='upper',zorder=2)
        cp=Path.make_compound_path(*[Path(np.r_[pv,pv[:1]]) for pv in []]) if False else None
        # clip bands to circle: intersect polygons numerically
        from matplotlib.path import Path as P
        cx,cy,r=clip
        th=np.linspace(0,2*np.pi,200); 
        bands=[]
        for bp in paths:
            poly=bp.vertices[:-1]
            bands.append(poly)
        # rasterised intersection: use clip on image with band compound, and clip bands patch with circle via separate axes trick
        circ=Circle((cx,cy),r,transform=ax.transData,fc='none',ec='none'); ax.add_patch(circ)
        # combine: draw image clipped to bands, then mask outside circle with white ring later
        im.set_clip_path(pp)
        return im,circ
    return im

def canvas(W,H):
    fig=plt.figure(figsize=(W/100,H/100),dpi=100); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.set_aspect('equal'); ax.axis('off')
    return fig,ax

def save(fig,name):
    fig.savefig(name+'.png',dpi=200,facecolor='white'); fig.savefig(name+'.svg',facecolor='white'); fig.savefig(name+'.pdf',facecolor='white'); plt.close(fig)

LIGHT=LinearSegmentedColormap.from_list('l',['#CFE7FF','#8CC4FB','#6A9FD6'])
def band_in_circle(cx,cy,ri,yc,A,t,lam,phase,n=600):
    x=np.linspace(cx-ri,cx+ri,n); s=A*np.sin(2*np.pi*(x-(cx-ri))/lam+phase)
    h=np.sqrt(np.clip(ri**2-(x-cx)**2,0,None))
    yu=np.minimum(yc+s+t/2,cy+h); yl=np.maximum(yc+s-t/2,cy-h); ok=yu>yl
    polys=[];i=0
    while i<n:
        if ok[i]:
            j=i
            while j<n and ok[j]: j+=1
            xs=x[i:j]; polys.append(np.r_[np.c_[xs,yu[i:j]],np.c_[xs[::-1],yl[i:j][::-1]]]); i=j
        else: i+=1
    return polys

def atoll(ax,cx,cy,R,w,color=DARK,fill='white',cmap=None):
    ri=R-w; cmap=GRAD if cmap is None else cmap
    if fill!='none': ax.add_patch(Circle((cx,cy),ri+1,fc=fill,ec='none',zorder=6))
    lam=ri*1.55; A=ri*0.10; t=ri*0.26; gap=ri*0.50
    polys=[]
    for i in range(3): polys+=band_in_circle(cx,cy,ri,cy+gap-i*gap,A,t,lam,phase=0.9)
    v=[];c=[]
    for q in polys:
        q=np.r_[q,q[:1]]; v+=list(q); c+=[Path.MOVETO]+[Path.LINETO]*(len(q)-2)+[Path.CLOSEPOLY]
    comp=Path(v,c)
    im=ax.imshow(np.linspace(0,1,256)[:,None],cmap=cmap,extent=(cx-ri,cx+ri,cy-ri,cy+ri),aspect='auto',origin='upper',zorder=7)
    pp=PathPatch(comp,transform=ax.transData,fc='none',ec='none'); ax.add_patch(pp); im.set_clip_path(pp)
    th=np.linspace(0,2*np.pi,361)
    ring=Path(np.r_[np.c_[cx+R*np.cos(th),cy+R*np.sin(th)],np.c_[cx+ri*np.cos(th[::-1]),cy+ri*np.sin(th[::-1])]],[Path.MOVETO]+[Path.LINETO]*360+[Path.MOVETO]+[Path.LINETO]*360)
    ax.add_patch(PathPatch(ring,fc=color,ec='none',zorder=8))

SIZE=190  # font size -> cap height ~ 0.72*SIZE
# ---------- word metrics
full=text_path('LAGOON',0,0,SIZE); ext=full.get_extents()
capH=ext.y1-ext.y0

def concept_family(name,atoll_o=False):
    W,H=1000,540
    fig,ax=canvas(W,H)
    x0,y0=40,300
    p=text_path('LAGOON',x0,y0,SIZE)
    if atoll_o:
        polys=p.to_polygons()
        eLAG=text_path('LAG',x0,y0,SIZE).get_extents(); eLAGO=text_path('LAGO',x0,y0,SIZE).get_extents()
        keep=[];O=[]
        for q in polys:
            cxq=q[:,0].mean()
            (O if eLAG.x1< cxq < eLAGO.x1 else keep).append(q)
        v=[];c=[]
        for q in keep:
            v+=list(q); c+=[Path.MOVETO]+[Path.LINETO]*(len(q)-2)+[Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(v,c),fc=DARK,ec='none',zorder=5))
        allO=np.vstack(O); ox0,ox1=allO[:,0].min(),allO[:,0].max(); oy0,oy1=allO[:,1].min(),allO[:,1].max()
        R=max(ox1-ox0,oy1-oy0)/2; inner=min(O,key=lambda q:np.ptp(q[:,0])); w=((ox1-ox0)-np.ptp(inner[:,0]))/2
        atoll(ax,(ox0+ox1)/2,(oy0+oy1)/2,R,w*1.0)
    else:
        ax.add_patch(PathPatch(p,fc=DARK,ec='none',zorder=5))
    e=p.get_extents()
    # waves under the word
    wx1=x0+ (e.x1-x0)*0.66
    waves(ax,x0,wx1,y0-60,A=12,t=30,gap=66,lam=(wx1-x0)/3.05)
    tp=text_path('open-source',wx1+26,y0-110,50); ax.add_patch(PathPatch(tp,fc=DARK,ec='none'))
    tp=text_path('visualization',wx1+26,y0-172,50); ax.add_patch(PathPatch(tp,fc=DARK,ec='none'))
    save(fig,name)

def concept_box(name):
    W,H=1200,560
    fig,ax=canvas(W,H)
    x0,y0=40,300
    p=text_path('LAGOON',x0,y0,SIZE); e=p.get_extents()
    ax.add_patch(PathPatch(p,fc=DARK,ec='none',zorder=5))
    # box like the REEF3D '3D' box
    bx0=e.x1+22; bh=capH*1.55; by0=y0-capH*0.30; bw=bh
    ax.add_patch(Rectangle((bx0,by0),bw,bh,fc=DARK,ec='none',zorder=1))
    cx,cy=bx0+bw/2,by0+bh/2
    atoll(ax,cx,cy,R=bh*0.38,w=bh*0.07,color='white',fill=DARK,cmap=LIGHT)
    waves(ax,x0,e.x1,y0-60,A=12,t=30,gap=66,lam=(e.x1-x0)/4.05)
    tp=text_path('open-source',bx0+4,y0-150,44); ax.add_patch(PathPatch(tp,fc=DARK,ec='none'))
    tp=text_path('visualization',bx0+4,y0-205,44); ax.add_patch(PathPatch(tp,fc=DARK,ec='none'))
    save(fig,name)

def icon(name):
    W=H=512
    fig,ax=canvas(W,H)
    ax.add_patch(Rectangle((0,0),W,H,fc=DARK,ec='none'))
    atoll(ax,W/2,H/2,R=190,w=34,color='white',fill=DARK,cmap=LIGHT)
    save(fig,name)

def icon_light(name):
    W=H=512
    fig,ax=canvas(W,H)
    atoll(ax,W/2,H/2,R=220,w=44,color=DARK,fill='white')
    save(fig,name)

concept_family('A_family')
concept_family('B_atoll_O',atoll_o=True)
concept_box('C_boxed')
icon('icon_dark'); icon_light('icon_light')
print('done')

def concept_wordmark(name):
    m=40
    p0=text_path('LAGOON',0,0,SIZE); e0=p0.get_extents()
    bh=capH*1.55; bw=bh
    by_off=-(bh-capH)/2+e0.y0
    W=int((e0.x1-e0.x0)+22+bw+2*m); ylo=by_off; yhi=by_off+bh; H=int(yhi-ylo+2*m)
    fig,ax=canvas(W,H)
    x0=m-e0.x0; y0=m-ylo
    p=text_path('LAGOON',x0,y0,SIZE); e=p.get_extents()
    ax.add_patch(PathPatch(p,fc=DARK,ec='none',zorder=5))
    bx0=e.x1+22; by0=y0+by_off
    ax.add_patch(Rectangle((bx0,by0),bw,bh,fc=DARK,ec='none',zorder=1))
    atoll(ax,bx0+bw/2,by0+bh/2,R=bh*0.38,w=bh*0.07,color='white',fill=DARK,cmap=LIGHT)
    save(fig,name)

concept_wordmark('LAGOON_logo')

HGRAD=LinearSegmentedColormap.from_list('h',['#2F5282','#5B8EC5','#7ABBFB'])
def concept_wordmark_grad(name):
    m=40
    p0=text_path('LAGOON',0,0,SIZE); e0=p0.get_extents()
    bh=capH*1.55; bw=bh
    by_off=-(bh-capH)/2+e0.y0
    W=int((e0.x1-e0.x0)+22+bw+2*m); ylo=by_off; yhi=by_off+bh; H=int(yhi-ylo+2*m)
    fig,ax=canvas(W,H)
    x0=m-e0.x0; y0=m-ylo
    p=text_path('LAGOON',x0,y0,SIZE); e=p.get_extents()
    im=ax.imshow(np.linspace(0,1,1024)[None,:],cmap=HGRAD,extent=(e.x0,e.x1,e.y0-2,e.y1+2),aspect='auto',zorder=5,interpolation='bicubic')
    pp=PathPatch(p,transform=ax.transData,fc='none',ec='none'); ax.add_patch(pp); im.set_clip_path(pp)
    bx0=e.x1+22; by0=y0+by_off
    ax.add_patch(Rectangle((bx0,by0),bw,bh,fc=DARK,ec='none',zorder=1))
    atoll(ax,bx0+bw/2,by0+bh/2,R=bh*0.38,w=bh*0.07,color='white',fill=DARK,cmap=LIGHT)
    save(fig,name)

concept_wordmark_grad('LAGOON_logo_gradient')

HGRAD_R=LinearSegmentedColormap.from_list('hr',['#7ABBFB','#5B8EC5','#2F5282'])
_hg=HGRAD
HGRAD=HGRAD_R
concept_wordmark_grad('LAGOON_logo_gradient_reversed')
HGRAD=_hg
