#!/usr/bin/env python3
"""Generate 3,000 hand-drawn-style annotated PaperPhysics scenes (V3).

Features:
  * Inclined-plane blocks are rotated so their bottom edge lies on the ramp.
  * Blocks on horizontal ground lie flat.
  * Contextual labels: theta by ramp corner; speed by arrow; mass by body;
    friction by ramp; spring constant by spring; restitution by target.
  * Rich polygon/edge/scene/text JSONL + YOLO box labels.

Dependencies: Pillow only. Python 3.10+.
Usage: python paperphysics_generate_v3.py --count 3000 --output paperphysics_v3
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASSES = ["ball", "block", "ramp", "spring"]
CASES = [("ball", "plain", None), ("block", "plain", None),
         ("ball", "spring", None), ("block", "spring", None),
         ("ball", "collision", "ramp"), ("ball", "collision", "flat"),
         ("block", "collision", "ramp"), ("block", "collision", "flat")]


def pick_font(px: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, px)
        except (OSError, ValueError):
            pass
    return ImageFont.load_default()


def plus(a, b):
    return (a[0]+b[0], a[1]+b[1])


def times(k, v):
    return (k*v[0],k*v[1])


def bounds(points, W, H, pad=2):
    return [round(max(0,min(p[0] for p in points)-pad),2),
            round(max(0,min(p[1] for p in points)-pad),2),
            round(min(W-1,max(p[0] for p in points)+pad),2),
            round(min(H-1,max(p[1] for p in points)+pad),2)]


def intersects(a, b, padding=0):
    return (a[0] < b[2]+padding and a[2] > b[0]-padding and
            a[1] < b[3]+padding and a[3] > b[1]-padding)


def line(draw, points, ink, width, rng, noise=.8):
    modified=[(round(x+rng.gauss(0,noise)),round(y+rng.gauss(0,noise))) for x,y in points]
    draw.line(modified,fill=ink,width=width,joint="curve")


def create_one(i, which, out, seed, width=768, height=576):
    body_kind, scene, target_surface=which
    rng=random.Random(seed+i*10103)
    sx,sy=width/768,height/576
    xy=lambda x,y:(x*sx,y*sy)
    image=Image.new("RGB",(width,height),rng.choice([
        (255,255,255), (252,251,247), (250,252,255), (249,249,246)]))
    draw=ImageDraw.Draw(image)
    ink=rng.choice([(23,29,36),(37,42,54),(38,38,38)])
    stroke=max(2,round(rng.randint(2,4)*sx))
    font=pick_font(round(rng.randint(18,21)*sx))
    ground_y=rng.randint(424,472)*sy
    angle_deg=rng.randint(18,45)
    theta=math.radians(angle_deg)
    side=rng.choice([-1,1])
    ramp_length=rng.randint(330,390)*sx
    low_x=rng.randint(175,220)*sx if side==1 else rng.randint(548,593)*sx
    low=(low_x,ground_y)
    up=(side*math.cos(theta),-math.sin(theta))
    outward=(-side*math.sin(theta),-math.cos(theta))
    high=plus(low,times(ramp_length,up))
    flat_end=(50*sx,ground_y) if side==1 else (718*sx,ground_y)
    objects=[]
    labels=[]
    arrows=[]
    occupied=[]

    def on_ramp(t,offset=0):
        return plus(plus(low,times(t*ramp_length,up)),times(offset,outward))

    def object_info(cls, role, points, **kwargs):
        obj_id=f"{role}_{len(objects)}"
        obj={"id":obj_id,"class":cls,"role":role,
             "polygon_xy":[[round(x,2),round(y,2)] for x,y in points],
             "bbox_xyxy":bounds(points,width,height),**kwargs}
        objects.append(obj)
        if cls!="ramp":occupied.append(obj["bbox_xyxy"])
        return obj

    # Draw inclined plane and floor. The actual tilted segment is annotated.
    triangle=[low,high,(high[0],ground_y)]
    draw.polygon(triangle,fill=rng.choice([(245,246,245),(249,248,245),(241,244,246)]))
    line(draw,[low,high],ink,stroke,rng)
    line(draw,[high,(high[0],ground_y)],ink,stroke,rng)
    line(draw,[(high[0],ground_y),low],ink,stroke,rng)
    line(draw,[flat_end,low],ink,stroke,rng)
    ramp=object_info("ramp","ramp",triangle,
        low_point_xy=[round(v,2) for v in low], high_point_xy=[round(v,2) for v in high],
        angle_degrees=angle_deg)

    def draw_body(shape, role, t=None, surface="ramp", radius=20, flat_offset=70):
        rr=radius*sx
        if surface=="ramp":
            contact=on_ramp(t)
            tangent=up
            norm=outward
        else:
            contact=(low[0]-side*flat_offset*sx,ground_y)
            tangent=(1,0)
            norm=(0,-1)
        if shape=="block":
            half_w=rr*1.16
            tall=rr*1.7
            # Base edge is exactly on surface; edges use the surface tangent.
            bottom_left=plus(contact,times(-half_w,tangent))
            bottom_right=plus(contact,times(half_w,tangent))
            top_right=plus(bottom_right,times(tall,norm))
            top_left=plus(bottom_left,times(tall,norm))
            vertices=[bottom_left,bottom_right,top_right,top_left]
            line(draw,vertices+[vertices[0]],ink,stroke,rng,.3)
            center=plus(contact,times(tall/2,norm))
            obj=object_info(shape,role,vertices,support_surface=surface,
                contact_edge_xy=[[round(x,2),round(y,2)] for x,y in (bottom_left,bottom_right)],
                contact_point_xy=[round(x,2) for x in contact],
                bottom_edge_parallel_to_surface=True,
                block_angle_radians=round(math.atan2(tangent[1],tangent[0]),6),
                center_xy=[round(x,2) for x in center])
        else:
            center=plus(contact,times(rr,norm))
            box=[center[0]-rr,center[1]-rr,center[0]+rr,center[1]+rr]
            draw.ellipse(tuple(round(x) for x in box),outline=ink,width=stroke)
            obj=object_info(shape,role,[(box[0],box[1]),(box[2],box[1]),
                                            (box[2],box[3]),(box[0],box[3])],
                support_surface=surface,
                contact_point_xy=[round(x,2) for x in contact],
                center_xy=[round(x,2) for x in center], radius_px=round(rr,2))
        return obj,center

    src_t=rng.uniform(.77,.83)
    primary,src_center=draw_body(body_kind,"primary",src_t,radius=rng.randint(18,24))
    second=None;second_center=None
    if scene=="collision":
        second_kind=rng.choice(["ball","block"])
        second,second_center=draw_body(second_kind,"collision_target",
            t=rng.uniform(.22,.29) if target_surface=="ramp" else None,
            surface=target_surface,radius=rng.randint(17,22),flat_offset=rng.randint(70,102))

    spring=None
    if scene=="spring":
        coil=[]
        t0,t1=.25,.51
        for step in range(15):
            alpha=step/14
            displacement=(7*sx if step%2 else -7*sx) if 0<step<14 else 0
            coil.append(on_ramp(t0+(t1-t0)*alpha,9*sx+displacement))
        line(draw,coil,ink,max(2,stroke-1),rng,.4)
        line(draw,[on_ramp(t0,-15*sx),on_ramp(t0,27*sx)],ink,stroke,rng,.3)
        spring=object_info("spring","spring",coil,
            anchor_xy=[round(q,2) for q in coil[0]],free_end_xy=[round(q,2) for q in coil[-1]])

    # Draw an angle arc centred on the lower vertex, inside the wedge.
    arc=[plus(low,xy(side*40*math.cos(a*theta/16),-40*math.sin(a*theta/16))) for a in range(17)]
    line(draw,arc,ink,max(2,stroke-1),rng,.25)

    v0=rng.choice([0,1,2,3,4,5])
    motion="stationary" if v0==0 else rng.choices(["down_slope","up_slope"],[.8,.2])[0]
    if v0:
        vec=times(-1 if motion=="down_slope" else 1,up)
        start=plus(src_center,times(43*sx,outward))
        start=plus(start,times(-20*sx,vec))
        tip=plus(start,times(68*sx,vec))
        line(draw,[start,tip],ink,stroke,rng,.25)
        for sign in (-1,1):
            wing=plus(tip,plus(times(-12*sx,vec),times(sign*7*sx,(-vec[1],vec[0]))))
            line(draw,[tip,wing],ink,stroke,rng,.25)
        arrows.append({"id":"initial_velocity","related_object_id":primary["id"],
            "start_xy":[round(x,2) for x in start],"tip_xy":[round(x,2) for x in tip],
            "direction":motion,"speed_m_s":v0})
        occupied.append(bounds([start,tip],width,height,7*sx))

    mass1=round(rng.uniform(.5,5),1)
    mu=round(rng.uniform(.02,.38),2)
    m2=round(rng.uniform(.5,5),1) if scene=="collision" else None
    e=round(rng.uniform(.2,1),2) if scene=="collision" else None
    k=rng.choice([40,60,80,100,120,150,200]) if scene=="spring" else None
    x0=rng.choice([0,.05,.1,.15,.2]) if scene=="spring" else None
    detail=rng.choices(["geometry_only","core","full"],[.10,.62,.28])[0]
    physics={"angle_degrees":angle_deg,"mass1_kg":mass1,"initial_speed_m_s":v0,
             "motion_direction":motion,"kinetic_friction_coefficient":mu,
             "gravity_m_s2":9.81,"mass2_kg":m2,"target_initial_speed_m_s":0.0 if m2 else None,
             "restitution":e,"spring_constant_N_m":k,"spring_compression_m":x0}

    def maybe_write(key, value, text, related, anchor_kind, candidates):
        if detail=="geometry_only":return
        # Layout positions supplied in pixels; keep text in bounds and prefer
        # candidates not overlapping prior text or object contours.
        best=None;best_penalty=1e9;chosen_display=None
        # Nearby fallback positions allow a label to remain associated with
        # its object even when its first preferred position is occupied.
        extras=[xy(-100,-35),xy(100,-35),xy(0,-75),xy(-95,35),
                xy(95,35),xy(-135,-70),xy(135,-70),xy(0,70)]
        positions=list(candidates)+[plus(c,extra) for c in candidates[:2] for extra in extras]
        for candidate in positions:
            for candidate_text in (text,text.replace("θ","theta").replace("μ","mu").replace("°","deg")):
                try:
                    metrics=draw.textbbox((0,0),candidate_text,font=font)
                    break
                except UnicodeEncodeError:
                    continue
            rect=[candidate[0],candidate[1],
                  candidate[0]+metrics[2]-metrics[0],candidate[1]+metrics[3]-metrics[1]]
            if not (3<rect[0] and rect[2]<width-3 and 3<rect[1] and rect[3]<height-3):
                continue
            overlap=sum(3 if intersects(rect,b,9*sx) else 0 for b in occupied)
            overlap+=sum(9 if intersects(rect,L["bbox_xyxy"],12*sx) else 0 for L in labels)
            cx=(rect[0]+rect[2])/2
            cy=(rect[1]+rect[3])/2
            if min(low[0],high[0]) < cx < max(low[0],high[0]):
                on_slope_y=low[1]+(cx-low[0])*(high[1]-low[1])/(high[0]-low[0])
                if abs(cy-on_slope_y)<28*sy:overlap+=5
            if rect[1]<ground_y+4*sy and rect[3]>ground_y-4*sy:
                overlap+=5
            if overlap<best_penalty:
                best_penalty=overlap;best=rect;chosen_display=candidate_text
            if overlap==0:break
        # An omitted label is preferable to a misleading overlap.
        if best is None or best_penalty>0:return
        draw.text((round(best[0]),round(best[1])),chosen_display,font=font,fill=ink)
        labels.append({"parameter_key":key,"value":value,"text":chosen_display,
                       "related_object_id":related,"placement":anchor_kind,
                       "bbox_xyxy":[round(q,2) for q in best]})

    if detail!="geometry_only":
        angle_place=on_ramp(.19,-31*sx)
        maybe_write("angle_degrees",angle_deg,f"θ={angle_deg}°",ramp["id"],"angle_at_corner",[
            plus(angle_place,xy(-22,-13)),plus(angle_place,xy(-15,16)),
            on_ramp(.26,-38*sx)])

        body_out=plus(src_center,times(65*sx,outward))
        maybe_write("mass1_kg",mass1,f"m1={mass1:g}kg",primary["id"],"next_to_primary",[
            plus(body_out,times(65*sx,up)),plus(body_out,times(-70*sx,up)),
            plus(src_center,times(106*sx,outward))])

        if arrows:
            arrow=arrows[0]
            tip=tuple(arrow["tip_xy"])
            maybe_write("initial_speed_m_s",v0,f"v0={v0:g}m/s",arrow["id"],"next_to_arrow",[
                plus(tip,times(47*sx,outward)),plus(tip,times(77*sx,outward)),
                plus(tuple(arrow["start_xy"]),times(65*sx,outward))])
        elif detail=="full":
            maybe_write("initial_speed_m_s",v0,"v0=0",primary["id"],"next_to_primary",[
                plus(src_center,times(80*sx,outward)),plus(src_center,times(-75*sx,up))])

        if detail=="full" or rng.random()<.66:
            # Friction belongs to the ramp; place below its slope in the triangle.
            center_ramp=on_ramp(.54,-44*sx)
            maybe_write("kinetic_friction_coefficient",mu,f"μ={mu:.2f}",ramp["id"],"under_ramp",[
                plus(center_ramp,xy(-20,0)),on_ramp(.65,-68*sx),on_ramp(.43,-56*sx)])

        if spring is not None:
            maybe_write("spring_constant_N_m",k,f"k={k}N/m",spring["id"],"next_to_spring",[
                on_ramp(.4,55*sx),on_ramp(.37,-65*sx),on_ramp(.35,85*sx)])
            if detail=="full":
                maybe_write("spring_compression_m",x0,f"x0={x0:g}m",spring["id"],"next_to_spring",[
                    on_ramp(.31,-80*sx),on_ramp(.32,91*sx)])
        if second is not None:
            second_pos=tuple(second["center_xy"])
            maybe_write("mass2_kg",m2,f"m2={m2:g}kg",second["id"],"next_to_collision_target",[
                plus(second_pos,times(42*sx,outward)) if target_surface=="ramp" else plus(second_pos,xy(-25,-57)),
                plus(second_pos,xy(-35,-68)),plus(second_pos,xy(20,46))])
            maybe_write("restitution",e,f"e={e:.2f}",second["id"],"collision_parameter_near_target",[
                plus(second_pos,xy(-22,65)),plus(second_pos,xy(30,-75)),
                plus(second_pos,xy(-30,-96))])
            if detail=="full":
                maybe_write("target_initial_speed_m_s",0,"v2=0",second["id"],"next_to_collision_target",[
                    plus(second_pos,xy(-45,95)),plus(second_pos,xy(40,57))])
        if detail=="full":
            maybe_write("gravity_m_s2",9.81,"g=9.81m/s²",None,"global",[
                xy(22,22),xy(550,22),xy(290,25)])

    # Visibility is recorded per parameter, distinguishing underlying
    # generated physics values from text actually shown on the image.
    visible={label["parameter_key"] for label in labels}
    for key,value in physics.items():
        if isinstance(value,(int,float)):
            physics[key]={"value":value,"shown_on_image":key in visible}

    name=f"paperphysics_v3_{i:05d}.png"
    image.save(out/"images"/name,optimize=False)
    yolo=[]
    for obj in objects:
        x1,y1,x2,y2=obj["bbox_xyxy"]
        yolo.append(f"{CLASSES.index(obj['class'])} {(x1+x2)/(2*width):.6f} "
                    f"{(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
    (out/"labels"/name.replace(".png",".txt")).write_text("\n".join(yolo)+"\n",encoding="utf-8")
    return {"image":f"images/{name}","width":width,"height":height,
            "scene_type":scene,"primary_type":body_kind,"collision_target_surface":target_surface,
            "collision_target_type":second["class"] if second else None,
            "ramp_rises":"right" if side==1 else "left",
            "detail_level":detail,"objects":objects,"physics":physics,
            "text_annotations":labels,"arrows":arrows,"ground_y_px":ground_y,
            "generator_seed":seed+i*10103}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--count",type=int,default=3000)
    ap.add_argument("--seed",type=int,default=20261107)
    ap.add_argument("--output","--out",default="paperphysics_v3")
    ap.add_argument("--width",type=int,default=768)
    ap.add_argument("--height",type=int,default=576)
    args=ap.parse_args()
    if args.count<=0 or min(args.width,args.height)<512:
        ap.error("count must be >0, width and height must each be >=512")
    root=Path(args.output)
    (root/"images").mkdir(parents=True,exist_ok=True)
    (root/"labels").mkdir(parents=True,exist_ok=True)
    distribution=Counter()
    grouping=defaultdict(list)
    with (root/"annotations.jsonl").open("w",encoding="utf-8") as file:
        for i in range(args.count):
            # Round-robin of all eight classes: for 3000, exactly 375 each.
            combo=CASES[i%8]
            result=create_one(i,combo,root,args.seed,args.width,args.height)
            file.write(json.dumps(result,ensure_ascii=False)+"\n")
            key="_".join(str(x) for x in combo if x is not None)
            distribution[key]+=1
            grouping[key].append(result["image"])

    # Stratified, reproducible splits; unlike index-modulo, these do not
    # accidentally correlate labels/detail level with the split.
    assignments=defaultdict(list)
    for key,files in grouping.items():
        random.Random(args.seed+sum(map(ord,key))*307).shuffle(files)
        total=len(files); ntrain=round(total*.8); nval=round(total*.1)
        assignments["train"].extend(files[:ntrain])
        assignments["val"].extend(files[ntrain:ntrain+nval])
        assignments["test"].extend(files[ntrain+nval:])
    for name,paths in assignments.items():
        random.Random(args.seed+len(name)).shuffle(paths)
        (root/f"{name}.txt").write_text("\n".join(paths)+"\n",encoding="utf-8")
    (root/"classes.txt").write_text("\n".join(CLASSES)+"\n",encoding="utf-8")
    (root/"dataset.yaml").write_text(
        "path: .\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        "names: {0: ball, 1: block, 2: ramp, 3: spring}\n",encoding="utf-8")
    manifest={"dataset":"PaperPhysics V3","num_images":args.count,
              "image_size":[args.width,args.height],"seed":args.seed,
              "classes":CLASSES,"distribution":dict(distribution),
              "splits":{k:len(v) for k,v in assignments.items()}}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=="__main__":
    main()
