"""Stream the source BVH into Character Studio names and uniform ZXY channels."""
import warnings
import numpy as np
from scipy.spatial.transform import Rotation


def bone_names():
    names={'Hips':'Hips','Spine':'Chest','Spine1':'Chest2','Spine2':'Chest3',
           'Neck':'Neck','Head':'Head'}
    for side in ('Left','Right'):
        for source,target in [('UpLeg','Hip'),('Leg','Knee'),('Foot','Ankle'),
                              ('ToeBase','Toe'),('Shoulder','Collar'),('Arm','Shoulder'),
                              ('ForeArm','Elbow'),('Hand','Wrist')]:
            names[side+source]=side+target
        for digit,finger in enumerate(('Thumb','Index','Middle','Ring','Pinky')):
            for segment in range(1,4):
                names[f'{side}Hand{finger}{segment}']=f'{side}Finger{digit}'+('' if segment==1 else str(segment-1))
    return names


def convert(inp,out,progress=lambda:None):
    mapping=bone_names();orders=[];seen=[]
    for _ in range(4096):
        line=inp.readline(65537)
        if not line or len(line)>65536:raise ValueError('Invalid BVH hierarchy')
        tokens=line.split()
        if tokens[:1] in ([b'ROOT'],[b'JOINT']):
            name=tokens[1].decode('ascii')
            if name not in mapping:raise ValueError('Unsupported Biped bone: '+name)
            seen.append(name)
            line=line.replace(tokens[1],mapping[name].encode('ascii'),1)
        elif tokens[:1]==[b'CHANNELS']:
            count=int(tokens[1]);rot=tokens[5:] if not orders else tokens[2:]
            if count!=(6 if not orders else 3) or sorted(rot)!=[b'Xrotation',b'Yrotation',b'Zrotation']:
                raise ValueError('Unsupported BVH channels')
            orders.append(''.join(x[:1].decode('ascii') for x in rot))
            indent=line[:len(line)-len(line.lstrip())]
            line=indent+(b'CHANNELS 6 Xposition Yposition Zposition ' if len(orders)==1 else b'CHANNELS 3 ')+b'Zrotation Xrotation Yrotation\n'
        out.write(line)
        if line.strip()==b'MOTION':break
    else:raise ValueError('BVH hierarchy too large')
    if len(orders)!=52 or set(seen)!=set(mapping):raise ValueError('Expected complete source skeleton')
    count_line=inp.readline(256);time_line=inp.readline(256)
    if not count_line.startswith(b'Frames:') or not time_line.startswith(b'Frame Time:'):raise ValueError('Invalid BVH header')
    count=int(count_line.split(b':')[1]);step=float(time_line.split(b':')[1])
    if count<1 or not np.isfinite(step) or step<=0:raise ValueError('Invalid BVH timing')
    out.write(count_line);out.write(time_line)
    previous=np.zeros((52,3))
    for start in range(0,count,128):
        rows=[]
        for _ in range(min(128,count-start)):
            line=inp.readline(65537)
            if len(line)>65536:raise ValueError('BVH frame too large')
            fields=line.split()
            if len(fields)!=159:raise ValueError('Invalid BVH frame')
            rows.append(np.asarray(fields,dtype=float))
        values=np.asarray(rows)
        if not np.isfinite(values).all():raise ValueError('Non-finite BVH motion')
        for j,order in enumerate(orders):
            column=slice(3+3*j,6+3*j)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Gimbal lock detected')
                angles=Rotation.from_euler(order,values[:,column],degrees=True).as_euler('ZXY')
            for index,angle in enumerate(angles):
                alternative=angle.copy();alternative[0]+=np.pi
                alternative[1]=np.pi-alternative[1];alternative[2]+=np.pi
                angle=angle+2*np.pi*np.round((previous[j]-angle)/(2*np.pi))
                alternative+=2*np.pi*np.round((previous[j]-alternative)/(2*np.pi))
                angles[index]=angle if np.linalg.norm(angle-previous[j])<=np.linalg.norm(alternative-previous[j]) else alternative
                previous[j]=angles[index]
            previous[j]=angles[-1]
            values[:,column]=np.degrees(angles)
        for row in values:out.write((' '.join(f'{v:.9f}' for v in row)+'\n').encode('ascii'))
        progress()
    if inp.read(1):raise ValueError('Unexpected BVH trailing data')
