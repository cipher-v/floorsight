"""Build the final narrated submission from public repo media.

Requires FFmpeg/ffprobe, Pillow==12.1.0 and boto3; uses Amazon Polly neural Matthew.
Run: python scripts/video/build_submission.py SPEC.json WORK_DIR OUTPUT.mp4
No raw footage is read. Captions and a build receipt are written beside OUTPUT.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, math, subprocess, textwrap
import boto3
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS = 1280, 720, 25
BG, WHITE, MUTED, ACCENT = '#0b1220', '#f3f6fc', '#b6c4d8', '#55d9c0'


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-1800:])
    return r.stdout


def duration(path):
    return float(run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(path)]))


def font(size, bold=False):
    candidates = [Path('C:/Windows/Fonts')/('segoeuib.ttf' if bold else 'segoeui.ttf'),
                  Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    return ImageFont.truetype(str(next(p for p in candidates if p.exists())),size)


def panel(s, path):
    media = 'video' in s or 'image' in s
    im=Image.new('RGBA',(W,H),BG);d=ImageDraw.Draw(im)
    if media: d.rectangle((0,128,W,605),fill=(0,0,0,0))
    d.rectangle((36,28,42,108),fill=ACCENT)
    d.text((58,25),s['eyebrow'],font=font(20,True),fill=ACCENT)
    size=40
    while d.textlength(s['title'],font=font(size,True))>1180: size-=1
    d.text((56,60),s['title'],font=font(size,True),fill=WHITE)
    if 'metrics' in s:
        n=len(s['metrics']);cw=1180/n
        for i,(value,label) in enumerate(s['metrics']):
            x=int(50+i*cw);d.rounded_rectangle((x,195,x+cw-20,475),radius=18,fill='#142338')
            d.text((x+24,223),value,font=font(70,True),fill=ACCENT)
            for j,line in enumerate(textwrap.wrap(label,25 if n==2 else 20)):
                d.text((x+24,335+j*37),line,font=font(27),fill=WHITE)
    elif s.get('architecture'):
        top=[('CAMERAS','Video input'),('OPENCV 5','Perception + floor'),('AGENT','Evidence + decision'),('REVIEWER','Approve / reject')]
        for i,(a,b) in enumerate(top):
            x=35+i*312;d.rounded_rectangle((x,195,x+278,325),radius=14,fill='#142338')
            d.text((x+16,218),a,font=font(26,True),fill=ACCENT);d.text((x+16,265),b,font=font(21),fill=WHITE)
            if i<3:d.text((x+283,240),'→',font=font(28),fill=ACCENT)
        d.text((38,357),'AWS GRAVITON4 + COOL',font=font(24,True),fill=WHITE)
        for i,(a,b) in enumerate([('S3','Evidence'),('DynamoDB','Track history'),('CloudWatch','Logs + metrics'),('SNS','Approved alerts')]):
            x=35+i*312;d.rounded_rectangle((x,410,x+278,515),radius=14,outline='#365273',width=2)
            d.text((x+16,424),a,font=font(26,True),fill=WHITE);d.text((x+16,464),b,font=font(22),fill=MUTED)
    elif 'lines' in s:
        for i,line in enumerate(s['lines']):
            size=44
            while d.textlength(line,font=font(size,True))>1140:size-=1
            d.text((58,220+i*94),line,font=font(size,True),fill=WHITE if i else ACCENT)
    d.text((48,606),s.get('note',''),font=font(17),fill=MUTED)
    im.save(path)


def stamp(t, sep=','):
    ms=round(t*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}'


def ass_time(t):
    cs=round(t*100);h,cs=divmod(cs,360000);m,cs=divmod(cs,6000);s,cs=divmod(cs,100)
    return f'{h}:{m:02d}:{s:02d}.{cs:02d}'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);p.add_argument('work',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.work=a.work.resolve();a.output=a.output.resolve();a.work.mkdir(parents=True,exist_ok=True);a.output.parent.mkdir(parents=True,exist_ok=True)
    spec=json.loads(a.spec.read_text(encoding='utf-8'));client=boto3.client('polly',region_name='us-east-1')
    def prepare(pair):
        i,s=pair;folder=a.work/f'{i:02d}-{s["id"]}';folder.mkdir(exist_ok=True)
        audio=folder/'voice.mp3';marks=folder/'marks.jsonl'
        signature=hashlib.sha256(json.dumps([s,spec.get('voice','Matthew')],sort_keys=True).encode()).hexdigest()
        marker=folder/'signature.txt'
        if marker.exists() and marker.read_text()!=signature:
            for name in ('voice.mp3','marks.jsonl','segment.mp4'):
                (folder/name).unlink(missing_ok=True)
        marker.write_text(signature)
        kwargs=dict(Text=s['narration'],VoiceId=spec.get('voice','Matthew'),Engine='neural')
        if not audio.exists():audio.write_bytes(client.synthesize_speech(**kwargs,OutputFormat='mp3',SampleRate='24000')['AudioStream'].read())
        if not marks.exists():marks.write_bytes(client.synthesize_speech(**kwargs,OutputFormat='json',SpeechMarkTypes=['sentence'])['AudioStream'].read())
        length=duration(audio);total=math.ceil((length+0.6)*FPS)/FPS
        speech=[json.loads(x) for x in marks.read_text().splitlines()];captions=[]
        for j,m in enumerate(speech):
            start=m['time']/1000;end=speech[j+1]['time']/1000 if j+1<len(speech) else length
            chunks=textwrap.wrap(m['value'],92)
            for k,chunk in enumerate(chunks):
                captions.append((start+(end-start)*k/len(chunks),start+(end-start)*(k+1)/len(chunks)-.03,'\n'.join(textwrap.wrap(chunk,65))))
        header='[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Segoe UI,26,&H00FFFFFF,&H000000FF,&H000B1220,&H000B1220,0,0,0,0,100,100,0,0,3,2,0,2,45,45,22,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
        for st,en,txt in captions:header+=f'Dialogue: 0,{ass_time(st)},{ass_time(en)},Default,,0,0,0,,{txt.replace(chr(10),chr(92)+"N")}\n'
        (folder/'captions.ass').write_text(header,encoding='utf-8');panel(s,folder/'panel.png')
        return i,s,folder,total,captions,length
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:prepared=list(pool.map(prepare,enumerate(spec['sections'])))
    print('NARRATION READY',round(sum(x[3] for x in prepared),2),'seconds',flush=True)
    receipt={'schema':'floorsight.submission-video.v1','voice':spec.get('voice','Matthew'),'narration':'Amazon Polly synthetic voice','team':spec['team'],'sections':[]};allcaps=[];offset=0;segments=[]
    for i,s,folder,total,captions,audio_len in prepared:
        out=folder/'segment.mp4'
        if not out.exists():
            cmd=['ffmpeg','-v','error','-y','-filter_complex_threads','1']
            if 'video' in s:
                cmd+=['-ss',str(s.get('start',0)),'-i',str(ROOT/s['video'])]
            elif 'image' in s:cmd+=['-loop','1','-i',str(ROOT/s['image'])]
            else:cmd+=['-loop','1','-i',str(folder/'panel.png')]
            cmd+=['-i',str(folder/'voice.mp3')]
            if 'video' in s or 'image' in s:
                cmd+=['-loop','1','-i',str(folder/'panel.png')]
                vf='[0:v]scale=1280:477:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:128+(477-ih)/2:color=0x0b1220,tpad=stop_mode=clone:stop_duration=60[base];[base][2:v]overlay=0:0,subtitles=captions.ass[v]'
            else:vf='[0:v]subtitles=captions.ass[v]'
            cmd+=['-filter_complex',vf,'-map','[v]','-map','1:a','-af','apad','-t',str(total),'-r',str(FPS),'-c:v','libx264','-threads','2','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-ar','48000','-ac','2',str(out)]
            run(cmd,cwd=folder)
        print('RENDERED',s['id'],total,flush=True);segments.append(out)
        allcaps.extend((offset+st,offset+en,txt) for st,en,txt in captions)
        receipt['sections'].append({'id':s['id'],'start_s':round(offset,2),'duration_s':total,'source':s.get('video',s.get('image','generated title card')),'narration_s':audio_len});offset+=total
    concat=a.work/'concat.txt';concat.write_text(''.join(f"file '{q.as_posix()}'\n" for q in segments),encoding='utf-8')
    run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(concat),'-c:v','copy',
         '-af','loudnorm=I=-16:TP=-1.5:LRA=7','-c:a','aac','-b:a','128k','-ar','48000',
         '-movflags','+faststart',str(a.output)])
    a.output.with_suffix('.srt').write_text(''.join(f'{i+1}\n{stamp(st)} --> {stamp(en)}\n{txt}\n\n' for i,(st,en,txt) in enumerate(allcaps)),encoding='utf-8')
    a.output.with_suffix('.vtt').write_text('WEBVTT\n\n'+''.join(f'{i+1}\n{stamp(st,".")} --> {stamp(en,".")}\n{txt}\n\n' for i,(st,en,txt) in enumerate(allcaps)),encoding='utf-8')
    receipt.update(duration_s=duration(a.output),bytes=a.output.stat().st_size,sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),spec_sha256=hashlib.sha256(a.spec.read_bytes()).hexdigest())
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print('DONE',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
