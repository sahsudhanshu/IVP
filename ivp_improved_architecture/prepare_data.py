"""Extract only NW-UCLA action videos from the provided ZIP safely."""
import argparse, shutil, zipfile
from pathlib import Path, PurePosixPath

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--archive',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    dest=Path(args.output).resolve(); dest.mkdir(parents=True,exist_ok=True); total=0
    with zipfile.ZipFile(args.archive) as z:
        for member in z.infolist():
            parts=PurePosixPath(member.filename).parts
            if member.is_dir() or 'multiview_action_videos' not in parts or not member.filename.lower().endswith('.avi'): continue
            suffix=parts[parts.index('multiview_action_videos'):]
            if any(x in ('..','.') for x in suffix): raise ValueError(f'Unsafe ZIP member: {member.filename}')
            target=dest.joinpath(*suffix).resolve()
            if not target.is_relative_to(dest): raise ValueError(f'Unsafe ZIP member: {member.filename}')
            if target.exists(): raise FileExistsError(f'Refusing to overwrite {target}')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as src,target.open('wb') as out: shutil.copyfileobj(src,out)
            total+=1
    if total==0: raise ValueError('No action videos found in archive')
    print(f'Extracted {total} videos into {dest}')
if __name__=='__main__': main()
