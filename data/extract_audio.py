import sys, os
import numpy as np
import subprocess
from glob import glob
import argparse
import cv2
import pdb

def extractData(videopath, audiopath=ModuleNotFoundError):
  if audiopath == None:
    sys.exit('Audiopath should be inputted.')

  if audiopath != None:
    audiocmd = 'ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s -loglevel quiet' % (videopath, audiopath)
    output = subprocess.call(audiocmd, shell=True, stdout=None)

  return output


def dataProcessing(videoroot, audioroot=None, listname='pretrain.list'):

  print('(INPUT)  VIDEO path: %s' % videoroot)
  print('(OUTPUT) AUDIO path: %s' % audioroot)

  if not os.path.exists(listname):
    videolist = sorted(glob(videoroot + os.sep + '*' + os.sep + '*.mp4'))
    prelist = []
  else:
    print('Read presaved list file from ', listname)
    fid = open(listname, 'r')
    prelist = fid.read().split('\n')[:-1]
    fid.close()
    videolist = sorted([os.path.join(videoroot, videopath+'.mp4') for videopath in prelist])

  print('Number of files: %d' % len(videolist))

  for idx, filepath in enumerate(videolist):
    if not os.path.exists(filepath):
      continue

    spk = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    vid = os.path.basename(os.path.dirname(filepath))
    mp4 = os.path.basename(filepath)


    audioname = None
    if audioroot != None:
      audioname = filepath.replace('.mp4', '.wav').replace(videoroot, audioroot)

    if audioroot != None:
      if not os.path.exists(os.path.dirname(audioname)):
        os.makedirs(os.path.dirname(audioname))
   
    extractData(filepath, audioname)

    sys.stdout.write('\rExtracting %s -- %03.03f%%' % (filepath, float(idx+1)/len(videolist)*100))
    sys.stdout.flush()

  sys.stdout.write('\n')



def main():
  parser = argparse.ArgumentParser(description='dataProcessing')

  parser.add_argument('--vid_path', type=str, default='data/lrs3/mp4', help='data path for lrs3 dataset')
  parser.add_argument('--aud_path', type=str, default='data/lrs3/wav', help='data path for lrs3 audio dataset')
  parser.add_argument('--list_path', type=str, help='list path for lrs3')

  args = parser.parse_args()

  print('Extract audio from videos.')
  dataProcessing(args.vid_path, args.aud_path, args.list_path)
  print('Data list are generated here > %s' % args.list_path)
  print('Complete video extraction step!\n')

if __name__ == '__main__':
  main()
