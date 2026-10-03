from __future__ import print_function
import pytorch_nndct
import argparse, json, os, pickle, sys, tarfile, hashlib
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from pytorch_nndct.apis import torch_quantizer

EXPECTED_SHA="331e2ea6720abf004154b1e58a74b675af2a20f306746ccfda91528533e3ddb3"
MEAN=[0.5071,0.4867,0.4408]
STD=[0.2675,0.2565,0.2761]

def die(x): raise RuntimeError(x)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        while True:
            b=f.read(1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()

class CIFAR100Pickle(Dataset):
    def __init__(self,p,transform=None):
        with open(p,"rb") as f: d=pickle.load(f,encoding="bytes")
        if b"data" not in d or b"fine_labels" not in d: die("bad CIFAR-100 pickle")
        self.data=d[b"data"]; self.labels=d[b"fine_labels"]; self.transform=transform
        if self.data.shape[1] != 3072: die("unexpected CIFAR shape")
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        x=self.data[i].reshape(3,32,32).transpose(1,2,0)
        x=Image.fromarray(x)
        if self.transform: x=self.transform(x)
        return x,int(self.labels[i])

def make_model():
    m=models.resnet101(pretrained=False)
    m.fc=nn.Linear(m.fc.in_features,100)
    return m

def load_checkpoint(model,p):
    if sha256_file(p)!=EXPECTED_SHA: die("checkpoint SHA changed")
    ckpt=torch.load(p,map_location="cpu")
    if not isinstance(ckpt,dict) or "state_dict" not in ckpt: die("bad checkpoint format")
    for k,v in {
      "architecture":"torchvision.models.resnet101(pretrained=False)",
      "num_classes":100,
      "input_shape":[1,3,224,224],
    }.items():
        if k in ckpt and ckpt[k]!=v: die("checkpoint metadata mismatch "+k)
    r=model.load_state_dict(ckpt["state_dict"],strict=True)
    if getattr(r,"missing_keys",[]): die("missing state_dict keys")
    if getattr(r,"unexpected_keys",[]): die("unexpected state_dict keys")
    if tuple(model.fc.weight.shape)!=(100,2048): die("fc shape mismatch")
    return ckpt

def extract_dataset(archive,out):
    root=os.path.join(out,"cifar-100-python")
    tr=os.path.join(root,"train"); te=os.path.join(root,"test")
    if os.path.isfile(tr) and os.path.isfile(te): return tr,te
    if not os.path.isdir(out): os.makedirs(out)
    with tarfile.open(archive,"r:gz") as tf:
        base=os.path.abspath(out)
        for m in tf.getmembers():
            d=os.path.abspath(os.path.join(out,m.name))
            if not (d==base or d.startswith(base+os.sep)): die("unsafe tar path")
        tf.extractall(out)
    if not os.path.isfile(tr) or not os.path.isfile(te): die("dataset extraction failed")
    return tr,te

def spread(total,n):
    if n<=0 or n>=total: return list(range(total))
    return [int((i*total)//n) for i in range(n)]

def _loader_worker_init(_worker_id):
    # DataLoader workers must stay single-threaded. Otherwise every worker
    # inherits the main OMP/MKL/PyTorch thread count and causes massive
    # oversubscription on large CPU nodes.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

def loaders(a):
    tr,te=extract_dataset(a.dataset,a.dataset_dir)
    tf=transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(MEAN,STD),
    ])
    trd=CIFAR100Pickle(tr,tf); ted=CIFAR100Pickle(te,tf)
    if len(trd)!=50000 or len(ted)!=10000: die("dataset size mismatch")
    cl=DataLoader(Subset(trd,spread(len(trd),a.calib_samples)),
                  batch_size=a.calib_batch,shuffle=False,num_workers=a.num_workers,
                  worker_init_fn=_loader_worker_init,
                  persistent_workers=(a.num_workers>0))
    tl=DataLoader(Subset(ted,spread(len(ted),a.test_samples)),
                  batch_size=a.test_batch,shuffle=False,num_workers=a.num_workers,
                  worker_init_fn=_loader_worker_init,
                  persistent_workers=(a.num_workers>0))
    el=DataLoader(Subset(ted,[0]),batch_size=1,shuffle=False,num_workers=0)
    return cl,tl,el

def evaluate(model,loader):
    model.eval(); total=top1=top5=0; loss=0.0
    ce=nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x,y in loader:
            o=model(x)
            if o.ndim!=2 or o.shape[1]!=100: die("output shape mismatch")
            loss += float(ce(o,y).item()); total += int(y.numel())
            top1 += int(o.argmax(1).eq(y).sum().item())
            _,p5=o.topk(5,1,True,True)
            top5 += int(p5.eq(y.view(-1,1)).any(1).sum().item())
    if total==0: die("empty loader")
    return {"samples":total,"loss":loss/total,"top1":top1/float(total),"top5":top5/float(total)}

def version_gate():
    import numpy as np, PIL, xir
    py="%d.%d.%d"%sys.version_info[:3]
    v={"python":py,"torch":torch.__version__,"torchvision":torchvision.__version__,
       "numpy":np.__version__,"pillow":PIL.__version__,
       "pytorch_nndct_file":pytorch_nndct.__file__,"xir_file":xir.__file__}
    if py!="3.6.15": die("Python mismatch: "+py)
    if torch.__version__!="1.7.1": die("torch mismatch: "+torch.__version__)
    if torchvision.__version__!="0.8.2": die("torchvision mismatch: "+torchvision.__version__)
    if np.__version__!="1.17.5": die("numpy mismatch: "+np.__version__)
    if PIL.__version__!="8.0.0": die("Pillow mismatch: "+PIL.__version__)
    prefix=os.path.realpath(os.environ.get("CONDA_PREFIX",""))
    nndct_path=os.path.realpath(pytorch_nndct.__file__)
    if not prefix or not nndct_path.startswith(prefix+os.sep):
        die("pytorch_nndct not loaded from active pinned vitis-ai-pytorch env: "+nndct_path)
    return v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--action",choices=["calib","test","export"],required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--dataset-dir",required=True)
    ap.add_argument("--quant-dir",required=True)
    ap.add_argument("--metrics-dir",required=True)
    ap.add_argument("--calib-samples",type=int,required=True)
    ap.add_argument("--test-samples",type=int,required=True)
    ap.add_argument("--calib-batch",type=int,required=True)
    ap.add_argument("--test-batch",type=int,required=True)
    ap.add_argument("--max-top1-drop",type=float,required=True)
    ap.add_argument("--cpu-threads",type=int,required=True)
    ap.add_argument("--interop-threads",type=int,required=True)
    ap.add_argument("--num-workers",type=int,required=True)
    a=ap.parse_args()

    if a.cpu_threads < 1: die("cpu_threads must be >=1")
    if a.interop_threads < 1: die("interop_threads must be >=1")
    if a.num_workers < 0: die("num_workers must be >=0")

    torch.set_num_threads(a.cpu_threads)
    try:
        torch.set_num_interop_threads(a.interop_threads)
    except RuntimeError as e:
        print("INTEROP_THREADS_WARNING="+str(e))

    print("TORCH_NUM_THREADS=%d"%torch.get_num_threads())
    try:
        print("TORCH_NUM_INTEROP_THREADS=%d"%torch.get_num_interop_threads())
    except Exception:
        pass
    print("DATALOADER_NUM_WORKERS=%d"%a.num_workers)

    versions=version_gate()
    for d in [a.quant_dir,a.metrics_dir]:
        if not os.path.isdir(d): os.makedirs(d)
    cl,tl,el=loaders(a)
    model=make_model().cpu().eval()
    load_checkpoint(model,a.checkpoint)

    z=torch.randn(1,3,224,224)
    traced=torch.jit.trace(model,z,strict=True)
    if tuple(traced(z).shape)!=(1,100): die("JIT trace output mismatch")
    print("STRICT_LOAD=PASS")
    print("JIT_TRACE_CHECK=PASS")

    if a.action=="calib":
        dummy=torch.randn(a.calib_batch,3,224,224)
        q=torch_quantizer("calib",model,(dummy,),output_dir=a.quant_dir,device=torch.device("cpu"))
        qm=q.quant_model; qm.eval()
        cm=evaluate(qm,cl)
        q.export_quant_config()
        with open(os.path.join(a.metrics_dir,"calib_metrics.json"),"w") as f:
            json.dump({"versions":versions,"calibration":cm},f,indent=2,sort_keys=True)
        print("CALIBRATION=PASS")
        return

    if a.action=="test":
        fm=evaluate(model,tl)
        dummy=torch.randn(a.test_batch,3,224,224)
        q=torch_quantizer("test",model,(dummy,),output_dir=a.quant_dir,device=torch.device("cpu"))
        qm=q.quant_model; qm.eval()
        im=evaluate(qm,tl)
        drop=fm["top1"]-im["top1"]
        out={"versions":versions,"float":fm,"int8":im,
             "top1_absolute_drop":drop,"max_allowed_top1_drop":a.max_top1_drop}
        with open(os.path.join(a.metrics_dir,"quant_test_metrics.json"),"w") as f:
            json.dump(out,f,indent=2,sort_keys=True)
        print("FLOAT_METRICS="+json.dumps(fm,sort_keys=True))
        print("INT8_METRICS="+json.dumps(im,sort_keys=True))
        print("TOP1_ABSOLUTE_DROP=%.8f"%drop)
        if drop>a.max_top1_drop: die("INT8 top1 drop exceeds limit")
        print("QUANT_ACCURACY_GATE=PASS")
        return

    dummy=torch.randn(1,3,224,224)
    q=torch_quantizer("test",model,(dummy,),output_dir=a.quant_dir,device=torch.device("cpu"))
    qm=q.quant_model; qm.eval()
    with torch.no_grad():
        for x,_ in el:
            if tuple(x.shape)!=(1,3,224,224): die("export is not batch 1")
            if tuple(qm(x).shape)!=(1,100): die("export forward shape mismatch")
            break
    q.export_xmodel(a.quant_dir,False)
    print("EXPORT_XMODEL_CALL=PASS")

if __name__=="__main__":
    main()
