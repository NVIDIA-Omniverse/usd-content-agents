"""Synthetic CUDA/Vulkan access probes and one actual native OVRTX frame.

Run inside the proposed author namespace. No benchmark source/model is used.
CUDA_VISIBLE_DEVICES is deliberately removed for the access probes.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import subprocess
import shutil
import time
import uuid
from pathlib import Path


def cuda():
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    lib = C.CDLL("libcuda.so.1")
    code = lib.cuInit(0)
    if code: return {"init": code, "devices": []}
    count = C.c_int(); result = lib.cuDeviceGetCount(C.byref(count))
    rows = []
    for index in range(count.value):
        dev = C.c_int(); lib.cuDeviceGet(C.byref(dev), index)
        identity = (C.c_ubyte * 16)()
        code = lib.cuDeviceGetUuid(C.byref(identity), dev)
        identifier = "GPU-" + str(uuid.UUID(bytes=bytes(identity))) if code == 0 else None
        ctx = C.c_void_p(); created = lib.cuDevicePrimaryCtxRetain(C.byref(ctx), dev)
        allocation = None
        if created == 0:
            lib.cuCtxSetCurrent(ctx)
            pointer = C.c_uint64()
            allocation = lib.cuMemAlloc_v2(C.byref(pointer), C.c_size_t(4096))
            if allocation == 0:
                cleared = lib.cuMemsetD8_v2(pointer, C.c_ubyte(173), C.c_size_t(4096))
                data = (C.c_ubyte * 4096)()
                copied = lib.cuMemcpyDtoH_v2(data, pointer, C.c_size_t(4096))
                allocation = 0 if cleared == copied == 0 and bytes(data) == bytes([173])*4096 else -1
                lib.cuMemFree_v2(pointer)
            lib.cuCtxSetCurrent(C.c_void_p())
            lib.cuDevicePrimaryCtxRelease(dev)
        rows.append({"index": index, "uuid": identifier, "context_status": created, "allocation_roundtrip_status": allocation})
    return {"init": 0, "enumeration_status": result, "devices": rows}


def vulkan():
    class App(C.Structure):
        _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("name", C.c_char_p), ("version", C.c_uint32), ("engine", C.c_char_p), ("engineVersion", C.c_uint32), ("apiVersion", C.c_uint32)]
    class InstanceInfo(C.Structure):
        _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32), ("app", C.POINTER(App)), ("layerCount", C.c_uint32), ("layers", C.c_void_p), ("extensionCount", C.c_uint32), ("extensions", C.c_void_p)]
    class Identity(C.Structure):
        _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("deviceUUID", C.c_ubyte*16), ("driverUUID", C.c_ubyte*16), ("luid", C.c_ubyte*8), ("nodeMask", C.c_uint32), ("luidValid", C.c_uint32)]
    class Props2(C.Structure):
        _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("properties", C.c_ubyte*4096)]
    class QueueFamily(C.Structure):
        _fields_ = [("flags", C.c_uint32), ("count", C.c_uint32), ("timestampBits", C.c_uint32), ("granularity", C.c_uint32*3)]
    class QueueInfo(C.Structure):
        _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32), ("family", C.c_uint32), ("count", C.c_uint32), ("priorities", C.POINTER(C.c_float))]
    class DeviceInfo(C.Structure):
        _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32), ("queueCount", C.c_uint32), ("queues", C.POINTER(QueueInfo)), ("layerCount", C.c_uint32), ("layers", C.c_void_p), ("extensionCount", C.c_uint32), ("extensions", C.c_void_p), ("features", C.c_void_p)]
    lib = C.CDLL("libvulkan.so.1")
    app = App(0, None, b"isolated-gpu-qualification", 1, b"synthetic", 1, (1<<22)|(2<<12))
    info = InstanceInfo(1, None, 0, C.pointer(app), 0, None, 0, None)
    instance = C.c_void_p(); code = lib.vkCreateInstance(C.byref(info), None, C.byref(instance))
    if code: return {"instance_status": code, "devices": []}
    rows = []
    try:
        count = C.c_uint32(); lib.vkEnumeratePhysicalDevices(instance, C.byref(count), None)
        devices = (C.c_void_p * count.value)(); lib.vkEnumeratePhysicalDevices(instance, C.byref(count), devices)
        for ptr in devices:
            physical = C.c_void_p(ptr)
            identity = Identity(); identity.sType = 1000071004
            props = Props2(); props.sType = 1000059001; props.pNext = C.cast(C.pointer(identity), C.c_void_p)
            lib.vkGetPhysicalDeviceProperties2(physical, C.byref(props))
            raw = bytes(props.properties)
            vendor = int.from_bytes(raw[8:12], "little")
            name = raw[20:276].split(b"\0", 1)[0].decode(errors="replace")
            if vendor != 0x10DE:
                rows.append({"vendor_id": vendor, "name": name, "nvidia": False}); continue
            family_count = C.c_uint32(); lib.vkGetPhysicalDeviceQueueFamilyProperties(physical, C.byref(family_count), None)
            families = (QueueFamily * family_count.value)(); lib.vkGetPhysicalDeviceQueueFamilyProperties(physical, C.byref(family_count), families)
            family = next(i for i, q in enumerate(families) if q.count and q.flags & 3)
            priority = C.c_float(1); queue = QueueInfo(2, None, 0, family, 1, C.pointer(priority))
            device_info = DeviceInfo(3, None, 0, 1, C.pointer(queue), 0, None, 0, None, None)
            device = C.c_void_p(); status = lib.vkCreateDevice(physical, C.byref(device_info), None, C.byref(device))
            if status == 0: lib.vkDestroyDevice(device, None)
            rows.append({"nvidia": True, "vendor_id": vendor, "name": name,
                         "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identity.deviceUUID))), "device_create_status": status})
    finally:
        lib.vkDestroyInstance(instance, None)
    return {"instance_status": 0, "devices": rows}


def render():
    from pxr import Usd, UsdGeom, UsdLux, Gf
    from PIL import Image, ImageStat
    source = Path("/work/synthetic.usda")
    stage = Usd.Stage.CreateNew(str(source)); world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Y"); UsdGeom.SetStageMetersPerUnit(stage, 1)
    cube = UsdGeom.Cube.Define(stage, "/World/Cube"); cube.CreateSizeAttr(1); cube.CreateDisplayColorAttr([(0.1, .55, .85)])
    light = UsdLux.DomeLight.Define(stage, "/World/Light"); light.CreateIntensityAttr(700)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera"); camera.CreateFocalLengthAttr(35)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(2,2,3),Gf.Vec3d(0,0,0),Gf.Vec3d(0,1,0)).GetInverse())
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(0); stage.GetRootLayer().Save()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = Path("/work/.usd-cli"); config.mkdir(exist_ok=True)
    (config/"config.toml").write_text('[server]\nallowed_roots=["/work","/tools"]\nallowed_write_roots=["/work"]\n[render]\nrenderer="ovrtx"\novrtx_auto_install=false\n')
    argv = ["/tools/bin/usd-cli", "--json", "--timeout", "600", "render-frames", "--scene", str(source), "--frames", "0", "--res", "640x480", "--camera", "/World/Camera", "--renderer", "ovrtx", "--mode", "fast", "--no-animate", "--output", "/work/render"]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=650)
    finally:
        logs=Path('/work/native_logs'); logs.mkdir(exist_ok=True)
        for log in Path('/tmp').glob('ovrtx_err_*.log'):
            if log.is_file() and not log.is_symlink():shutil.copyfile(log,logs/log.name)
    Path("/work/render.stdout").write_text(p.stdout); Path("/work/render.stderr").write_text(p.stderr)
    images = []
    for path in Path("/work/render").rglob("*.png"):
        with Image.open(path) as image:
            rgb=image.convert("RGB"); deviation=max(ImageStat.Stat(rgb).stddev)
            images.append({"path": str(path), "size": list(rgb.size), "stddev_max": deviation, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"returncode": p.returncode, "argv": argv, "images": images,
            "source_sha256": digest, "source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest()==digest,
            "passed": p.returncode == 0 and any(x["stddev_max"] > 1 and x['size']==[640,480] for x in images)}


def main():
    p=argparse.ArgumentParser();p.add_argument("--assigned",required=True);p.add_argument("--render",action="store_true");p.add_argument('--sandbox-only',action='store_true');a=p.parse_args()
    if a.sandbox_only:
        home=Path('/home/agent/.codex');home.mkdir(mode=0o700,exist_ok=True)
        # No real or valid model credential. These commands execute only local
        # sandbox helpers; no exec/model request is issued by this fixture.
        (home/'config.toml').write_text('model="gpt-6-astra"\nmodel_reasoning_effort="ultra"\n')
        commands=[('bwrap_prerequisite',['/usr/bin/bwrap','--ro-bind','/','/','--dev','/dev','--proc','/proc','--','/usr/bin/true']),
                  ('codex_default',['/tools/bin/codex','sandbox','-c','sandbox_mode="workspace-write"','--','/usr/bin/sh','-c','printf default-ok > /work/default_marker && if printf forbidden > /home/agent/forbidden_child_write; then exit 73; fi']),
                  ('codex_legacy',['/tools/bin/codex','sandbox','-c','sandbox_mode="workspace-write"','-c','features.use_legacy_landlock=true','--','/usr/bin/sh','-c','printf legacy-ok > /work/legacy_marker'])]
        rows=[]
        for name,cmd in commands:
            r=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
            rows.append({'name':name,'argv':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        (home/'auth.json').write_text(json.dumps({'auth_mode':'apikey','OPENAI_API_KEY':'qualification-dummy-not-upstream'}))
        (home/'auth.json').chmod(0o600)
        cmd=['/tools/bin/codex','login','status'];r=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
        rows.append({'name':'dummy_only_login_status','argv':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,
                     'credential_has_upstream_authority':False})
        result={'schema_version':'native-child-sandbox.partial.v2','model_calls':0,'renderer_runs':0,'solver_runs':0,'checks':rows,
                'legacy_marker_exact':Path('/work/legacy_marker').exists() and Path('/work/legacy_marker').read_text()=='legacy-ok',
                'default_marker_exact':Path('/work/default_marker').exists() and Path('/work/default_marker').read_text()=='default-ok',
                'outside_workspace_write_refused':not Path('/home/agent/forbidden_child_write').exists(),
                'passed':False,'scope':'Provider-free sandbox preflight only; actual native model child still requires qualification'}
        Path('/work/gpu_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0
    began=time.time(); result={"schema_version":"gpu-boundary.partial.v2","assigned_uuid":a.assigned,"cuda_visible_devices_removed":True,"model_calls":0,"benchmark_sources_used":False}
    for name,fn in (("cuda",cuda),("vulkan",vulkan)):
        try:result[name]=fn()
        except Exception as exc:result[name]={"error":type(exc).__name__+": "+str(exc)}
    cuda_owned={x["uuid"] for x in result["cuda"].get("devices",[]) if x.get("allocation_roundtrip_status")==0}
    vk_owned={x["uuid"] for x in result["vulkan"].get("devices",[]) if x.get("nvidia") and x.get("device_create_status")==0}
    result["cuda_only_assigned_accessible"]=cuda_owned=={a.assigned}
    result["vulkan_only_assigned_accessible"]=vk_owned=={a.assigned}
    if a.render:
        try:result["render"]=render()
        except Exception as exc:result["render"]={"passed":False,"error":type(exc).__name__+": "+str(exc)}
    result["passed"]=result["cuda_only_assigned_accessible"] and result["vulkan_only_assigned_accessible"] and (not a.render or result["render"]["passed"])
    result["elapsed_seconds"]=time.time()-began
    Path("/work/gpu_receipt.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
