import os
import time

# --- 1. 镜像源优化：使用国内 hf-mirror 镜像站 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 2. 缓存路径强制指定到系统盘 ---
os.environ["HF_HOME"] = "/root/system_models/huggingface"

import gradio as gr
import numpy as np
import random
import torch
from PIL import Image

from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download
from rembg import remove, new_session

MAX_SEED = np.iinfo(np.int32).max

# --- Model Loading (加载 GGUF 量化模型) ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

print("正在检查/加载 GGUF 量化模型 (Q2_K - 7.47GB)...")

gguf_file = hf_hub_download(
    repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",
    filename="qwen-image-edit-2511-Q2_K.gguf"
)

transformer = QwenImageTransformer2DModel.from_single_file(
    gguf_file,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="Qwen/Qwen-Image-Edit-2511",
    subfolder="transformer",
)

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    transformer=transformer,
    torch_dtype=dtype
)

# --- 核心修改：读取 start.sh 传来的环境变量，决定是否开启 CPU 优化 ---
if os.environ.get("DISABLE_CPU_OFFLOAD") == "1":
    pipe.to(device)
    print("🚀 CPU 优化已关闭: 模型已全量载入显存 (VRAM)，生成速度提升！")
else:
    pipe.enable_model_cpu_offload()
    print("🛡️ CPU 优化已开启: 模型成功加载至 cuda 并启用 CPU 卸载 (防止爆显存)")

pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Edit-2511-Lightning",
    weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    adapter_name="lightning"
)

pipe.load_lora_weights(
    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors",
    adapter_name="angles"
)

pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])

# --- 预加载 rembg session（强制 CPU，避免和 Qwen 抢显存）---
print("正在预加载 rembg 抠图模型 (CPU 模式)...")
rembg_session = new_session("u2net", providers=["CPUExecutionProvider"])
print("✅ rembg 模型加载完成")

# --- Prompt Building ---
AZIMUTH_MAP = {
    0: "front view", 45: "front-right quarter view", 90: "right side view",
    135: "back-right quarter view", 180: "back view", 225: "back-left quarter view",
    270: "left side view", 315: "front-left quarter view"
}

ELEVATION_MAP = {-30: "low-angle shot", 0: "eye-level shot", 30: "elevated shot", 60: "high-angle shot"}
DISTANCE_MAP = {0.6: "close-up", 1.0: "medium shot", 1.8: "wide shot"}

def snap_to_nearest(value, options):
    return min(options, key=lambda x: abs(x - value))

def build_camera_prompt(azimuth: float, elevation: float, distance: float, extra_prompt: str = "") -> str:
    azimuth_snapped = snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    elevation_snapped = snap_to_nearest(elevation, list(ELEVATION_MAP.keys()))
    distance_snapped = snap_to_nearest(distance, list(DISTANCE_MAP.keys()))

    base = f"<sks> {AZIMUTH_MAP[azimuth_snapped]} {ELEVATION_MAP[elevation_snapped]} {DISTANCE_MAP[distance_snapped]}"
    if extra_prompt and extra_prompt.strip():
        return f"{base}, {extra_prompt.strip()}"
    return base

# --- 核心生成逻辑 ---
def _generate_single_image(
    image: Image.Image, azimuth: float, elevation: float, distance: float,
    extra_prompt: str, remove_bg: bool, history: list,
    seed: int, randomize_seed: bool, guidance_scale: float,
    num_inference_steps: int, height: int, width: int
):
    prompt = build_camera_prompt(azimuth, elevation, distance, extra_prompt)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # 使用 CPU generator，避免与 enable_model_cpu_offload() 发生跨设备冲突
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if image is None:
        raise gr.Error("请先上传一张图像。")

    pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")

    result = pipe(
        image=[pil_image], prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    # 推理完成后立即释放显存缓存，再做抠图
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if remove_bg:
        result = remove(result, session=rembg_session)

    # 浅拷贝列表，防止 Gradio 状态被原地污染
    new_history = list(history) if history is not None else []
    new_history.append((result, f"A:{azimuth}° E:{elevation}°"))

    return new_history, seed, prompt


# --- 统一生成调度器（带进度和计时） ---
def generation_dispatcher(
    mode: str, image: Image.Image, azimuth: float, elevation: float, distance: float,
    extra_prompt: str, remove_bg: bool, history: list,
    seed: int, randomize_seed: bool, guidance_scale: float,
    num_inference_steps: int, height: int, width: int,
    cancel_flag: bool,
    progress=gr.Progress()
):
    # 终极防护网：确保任何可能的 None 值都不会透传进来
    azimuth             = 0.0  if azimuth             is None else float(azimuth)
    elevation           = 0.0  if elevation           is None else float(elevation)
    distance            = 1.0  if distance            is None else float(distance)
    guidance_scale      = 1.0  if guidance_scale      is None else float(guidance_scale)
    num_inference_steps = 4    if num_inference_steps is None else int(num_inference_steps)
    height              = 1024 if height              is None else int(height)
    width               = 1024 if width               is None else int(width)
    seed                = 0    if seed                is None else int(seed)
    extra_prompt        = extra_prompt or ""

    print(f"[GEN] mode={mode}, az={azimuth}, el={elevation}, dist={distance}, steps={num_inference_steps}, cfg={guidance_scale}, size={width}x{height}")

    start_time = time.time()

    if image is None:
        raise gr.Error("请先上传一张图像。")

    if history is None:
        history = []

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    if mode == "单张生成（当前视角）":
        progress(0, desc="🔄 正在生成...")
        new_history, seed, prompt = _generate_single_image(
            image, azimuth, elevation, distance, extra_prompt, remove_bg, history,
            seed, False, guidance_scale, num_inference_steps, height, width
        )
        elapsed = time.time() - start_time
        yield new_history, new_history, f"✅ 完成！用时 {elapsed:.1f} 秒", seed, prompt
        return

    else:  # 360° 八方向序列
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        total = len(angles)
        new_history = list(history)

        for i, az in enumerate(angles):
            if cancel_flag:
                elapsed = time.time() - start_time
                yield new_history, new_history, f"❌ 任务已取消 | 已用 {elapsed:.1f} 秒", seed, ""
                return

            elapsed = time.time() - start_time
            progress(i / total, desc=f"🔄 生成第 {i+1}/{total} 张 ({az}°) | 已用 {elapsed:.1f}s")

            prompt = build_camera_prompt(az, elevation, distance, extra_prompt)
            print(f"正在生成 360° 序列 [{az}°]: {prompt}")

            generator = torch.Generator(device="cpu").manual_seed(seed)
            pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")

            result = pipe(
                image=[pil_image], prompt=prompt,
                height=height if height != 0 else None,
                width=width if width != 0 else None,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
            ).images[0]

            # 每张推理完成后释放显存缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if remove_bg:
                result = remove(result, session=rembg_session)

            new_history.append((result, f"序列: {az}°"))

            elapsed = time.time() - start_time
            yield new_history, new_history, f"🔄 正在生成第 {i+1}/{total} 张 | 已用 {elapsed:.1f}s", seed, prompt

        elapsed = time.time() - start_time
        yield new_history, new_history, f"✅ 全部完成！共 {total} 张，用时 {elapsed:.1f} 秒", seed, prompt


def update_dimensions_on_upload(image):
    if image is None: return 1024, 1024
    w, h = image.size
    if w > h:
        new_w, new_h = 1024, int(1024 * (h / w))
    else:
        new_h, new_w = 1024, int(1024 * (w / h))
    return (new_w // 8) * 8, (new_h // 8) * 8


# --- 3D Camera Control Component ---
class CameraControl3D(gr.HTML):
    def __init__(self, value=None, imageUrl=None, **kwargs):
        if value is None:
            value = {"azimuth": 0, "elevation": 0, "distance": 1.0}

        html_template = """
        <div id="camera-control-wrapper" style="width: 100%; height: 450px; position: relative; background: #1a1a1a; border-radius: 12px; overflow: hidden;">
            <div id="prompt-overlay" style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); padding: 8px 16px; border-radius: 8px; font-family: monospace; font-size: 12px; color: #00ff88; white-space: nowrap; z-index: 10;"></div>
        </div>
        """

        js_on_load = """
        (() => {
            const wrapper = element.querySelector('#camera-control-wrapper');
            const promptOverlay = element.querySelector('#prompt-overlay');

            const initScene = () => {
                if (typeof THREE === 'undefined') {
                    setTimeout(initScene, 100);
                    return;
                }

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);

                const camera = new THREE.PerspectiveCamera(50, wrapper.clientWidth / wrapper.clientHeight, 0.1, 1000);
                camera.position.set(4.5, 3, 4.5);
                camera.lookAt(0, 0.75, 0);

                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                wrapper.insertBefore(renderer.domElement, promptOverlay);

                scene.add(new THREE.AmbientLight(0xffffff, 0.6));
                const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
                dirLight.position.set(5, 10, 5);
                scene.add(dirLight);

                scene.add(new THREE.GridHelper(8, 16, 0x333333, 0x222222));

                const CENTER = new THREE.Vector3(0, 0.75, 0);
                const BASE_DISTANCE = 1.6;
                const AZIMUTH_RADIUS = 2.4;
                const ELEVATION_RADIUS = 1.8;

                let azimuthAngle = props.value?.azimuth || 0;
                let elevationAngle = props.value?.elevation || 0;
                let distanceFactor = props.value?.distance || 1.0;

                const azimuthSteps = [0, 45, 90, 135, 180, 225, 270, 315];
                const elevationSteps = [-30, 0, 30, 60];
                const distanceSteps = [0.6, 1.0, 1.4];

                const azimuthNames = {
                    0: 'front view', 45: 'front-right quarter view', 90: 'right side view',
                    135: 'back-right quarter view', 180: 'back view', 225: 'back-left quarter view',
                    270: 'left side view', 315: 'front-left quarter view'
                };
                const elevationNames = { '-30': 'low-angle shot', '0': 'eye-level shot', '30': 'elevated shot', '60': 'high-angle shot' };
                const distanceNames = { '0.6': 'close-up', '1': 'medium shot', '1.4': 'wide shot' };

                function snapToNearest(value, steps) {
                    return steps.reduce((prev, curr) => Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev);
                }

                function createPlaceholderTexture() {
                    const canvas = document.createElement('canvas');
                    canvas.width = 256; canvas.height = 256;
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#3a3a4a'; ctx.fillRect(0, 0, 256, 256);
                    ctx.fillStyle = '#ffcc99'; ctx.beginPath(); ctx.arc(128, 128, 80, 0, Math.PI * 2); ctx.fill();
                    ctx.fillStyle = '#333'; ctx.beginPath(); ctx.arc(100, 110, 10, 0, Math.PI * 2); ctx.arc(156, 110, 10, 0, Math.PI * 2); ctx.fill();
                    ctx.strokeStyle = '#333'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(128, 130, 35, 0.2, Math.PI - 0.2); ctx.stroke();
                    return new THREE.CanvasTexture(canvas);
                }

                let currentTexture = createPlaceholderTexture();
                const planeMaterial = new THREE.MeshBasicMaterial({ map: currentTexture, side: THREE.DoubleSide });
                let targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                targetPlane.position.copy(CENTER);
                scene.add(targetPlane);

                function updateTextureFromUrl(url) {
                    if (!url) {
                        planeMaterial.map = createPlaceholderTexture();
                        planeMaterial.needsUpdate = true;
                        scene.remove(targetPlane);
                        targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                        targetPlane.position.copy(CENTER);
                        scene.add(targetPlane);
                        return;
                    }
                    const loader = new THREE.TextureLoader();
                    loader.crossOrigin = 'anonymous';
                    loader.load(url, (texture) => {
                        texture.minFilter = THREE.LinearFilter;
                        texture.magFilter = THREE.LinearFilter;
                        planeMaterial.map = texture;
                        planeMaterial.needsUpdate = true;
                        const img = texture.image;
                        if (img && img.width && img.height) {
                            const aspect = img.width / img.height;
                            const maxSize = 1.5;
                            let planeWidth, planeHeight;
                            if (aspect > 1) { planeWidth = maxSize; planeHeight = maxSize / aspect; }
                            else { planeHeight = maxSize; planeWidth = maxSize * aspect; }
                            scene.remove(targetPlane);
                            targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(planeWidth, planeHeight), planeMaterial);
                            targetPlane.position.copy(CENTER);
                            scene.add(targetPlane);
                        }
                    }, undefined, (err) => { console.error('加载纹理失败:', err); });
                }

                if (props.imageUrl) { updateTextureFromUrl(props.imageUrl); }

                const cameraGroup = new THREE.Group();
                const bodyMat = new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 });
                const body = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.22, 0.38), bodyMat);
                cameraGroup.add(body);
                const lens = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.09, 0.11, 0.18, 16),
                    new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 })
                );
                lens.rotation.x = Math.PI / 2; lens.position.z = 0.26;
                cameraGroup.add(lens);
                scene.add(cameraGroup);

                const azimuthRing = new THREE.Mesh(
                    new THREE.TorusGeometry(AZIMUTH_RADIUS, 0.04, 16, 64),
                    new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.3 })
                );
                azimuthRing.rotation.x = Math.PI / 2; azimuthRing.position.y = 0.05;
                scene.add(azimuthRing);

                const azimuthHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.5 })
                );
                azimuthHandle.userData.type = 'azimuth';
                scene.add(azimuthHandle);

                const arcPoints = [];
                for (let i = 0; i <= 32; i++) {
                    const angle = THREE.MathUtils.degToRad(-30 + (90 * i / 32));
                    arcPoints.push(new THREE.Vector3(-0.8, ELEVATION_RADIUS * Math.sin(angle) + CENTER.y, ELEVATION_RADIUS * Math.cos(angle)));
                }
                const arcCurve = new THREE.CatmullRomCurve3(arcPoints);
                const elevationArc = new THREE.Mesh(
                    new THREE.TubeGeometry(arcCurve, 32, 0.04, 8, false),
                    new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.3 })
                );
                scene.add(elevationArc);

                const elevationHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.5 })
                );
                elevationHandle.userData.type = 'elevation';
                scene.add(elevationHandle);

                const distanceLineGeo = new THREE.BufferGeometry();
                const distanceLine = new THREE.Line(distanceLineGeo, new THREE.LineBasicMaterial({ color: 0xffa500 }));
                scene.add(distanceLine);

                const distanceHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0xffa500, emissive: 0xffa500, emissiveIntensity: 0.5 })
                );
                distanceHandle.userData.type = 'distance';
                scene.add(distanceHandle);

                function updatePositions() {
                    const distance = BASE_DISTANCE * distanceFactor;
                    const azRad = THREE.MathUtils.degToRad(azimuthAngle);
                    const elRad = THREE.MathUtils.degToRad(elevationAngle);
                    const camX = distance * Math.sin(azRad) * Math.cos(elRad);
                    const camY = distance * Math.sin(elRad) + CENTER.y;
                    const camZ = distance * Math.cos(azRad) * Math.cos(elRad);
                    cameraGroup.position.set(camX, camY, camZ);
                    cameraGroup.lookAt(CENTER);
                    azimuthHandle.position.set(AZIMUTH_RADIUS * Math.sin(azRad), 0.05, AZIMUTH_RADIUS * Math.cos(azRad));
                    elevationHandle.position.set(-0.8, ELEVATION_RADIUS * Math.sin(elRad) + CENTER.y, ELEVATION_RADIUS * Math.cos(elRad));
                    const orangeDist = distance - 0.5;
                    distanceHandle.position.set(
                        orangeDist * Math.sin(azRad) * Math.cos(elRad),
                        orangeDist * Math.sin(elRad) + CENTER.y,
                        orangeDist * Math.cos(azRad) * Math.cos(elRad)
                    );
                    distanceLineGeo.setFromPoints([cameraGroup.position.clone(), CENTER.clone()]);
                    const azSnap = snapToNearest(azimuthAngle, azimuthSteps);
                    const elSnap = snapToNearest(elevationAngle, elevationSteps);
                    const distSnap = snapToNearest(distanceFactor, distanceSteps);
                    const distKey = distSnap === 1 ? '1' : distSnap.toFixed(1);
                    promptOverlay.textContent = '<sks> ' + azimuthNames[azSnap] + ' ' + elevationNames[String(elSnap)] + ' ' + distanceNames[distKey];
                }

                function updatePropsAndTrigger() {
                    const azSnap = snapToNearest(azimuthAngle, azimuthSteps);
                    const elSnap = snapToNearest(elevationAngle, elevationSteps);
                    const distSnap = snapToNearest(distanceFactor, distanceSteps);
                    props.value = { azimuth: azSnap, elevation: elSnap, distance: distSnap };
                    trigger('change', props.value);
                }

                const raycaster = new THREE.Raycaster();
                const mouse = new THREE.Vector2();
                let isDragging = false;
                let dragTarget = null;
                let dragStartMouse = new THREE.Vector2();
                let dragStartDistance = 1.0;
                const intersection = new THREE.Vector3();
                const canvas = renderer.domElement;

                canvas.addEventListener('mousedown', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle]);
                    if (intersects.length > 0) {
                        isDragging = true; dragTarget = intersects[0].object;
                        dragTarget.material.emissiveIntensity = 1.0; dragTarget.scale.setScalar(1.3);
                        dragStartMouse.copy(mouse); dragStartDistance = distanceFactor;
                        canvas.style.cursor = 'grabbing';
                    }
                });

                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                    if (isDragging && dragTarget) {
                        raycaster.setFromCamera(mouse, camera);
                        if (dragTarget.userData.type === 'azimuth') {
                            const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                azimuthAngle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                                if (azimuthAngle < 0) azimuthAngle += 360;
                            }
                        } else if (dragTarget.userData.type === 'elevation') {
                            const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.8);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                const relY = intersection.y - CENTER.y;
                                const relZ = intersection.z;
                                elevationAngle = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(relY, relZ)), -30, 60);
                            }
                        } else if (dragTarget.userData.type === 'distance') {
                            const deltaY = mouse.y - dragStartMouse.y;
                            distanceFactor = THREE.MathUtils.clamp(dragStartDistance - deltaY * 1.5, 0.6, 1.4);
                        }
                        updatePositions();
                    } else {
                        raycaster.setFromCamera(mouse, camera);
                        const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle]);
                        [azimuthHandle, elevationHandle, distanceHandle].forEach(h => { h.material.emissiveIntensity = 0.5; h.scale.setScalar(1); });
                        if (intersects.length > 0) {
                            intersects[0].object.material.emissiveIntensity = 0.8;
                            intersects[0].object.scale.setScalar(1.1);
                            canvas.style.cursor = 'grab';
                        } else { canvas.style.cursor = 'default'; }
                    }
                });

                const onMouseUp = () => {
                    if (dragTarget) {
                        dragTarget.material.emissiveIntensity = 0.5; dragTarget.scale.setScalar(1);
                        const targetAz = snapToNearest(azimuthAngle, azimuthSteps);
                        const targetEl = snapToNearest(elevationAngle, elevationSteps);
                        const targetDist = snapToNearest(distanceFactor, distanceSteps);
                        const startAz = azimuthAngle, startEl = elevationAngle, startDist = distanceFactor;
                        const startTime = Date.now();
                        function animateSnap() {
                            const t = Math.min((Date.now() - startTime) / 200, 1);
                            const ease = 1 - Math.pow(1 - t, 3);
                            let azDiff = targetAz - startAz;
                            if (azDiff > 180) azDiff -= 360;
                            if (azDiff < -180) azDiff += 360;
                            azimuthAngle = startAz + azDiff * ease;
                            if (azimuthAngle < 0) azimuthAngle += 360;
                            if (azimuthAngle >= 360) azimuthAngle -= 360;
                            elevationAngle = startEl + (targetEl - startEl) * ease;
                            distanceFactor = startDist + (targetDist - startDist) * ease;
                            updatePositions();
                            if (t < 1) requestAnimationFrame(animateSnap);
                            else updatePropsAndTrigger();
                        }
                        animateSnap();
                    }
                    isDragging = false; dragTarget = null; canvas.style.cursor = 'default';
                };

                canvas.addEventListener('mouseup', onMouseUp);
                canvas.addEventListener('mouseleave', onMouseUp);

                canvas.addEventListener('touchstart', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle]);
                    if (intersects.length > 0) {
                        isDragging = true; dragTarget = intersects[0].object;
                        dragTarget.material.emissiveIntensity = 1.0; dragTarget.scale.setScalar(1.3);
                        dragStartMouse.copy(mouse); dragStartDistance = distanceFactor;
                    }
                }, { passive: false });

                canvas.addEventListener('touchmove', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
                    if (isDragging && dragTarget) {
                        raycaster.setFromCamera(mouse, camera);
                        if (dragTarget.userData.type === 'azimuth') {
                            const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                azimuthAngle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                                if (azimuthAngle < 0) azimuthAngle += 360;
                            }
                        } else if (dragTarget.userData.type === 'elevation') {
                            const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.8);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                const relY = intersection.y - CENTER.y;
                                const relZ = intersection.z;
                                elevationAngle = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(relY, relZ)), -30, 60);
                            }
                        } else if (dragTarget.userData.type === 'distance') {
                            const deltaY = mouse.y - dragStartMouse.y;
                            distanceFactor = THREE.MathUtils.clamp(dragStartDistance - deltaY * 1.5, 0.6, 1.4);
                        }
                        updatePositions();
                    }
                }, { passive: false });

                canvas.addEventListener('touchend', (e) => { e.preventDefault(); onMouseUp(); }, { passive: false });
                canvas.addEventListener('touchcancel', (e) => { e.preventDefault(); onMouseUp(); }, { passive: false });

                updatePositions();

                function render() { requestAnimationFrame(render); renderer.render(scene, camera); }
                render();

                new ResizeObserver(() => {
                    camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
                }).observe(wrapper);

                let lastImageUrl = props.imageUrl;
                let lastValue = JSON.stringify(props.value);
                setInterval(() => {
                    if (props.imageUrl !== lastImageUrl) {
                        lastImageUrl = props.imageUrl;
                        updateTextureFromUrl(props.imageUrl);
                    }
                    const currentValue = JSON.stringify(props.value);
                    if (currentValue !== lastValue) {
                        lastValue = currentValue;
                        if (props.value && typeof props.value === 'object') {
                            azimuthAngle = props.value.azimuth ?? azimuthAngle;
                            elevationAngle = props.value.elevation ?? elevationAngle;
                            distanceFactor = props.value.distance ?? distanceFactor;
                            updatePositions();
                        }
                    }
                }, 100);
            };

            initScene();
        })();
        """
        super().__init__(value=value, html_template=html_template, js_on_load=js_on_load, imageUrl=imageUrl, **kwargs)


# --- UI (界面构建) ---
my_css = '''
#col-container { max-width: 1200px; margin: 0 auto; }
.dark .progress-text { color: white !important; }
#camera-3d-control { min-height: 450px; }
.slider-row { display: flex; gap: 10px; align-items: center; }
.fillable{max-width: 1200px !important}
.status-box { font-size: 16px; font-weight: bold; padding: 10px; border-radius: 8px; }
'''

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🎬 Qwen Image Edit 3D 资产生产工具 (高级增强版)
    利用固定视角生成 2D 游戏资产精灵图 (Sprite Sheet)。**新功能：附加提示词、一键 360° 序列生成、内置 AI 抠图。**
    """)

    gallery_history = gr.State([])
    cancel_flag = gr.State(False)

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(label="输入原始图像 (Input Image)", type="pil", height=250)

            with gr.Group():
                gr.Markdown("### 🎨 附加编辑控制")
                extra_prompt = gr.Textbox(
                    label="✍️ 附加编辑提示词 (英文，可选)",
                    placeholder="例如: wearing a red jacket, cyberpunk style...",
                    lines=1
                )
                remove_bg = gr.Checkbox(label="✅ 生成后一键移除背景 (透明 PNG)", value=False)

            gr.Markdown("### 🎮 3D 相机控制")
            camera_3d = CameraControl3D(
                value={"azimuth": 0, "elevation": 0, "distance": 1.0},
                elem_id="camera-3d-control"
            )

            gen_mode = gr.Radio(
                choices=["单张生成（当前视角）", "360° 八方向序列"],
                value="单张生成（当前视角）",
                label="📌 生成模式",
                interactive=True
            )

            with gr.Row():
                run_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
                cancel_btn = gr.Button("❌ 取消任务", variant="stop", size="lg", visible=False)

            with gr.Accordion("🎚️ 详细滑块控制", open=False):
                azimuth_slider = gr.Slider(label="水平旋转 / 方位角 (Azimuth)", minimum=0, maximum=315, step=45, value=0)
                elevation_slider = gr.Slider(label="垂直角度 / 俯仰角 (Elevation)", minimum=-30, maximum=60, step=30, value=0)
                distance_slider = gr.Slider(label="距离 (Distance)", minimum=0.6, maximum=1.4, step=0.4, value=1.0)
                prompt_preview = gr.Textbox(label="将发送给 AI 的拼接提示词", interactive=False)

        with gr.Column(scale=1):
            status_text = gr.Textbox(
                label="📊 任务状态",
                value="等待生成...",
                interactive=False,
                elem_classes=["status-box"]
            )
            result_gallery = gr.Gallery(label="🖼️ 输出历史画廊 (可点击查看、右键下载)", columns=2, preview=True, height=550)
            clear_history_btn = gr.Button("🗑️ 清空历史记录")

            with gr.Accordion("⚙️ 高级推理设置", open=False):
                seed = gr.Slider(label="随机种子 (Seed)", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="随机化种子 (Randomize Seed)", value=True)
                guidance_scale = gr.Slider(label="引导系数 (Guidance Scale)", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                num_inference_steps = gr.Slider(label="推理步数 (Inference Steps)", minimum=1, maximum=20, step=1, value=4)
                height = gr.Slider(label="高度 (Height)", minimum=256, maximum=2048, step=8, value=1024)
                width = gr.Slider(label="宽度 (Width)", minimum=256, maximum=2048, step=8, value=1024)

    # --- 事件处理 ---

    def disable_ui():
        return (
            gr.Button(interactive=False),
            gr.Radio(interactive=False),
            gr.Button(visible=True),
            False
        )

    def enable_ui():
        return (
            gr.Button(interactive=True),
            gr.Radio(interactive=True),
            gr.Button(visible=False),
        )

    def set_cancel_flag():
        return True

    def clear_history():
        return [], []

    def update_prompt_from_sliders(azimuth, elevation, distance, extra):
        az   = 0.0 if azimuth   is None else float(azimuth)
        el   = 0.0 if elevation is None else float(elevation)
        dist = 1.0 if distance  is None else float(distance)
        return build_camera_prompt(az, el, dist, extra or "")

    def sync_3d_to_sliders(camera_value, extra, cur_az, cur_el, cur_dist):
        """3D 控件变化时同步到滑块；值无效时保持当前滑块值不变"""
        print(f"[DEBUG] camera_3d.change 收到: {camera_value}")
        if camera_value and isinstance(camera_value, dict):
            az   = camera_value.get('azimuth')
            el   = camera_value.get('elevation')
            dist = camera_value.get('distance')
            if az is not None and el is not None and dist is not None:
                az, el, dist = float(az), float(el), float(dist)
                return az, el, dist, build_camera_prompt(az, el, dist, extra or "")
        # 无效数据：原样保持当前滑块值
        cur_az   = 0.0 if cur_az   is None else float(cur_az)
        cur_el   = 0.0 if cur_el   is None else float(cur_el)
        cur_dist = 1.0 if cur_dist is None else float(cur_dist)
        return cur_az, cur_el, cur_dist, build_camera_prompt(cur_az, cur_el, cur_dist, extra or "")

    def sync_sliders_to_3d(azimuth, elevation, distance):
        az   = 0.0 if azimuth   is None else float(azimuth)
        el   = 0.0 if elevation is None else float(elevation)
        dist = 1.0 if distance  is None else float(distance)
        return {"azimuth": az, "elevation": el, "distance": dist}

    def update_3d_image(image):
        if image is None: return gr.update(imageUrl=None)
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return gr.update(imageUrl=f"data:image/png;base64,{img_str}")

    # === 事件绑定 ===

    clear_history_btn.click(
        fn=clear_history,
        outputs=[gallery_history, result_gallery]
    ).then(
        fn=lambda: "等待生成...",
        outputs=[status_text]
    )

    cancel_btn.click(fn=set_cancel_flag, outputs=[cancel_flag])

    for trigger in [azimuth_slider, elevation_slider, distance_slider, extra_prompt]:
        trigger.change(
            fn=update_prompt_from_sliders,
            inputs=[azimuth_slider, elevation_slider, distance_slider, extra_prompt],
            outputs=[prompt_preview]
        )

    # 修复：把当前滑块值也作为输入，无效时原样返回
    camera_3d.change(
        fn=sync_3d_to_sliders,
        inputs=[camera_3d, extra_prompt, azimuth_slider, elevation_slider, distance_slider],
        outputs=[azimuth_slider, elevation_slider, distance_slider, prompt_preview]
    )

    for slider in [azimuth_slider, elevation_slider, distance_slider]:
        slider.release(
            fn=sync_sliders_to_3d,
            inputs=[azimuth_slider, elevation_slider, distance_slider],
            outputs=[camera_3d]
        )

    # === 主生成流程 ===
    run_btn.click(
        fn=disable_ui,
        outputs=[run_btn, gen_mode, cancel_btn, cancel_flag]
    ).then(
        fn=lambda: "🔄 准备生成...",
        outputs=[status_text]
    ).then(
        fn=generation_dispatcher,
        inputs=[gen_mode, image, azimuth_slider, elevation_slider, distance_slider,
                extra_prompt, remove_bg, gallery_history,
                seed, randomize_seed, guidance_scale, num_inference_steps, height, width,
                cancel_flag],
        outputs=[gallery_history, result_gallery, status_text, seed, prompt_preview]
    ).then(
        fn=enable_ui,
        outputs=[run_btn, gen_mode, cancel_btn]
    )

    image.upload(
        fn=update_dimensions_on_upload,
        inputs=[image],
        outputs=[width, height]
    ).then(
        fn=update_3d_image,
        inputs=[image],
        outputs=[camera_3d]
    )

    image.clear(
        fn=lambda: gr.update(imageUrl=None),
        outputs=[camera_3d]
    )

if __name__ == "__main__":
    head = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        head=head,
        css=my_css,
        theme=gr.themes.Soft()
    )