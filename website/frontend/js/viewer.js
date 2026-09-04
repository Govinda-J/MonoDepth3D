/* js/viewer.js — Three.js point cloud viewers
   Fixed for infer5.py output:
     - pts  : flat float array  [x,y,z, x,y,z, ...]   (raw from depth_to_pc_rgb_full)
     - cols : flat float array  [r,g,b, r,g,b, ...]   in [0,1]  (same length as pts)
*/

const Viewer = (() => {
  const state = {
    scenes:    {},
    renderers: {},
    cameras:   {},
    controls:  {},
    animIds:   {},
    clouds:    { raw: null, refined: null },
    ptSize:    0.005,
  };

  // ── Internal: boot a renderer+scene+camera for one panel ──
  function boot(id) {
    const canvasId = id === 'raw' ? 'canvasRaw' : 'canvasRefined';
    const canvas   = document.getElementById(canvasId);
    if (!canvas) { console.error('Canvas not found:', canvasId); return false; }

    // ── Show canvas, hide placeholder ──
    canvas.style.display = 'block';
    const sfx = id === 'raw' ? 'Raw' : 'Refined';
    const ph = document.getElementById('placeholder' + sfx);
    if (ph) ph.style.display = 'none';

    if (state.renderers[id]) return true;   // already booted

    const W = canvas.clientWidth  || canvas.offsetWidth  || 560;
    const H = canvas.clientHeight || canvas.offsetHeight || 340;

    // renderer
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H, false);
    renderer.setClearColor(0x0d1318, 1);
    state.renderers[id] = renderer;

    // scene
    const scene = new THREE.Scene();
    state.scenes[id] = scene;

    // grid
    const grid = new THREE.GridHelper(4, 32, 0x1a2530, 0x111820);
    grid.position.y = -1.0;
    scene.add(grid);

    // axis lines  X=red  Y=green  Z=blue
    const mkLine = (a, b, color) => {
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...a), new THREE.Vector3(...b)
      ]);
      return new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
    };
    scene.add(mkLine([0,0,0],[0.5,0,0], 0xff4444));
    scene.add(mkLine([0,0,0],[0,0.5,0], 0x44ff88));
    scene.add(mkLine([0,0,0],[0,0,0.5], 0x4488ff));

    // camera
    const camera = new THREE.PerspectiveCamera(45, W / H, 0.001, 500);
    camera.position.set(1.5, 1.2, 1.5);
    camera.lookAt(0, 0, 0);
    state.cameras[id] = camera;

    // orbit control
    const ctrl = makeOrbit(canvas, camera);
    state.controls[id] = ctrl;

    // render loop
    const loop = () => {
      state.animIds[id] = requestAnimationFrame(loop);
      ctrl.update();
      renderer.render(scene, camera);
    };
    loop();

    // resize observer
    new ResizeObserver(() => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight || 340;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }).observe(canvas);

    return true;
  }

  // ── Internal: add/replace point cloud in a scene ──────────
  function putCloud(id, flatPts, flatCols) {
    const scene = state.scenes[id];
    if (!scene) return;

    // remove old
    const old = scene.getObjectByName('cloud');
    if (old) { old.geometry.dispose(); old.material.dispose(); scene.remove(old); }

    const N = flatPts.length / 3;
    if (N === 0) { console.warn('putCloud: empty point array for', id); return; }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(flatPts), 3));

    // ── Colors ────────────────────────────────────────────────
    // infer5.py sends col_raw / col_corr as float32 RGB in [0,1]
    // They come through JSON as a plain JS array — same length as pts.
    let colAttr;
    if (flatCols && flatCols.length === flatPts.length) {
      colAttr = new THREE.BufferAttribute(new Float32Array(flatCols), 3);
    } else {
      // height-based fallback so the cloud is still visible
      console.warn(`Color array length mismatch (pts=${flatPts.length} cols=${flatCols?.length}) — using height gradient`);
      const fallback = new Float32Array(N * 3);
      const baseCol  = id === 'raw'
        ? new THREE.Color(0xff6b35)   // orange for raw
        : new THREE.Color(0x00d97e);  // green for refined
      const darkCol  = new THREE.Color(0x0a1520);
      for (let i = 0; i < N; i++) {
        const y = flatPts[i * 3 + 1];
        // pts from infer5.py have z = -depth (negative).  y is vertical.
        // auto-detect rough range
        const t = Math.max(0, Math.min(1, (y + 1.0) / 2.0));
        const c = new THREE.Color().lerpColors(darkCol, baseCol, t);
        fallback[i*3] = c.r; fallback[i*3+1] = c.g; fallback[i*3+2] = c.b;
      }
      colAttr = new THREE.BufferAttribute(fallback, 3);
    }
    geo.setAttribute('color', colAttr);

    // ── auto-center: compute centroid, shift so cloud is at origin ──
    geo.computeBoundingBox();
    const box = geo.boundingBox;
    const cx  = (box.min.x + box.max.x) / 2;
    const cy  = (box.min.y + box.max.y) / 2;
    const cz  = (box.min.z + box.max.z) / 2;
    const positions = geo.attributes.position.array;
    for (let i = 0; i < positions.length; i += 3) {
      positions[i]   -= cx;
      positions[i+1] -= cy;
      positions[i+2] -= cz;
    }
    geo.attributes.position.needsUpdate = true;
    geo.computeBoundingBox();
    geo.computeBoundingSphere();

    // ── auto-scale camera distance to fit cloud ──────────────
    const sphere = geo.boundingSphere;
    const cam    = state.cameras[id];
    const ctrl   = state.controls[id];
    if (cam && sphere) {
      const dist = sphere.radius * 2.5;
      ctrl.r = dist;
      ctrl.phi   = Math.PI / 4;
      ctrl.theta = Math.PI / 4;
      ctrl.target.set(0, 0, 0);
    }

    const mat   = new THREE.PointsMaterial({ size: state.ptSize, vertexColors: true, sizeAttenuation: true });
    const cloud = new THREE.Points(geo, mat);
    cloud.name  = 'cloud';
    scene.add(cloud);
  }

  // ── Orbit control ──────────────────────────────────────────
  function makeOrbit(canvas, camera) {
    let down = false, rightBtn = false;
    let lx = 0, ly = 0;
    const ctrl = { r: 2.5, phi: Math.PI/4, theta: Math.PI/4, target: new THREE.Vector3() };

    const onDown  = e => { down = true; rightBtn = e.button === 2; lx = e.clientX; ly = e.clientY; };
    const onUp    = ()  => { down = false; };
    const onMove  = e  => {
      if (!down) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      if (rightBtn) {
        const right = new THREE.Vector3().crossVectors(
          camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize();
        ctrl.target.addScaledVector(right, -dx * 0.004);
        ctrl.target.addScaledVector(new THREE.Vector3(0,1,0), dy * 0.004);
      } else {
        ctrl.theta -= dx * 0.007;
        ctrl.phi    = Math.max(0.05, Math.min(Math.PI - 0.05, ctrl.phi - dy * 0.007));
      }
    };
    const onWheel = e => { ctrl.r = Math.max(0.1, Math.min(50, ctrl.r + e.deltaY * 0.005)); };

    canvas.addEventListener('mousedown',  onDown);
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    window.addEventListener('mouseup',    onUp);
    window.addEventListener('mousemove',  onMove);
    canvas.addEventListener('wheel',      onWheel, { passive: true });

    // touch
    let lastDist = null;
    canvas.addEventListener('touchstart', e => {
      if (e.touches.length === 1) { down = true; lx = e.touches[0].clientX; ly = e.touches[0].clientY; }
    });
    canvas.addEventListener('touchend', () => { down = false; lastDist = null; });
    canvas.addEventListener('touchmove', e => {
      e.preventDefault();
      if (e.touches.length === 2) {
        const d = Math.hypot(e.touches[0].clientX-e.touches[1].clientX, e.touches[0].clientY-e.touches[1].clientY);
        if (lastDist !== null) ctrl.r = Math.max(0.1, Math.min(50, ctrl.r - (d-lastDist)*0.015));
        lastDist = d;
      } else if (e.touches.length === 1 && down) {
        const dx = e.touches[0].clientX-lx, dy = e.touches[0].clientY-ly;
        lx = e.touches[0].clientX; ly = e.touches[0].clientY;
        ctrl.theta -= dx*0.007;
        ctrl.phi    = Math.max(0.05, Math.min(Math.PI-0.05, ctrl.phi-dy*0.007));
      }
    }, { passive: false });

    ctrl.update = () => {
      const { r, phi, theta, target } = ctrl;
      camera.position.set(
        target.x + r * Math.sin(phi) * Math.sin(theta),
        target.y + r * Math.cos(phi),
        target.z + r * Math.sin(phi) * Math.cos(theta)
      );
      camera.lookAt(target);
    };
    return ctrl;
  }

  // ── Public API ─────────────────────────────────────────────
  return {
    renderRaw(pts, cols) {
      state.clouds.raw = { pts, cols };
      if (boot('raw')) putCloud('raw', pts, cols);
    },

    renderRefined(pts, cols) {
      state.clouds.refined = { pts, cols };
      if (boot('refined')) putCloud('refined', pts, cols);
    },

    setPointSize(s) {
      state.ptSize = s;
      ['raw', 'refined'].forEach(id => {
        const sc = state.scenes[id]; if (!sc) return;
        const cl = sc.getObjectByName('cloud'); if (!cl) return;
        cl.material.size = s;
      });
    },

    setLargerPts(id, larger) {
      const sc = state.scenes[id]; if (!sc) return;
      const cl = sc.getObjectByName('cloud'); if (!cl) return;
      cl.material.size = larger ? state.ptSize * 4 : state.ptSize;
    },

    resetCam(id) {
      const ctrl = state.controls[id]; if (!ctrl) return;
      ctrl.phi = Math.PI / 4; ctrl.theta = Math.PI / 4;
      ctrl.target.set(0, 0, 0);
    },

    getRaw()     { return state.clouds.raw; },
    getRefined() { return state.clouds.refined; },

    dispose(id) {
      if (state.animIds[id]) cancelAnimationFrame(state.animIds[id]);
      const sc = state.scenes[id];
      if (sc) { const cl = sc.getObjectByName('cloud'); if(cl){cl.geometry.dispose();cl.material.dispose();} }
      if (state.renderers[id]) state.renderers[id].dispose();
      delete state.scenes[id]; delete state.renderers[id];
      delete state.cameras[id]; delete state.controls[id];
      state.clouds[id] = null;
    },
  };
})();

// ── Global onclick helpers called from HTML ────────────────
function setMode(id, mode, btn) {
  btn.closest('.viewer-controls')
     .querySelectorAll('.ctrl-btn')
     .forEach(b => { if (b.textContent==='Points'||b.textContent==='Larger pts') b.classList.remove('active'); });
  btn.classList.add('active');
  Viewer.setLargerPts(id, mode === 'wireframe');
}

function resetCam(id) { Viewer.resetCam(id); }

function exportCloud(id) {
  const data = id === 'raw' ? Viewer.getRaw() : Viewer.getRefined();
  if (!data) return;
  const { pts, cols } = data;
  const N = pts.length / 3;
  let out = `# DepthCloud — ${id}  N=${N}\n`;
  for (let i = 0; i < N; i++) {
    const r = cols ? (cols[i*3]*255|0)   : 255;
    const g = cols ? (cols[i*3+1]*255|0) : 255;
    const b = cols ? (cols[i*3+2]*255|0) : 255;
    out += `${pts[i*3].toFixed(5)} ${pts[i*3+1].toFixed(5)} ${pts[i*3+2].toFixed(5)} ${r} ${g} ${b}\n`;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([out], { type: 'text/plain' }));
  a.download = `depthcloud_${id}.xyz`;
  a.click();
}
