"use client";

/**
 * The 3D view: a place, and whatever has been flown through it.
 *
 * A world preview and a finished flight are the same object here: a preview is
 * just a run with no telemetry, so looking around before you ask for something
 * and watching what you asked for are one component, not two.
 */

/* eslint-disable react-hooks/immutability --
 * A three.js scene graph is mutable state that deliberately lives outside
 * React: `useFrame` moves the aircraft, clips the trail and fades an obstacle
 * sixty times a second, and re-rendering the tree to do it would be both slower
 * and wrong. React Compiler's immutability rule models React state, which these
 * objects are not; they are created once per run and driven imperatively, the
 * way react-three-fiber is meant to be used. Everything React *does* own here
 * (the run, the labels flag, the capture handle) is still ordinary props.
 */
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useCallback, useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import { COLORS, sceneColors } from "@/lib/scene/palette";
import {
  buildDrone,
  buildGhost,
  buildPad,
  buildScene,
  disposeObject,
  elevationAt,
  flownShown,
  probeGround,
  showFlownTo,
  tintPad,
} from "@/lib/scene/build";
import { frameAt, lerp, type Playback } from "@/lib/playback";
import { useTheme } from "@/lib/theme";
import type { Run, Tick } from "@/lib/types";

/** Renders one frame from the aircraft's own gimbal, as a JPEG data URL. */
export type Capture = (tick: Tick, tickIndex: number) => string | null;

interface ViewportProps {
  run: Run | null;
  playback: Playback;
  /** Landmark name plates; `l` toggles them on the replay page. */
  labels?: boolean;
  /** Filled in with a capture function once the canvas exists. */
  captureRef?: { current: Capture | null };
  /** Where the aircraft starts, in metres east/north. Defaults to the origin. */
  home?: [number, number] | null;
  /** Where it will put down. Defaults to wherever it started. */
  landing?: [number, number] | null;
  /** Which pad a click moves: the launch pad, the landing pad, or neither. */
  picking?: "home" | "landing" | null;
  onPick?: (x: number, y: number) => void;
  /** A pad was dragged and let go: survey it and keep it, or move it. */
  onMovePad?: (role: "home" | "landing", x: number, y: number) => void;
  /** Told when the browser is drawing this without a GPU, which is worth saying. */
  onSoftwareRenderer?: (renderer: string) => void;
  className?: string;
}

const PHOTO = { fov: 55, pitch: -32, width: 800, height: 600 };

/** Where a dragged pad lands when a world has no terrain to follow. */
const GROUND_PLANE = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

function Scene({
  run,
  playback,
  labels,
  captureRef,
  home,
  landing,
  picking,
  onPick,
  onMovePad,
  onSoftwareRenderer,
}: ViewportProps) {
  const { camera, gl, scene, invalidate, raycaster } = useThree();
  const controls = useRef<OrbitControlsImpl>(null);

  // In demand mode nothing is drawn unless something asks. Every control
  // change bumps this version, so a seek, a rate change or a new world all
  // redraw the paused frame they just changed.
  const version = useSyncExternalStore(
    playback.subscribe,
    playback.getSnapshot,
    playback.getSnapshot,
  );

  // Materials are baked from the stylesheet's colours when the scene is built,
  // so switching palette rebuilds it. That is a rare click and a cheap rebuild,
  // and it keeps one source of truth for what "terrain" is coloured.
  const { theme } = useTheme();
  const built = useMemo(() => (run ? buildScene(run, theme) : null), [run, theme]);
  const aircraft = useMemo(() => buildDrone(), []);
  const gps = useMemo(() => buildGhost(), []);
  // Each pad knows which one it is, so one handler can serve both.
  const pad = useMemo(() => {
    const built = buildPad(COLORS.pad, "start");
    built.userData.role = "home";
    return built;
  }, []);
  const landingPad = useMemo(() => {
    const built = buildPad(COLORS.landing, "landing");
    built.userData.role = "landing";
    return built;
  }, []);
  const dragging = useRef<"home" | "landing" | null>(null);
  const sun = useRef<THREE.DirectionalLight>(null);

  // Each pad stands on whatever ground is under it, which is the same lookup
  // the server does when it decides the altitude to fly to.
  useEffect(() => {
    const [x, y] = home ?? [0, 0];
    pad.position.set(x, elevationAt(run?.world?.terrain, x, y), -y);
    invalidate();
  }, [pad, home, run, invalidate]);

  useEffect(() => {
    // Always drawn, so both ends of the flight are visible before you touch
    // anything. When it has not been moved it sits on the launch pad, which is
    // exactly what "lands back where it started" looks like.
    const [x, y] = landing ?? home ?? [0, 0];
    landingPad.position.set(x, elevationAt(run?.world?.terrain, x, y), -y);
    invalidate();
  }, [landingPad, landing, home, run, invalidate]);

  /**
   * Drag a pad across the ground.
   *
   * The pointer is raycast onto the terrain itself, so the pad follows the
   * hillside rather than a flat plane, and its height is re-read from the
   * ground on every move. The colour warns as you go: `world/landing.py` has
   * the last word when you let go, but a pad that is going to be refused
   * should not look fine until then.
   */
  useEffect(() => {
    if (!onMovePad) return;
    const canvas = gl.domElement;

    const groundUnder = (event: PointerEvent): THREE.Vector3 | null => {
      const rect = canvas.getBoundingClientRect();
      raycaster.setFromCamera(
        new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          -((event.clientY - rect.top) / rect.height) * 2 + 1,
        ),
        camera,
      );
      if (built?.terrain) {
        const [hit] = raycaster.intersectObject(built.terrain, false);
        if (hit) return hit.point;
      }
      const point = new THREE.Vector3();
      return raycaster.ray.intersectPlane(GROUND_PLANE, point) ? point : null;
    };

    const onMove = (event: PointerEvent) => {
      const role = dragging.current;
      if (!role) return;
      const point = groundUnder(event);
      if (!point) return;
      const moving = role === "home" ? pad : landingPad;
      const x = Math.round(point.x * 10) / 10;
      const y = Math.round(-point.z * 10) / 10;
      const ground = probeGround(run?.world?.terrain, x, y);
      moving.position.set(x, ground.elevation, -y);
      tintPad(moving, ground.usable ? (role === "home" ? COLORS.pad : COLORS.landing) : COLORS.nofly);
      invalidate();
    };

    const onUp = (event: PointerEvent) => {
      const role = dragging.current;
      if (!role) return;
      dragging.current = null;
      if (controls.current) controls.current.enabled = true;
      canvas.releasePointerCapture?.(event.pointerId);
      const moving = role === "home" ? pad : landingPad;
      tintPad(moving, role === "home" ? COLORS.pad : COLORS.landing);
      onMovePad(role, Math.round(moving.position.x * 10) / 10, Math.round(-moving.position.z * 10) / 10);
      invalidate();
    };

    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    return () => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
    };
  }, [onMovePad, gl, camera, raycaster, built, pad, landingPad, run, invalidate]);

  const grabPad = useCallback(
    (event: ThreeEvent<PointerEvent>) => {
      const role = event.eventObject.userData.role as "home" | "landing" | undefined;
      if (!onMovePad || !role) return;
      event.stopPropagation();
      dragging.current = role;
      if (controls.current) controls.current.enabled = false; // orbit, or drag: not both
      gl.domElement.setPointerCapture?.(event.pointerId);
    },
    [onMovePad, gl],
  );

  // Memoised because it is an effect dependency: `run?.ticks ?? []` is a fresh
  // array on every render, which would re-frame the camera continuously.
  const ticks = useMemo(() => run?.ticks ?? [], [run]);
  const hasFlight = ticks.length > 0;

  // Dispose a world's geometry when another one replaces it, rather than
  // leaking it every time somebody switches place.
  //
  // Deliberately *not* a cleanup function: Strict Mode mounts, cleans up and
  // mounts again, so disposing "the scene this effect was for" on cleanup
  // destroys the buffers of the scene that is still on screen and the canvas
  // goes black. Disposing the one that has just been superseded happens only
  // when `built` really changes, which is the case that leaks.
  const mounted = useRef<ReturnType<typeof buildScene> | null>(null);
  useEffect(() => {
    const replaced = mounted.current;
    mounted.current = built;
    if (replaced && replaced !== built) disposeObject(replaced.group);
  }, [built]);

  // Frame the camera on whatever has just been loaded, and pull the fog in to
  // match: a 40 m park and a 1.3 km vista cannot share one set of distances.
  useEffect(() => {
    if (!built) return;
    const { extent, center, focus } = built;
    const background = new THREE.Color(sceneColors().background);
    scene.background = background;
    scene.fog = new THREE.Fog(background.getHex(), extent * 1.6, extent * 6);
    camera.position.set(
      center.x + focus * 0.95,
      focus * 0.7,
      center.z + focus * 1.35,
    );
    camera.far = Math.max(6000, extent * 12);
    camera.updateProjectionMatrix();
    controls.current?.target.set(center.x, 10, center.z);
    controls.current?.update();

    // The sun follows the place: its shadow camera has to cover the whole
    // world and no more, or the shadows go soft and blocky.
    const light = sun.current;
    if (light) {
      const reach = Math.max(extent, focus) * 1.1;
      const direction = new THREE.Vector3(-0.55, 0.78, 0.42).normalize();
      light.position.copy(center).addScaledVector(direction, reach * 2);
      light.target.position.copy(center);
      if (!light.target.parent) scene.add(light.target);
      light.target.updateMatrixWorld();
      const shadow = light.shadow.camera;
      shadow.left = -reach;
      shadow.right = reach;
      shadow.top = reach;
      shadow.bottom = -reach;
      shadow.near = 1;
      shadow.far = reach * 5;
      shadow.updateProjectionMatrix();
      light.shadow.needsUpdate = true;
    }
    playback.load(hasFlight ? ticks[ticks.length - 1].timestamp : 0);
    if (hasFlight) playback.setPlaying(true);
  }, [built, camera, scene, playback, hasFlight, ticks]);

  useEffect(() => {
    built?.group.traverse((object) => {
      if (object.userData.isLabel) object.visible = labels !== false;
    });
    invalidate();
  }, [built, labels, invalidate]);

  useEffect(() => {
    invalidate();
  }, [version, invalidate]);

  /**
   * Is this actually being drawn by the GPU?
   *
   * Chrome falls back to a software rasteriser when it loses hardware
   * acceleration (after a GPU process crash, or when too many WebGL contexts
   * are open) and then quietly renders every frame on the CPU across a dozen
   * threads. It looks the same and costs an order of magnitude more, so the
   * page should say so rather than let you wonder why the fan is loud.
   */
  useEffect(() => {
    if (!onSoftwareRenderer) return;
    const context = gl.getContext();
    const info = context.getExtension("WEBGL_debug_renderer_info");
    const renderer = info
      ? String(context.getParameter(info.UNMASKED_RENDERER_WEBGL))
      : String(context.getParameter(context.RENDERER));
    if (/swiftshader|software|llvmpipe|basic render/i.test(renderer)) {
      onSoftwareRenderer(renderer);
    }
  }, [gl, onSoftwareRenderer]);

  // -- the onboard camera -------------------------------------------------
  const photo = useMemo(() => {
    const camera = new THREE.PerspectiveCamera(PHOTO.fov, 4 / 3, 0.5, 8000);
    const canvas = document.createElement("canvas");
    canvas.width = PHOTO.width;
    canvas.height = PHOTO.height;
    // Shots are rendered into an offscreen target rather than read back off the
    // visible canvas. Reading the canvas would mean `preserveDrawingBuffer`,
    // which costs memory bandwidth on every frame of every flight, for the sake
    // of the occasional photograph.
    const target = new THREE.WebGLRenderTarget(PHOTO.width, PHOTO.height);
    target.texture.colorSpace = THREE.SRGBColorSpace;
    return {
      camera,
      canvas,
      ctx: canvas.getContext("2d")!,
      target,
      pixels: new Uint8Array(PHOTO.width * PHOTO.height * 4),
    };
  }, []);

  useEffect(() => () => photo.target.dispose(), [photo]);

  useEffect(() => {
    if (!captureRef) return;
    captureRef.current = (tick, tickIndex) => {
      if (!built?.flown) return null;
      // A camera does not photograph itself, and a shot must not show the
      // future, so hide the aircraft and clip the trail to this moment.
      const wasDrone = aircraft.drone.visible;
      const wasGhost = gps.ghost.visible;
      const wasLink = gps.link.visible;
      const range = flownShown(built.flown);
      aircraft.drone.visible = false;
      gps.ghost.visible = false;
      gps.link.visible = false;
      showFlownTo(built.flown, tickIndex);

      const from = new THREE.Vector3(tick.x, tick.z, -tick.y);
      photo.camera.position.copy(from);
      const yaw = THREE.MathUtils.degToRad(tick.heading);
      const forward = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
      const drop = Math.tan(THREE.MathUtils.degToRad(-PHOTO.pitch));
      photo.camera.lookAt(
        from.clone().add(forward.multiplyScalar(40)).setY(from.y - 40 * drop),
      );

      photo.camera.aspect = 4 / 3;
      photo.camera.updateProjectionMatrix();
      gl.setRenderTarget(photo.target);
      gl.render(scene, photo.camera);
      gl.readRenderTargetPixels(photo.target, 0, 0, PHOTO.width, PHOTO.height, photo.pixels);
      gl.setRenderTarget(null);

      // WebGL hands back rows bottom-up; a canvas wants them top-down.
      const image = photo.ctx.createImageData(PHOTO.width, PHOTO.height);
      const stride = PHOTO.width * 4;
      for (let row = 0; row < PHOTO.height; row++) {
        const from = (PHOTO.height - 1 - row) * stride;
        image.data.set(photo.pixels.subarray(from, from + stride), row * stride);
      }
      photo.ctx.putImageData(image, 0, 0);

      aircraft.drone.visible = wasDrone;
      gps.ghost.visible = wasGhost;
      gps.link.visible = wasLink;
      showFlownTo(built.flown, range);
      return photo.canvas.toDataURL("image/jpeg", 0.82);
    };
    return () => {
      captureRef.current = null;
    };
  }, [captureRef, built, aircraft, gps, photo, gl, scene, camera]);

  // -- the render loop ----------------------------------------------------
  const lastSeek = useRef(-1);

  useFrame((_, delta) => {
    const dt = Math.min(0.1, delta);
    playback.advance(dt);

    const seeked = playback.seekNonce !== lastSeek.current;
    lastSeek.current = playback.seekNonce;

    // Idling rotors would mean the scene is never still, and a scene that is
    // never still can never stop being drawn.
    if (playback.playing) {
      for (const rotor of aircraft.rotors) rotor.rotation.y += 0.8;
    }

    if (!hasFlight || !built) {
      aircraft.drone.visible = false;
      gps.ghost.visible = false;
      gps.link.visible = false;
      controls.current?.update();
      return;
    }

    const frame = frameAt(ticks, playback.time);
    if (!frame) return;
    const { a, b, k, index } = frame;

    const position = new THREE.Vector3(
      lerp(a.x, b.x, k),
      lerp(a.z, b.z, k),
      -lerp(a.y, b.y, k),
    );
    aircraft.drone.position.copy(position);
    aircraft.drone.visible = true;

    // Heading wraps at 360; interpolating across the seam spins the model
    // the long way round, so hold the earlier heading through the wrap.
    const heading = lerp(a.heading, Math.abs(b.heading - a.heading) > 180 ? a.heading : b.heading, k);
    aircraft.drone.rotation.y = -THREE.MathUtils.degToRad(heading);
    const speed = lerp(a.velocity, b.velocity, k);
    aircraft.drone.rotation.z = -THREE.MathUtils.degToRad(Math.min(18, speed * 1.6));

    if (built.flown) showFlownTo(built.flown, index);

    const drift = a.gps_offset;
    if (drift && (drift[0] || drift[1] || drift[2])) {
      const reported = new THREE.Vector3(a.x + drift[0], a.z + drift[2], -(a.y + drift[1]));
      gps.ghost.position.copy(reported);
      gps.ghost.visible = true;
      gps.link.visible = true;
      gps.link.geometry.setFromPoints([position, reported]);
      gps.link.computeLineDistances();
    } else {
      gps.ghost.visible = false;
      gps.link.visible = false;
    }

    for (const mesh of built.obstacles) {
      const age = playback.time - (mesh.userData.appearedAt as number);
      mesh.visible = age >= 0;
      // Flash for a moment when perception first reports it.
      const material = mesh.material as THREE.MeshBasicMaterial;
      material.opacity = age >= 0 && age < 1.2 ? 0.35 + 0.4 * Math.abs(Math.sin(age * 9)) : 0.35;
    }

    if (playback.follow && controls.current) {
      controls.current.target.lerp(position, 0.12);
    }
    controls.current?.update();
    playback.publish({ time: playback.time, tick: a, index, speed }, performance.now() / 1000, seeked);
  });

  return (
    <>
      <hemisphereLight args={theme === "light" ? [0xf2f5ff, 0x8a8170, 1.25] : [0x9fd8ff, 0x0a0f14, 1.1]} />
      <directionalLight
        ref={sun}
        position={[120, 260, 140]}
        color={theme === "light" ? 0xfff1dc : 0xffffff}
        intensity={theme === "light" ? 2.1 : 1.5}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-bias={-0.0004}
        shadow-normalBias={0.8}
      />
      {built && (
        <primitive
          object={built.group}
          // Raycasting costs something, so the scene is only clickable while
          // you are actually placing the pad.
          {...(picking && onPick
            ? {
                onPointerDown: (event: ThreeEvent<PointerEvent>) => {
                  event.stopPropagation();
                  // Scene (x, y, z) back to ENU (x, y): z is negated north.
                  onPick(
                    Math.round(event.point.x * 10) / 10,
                    Math.round(-event.point.z * 10) / 10,
                  );
                },
              }
            : {})}
        />
      )}
      <primitive object={pad} onPointerDown={grabPad} />
      <primitive object={landingPad} onPointerDown={grabPad} />
      <primitive object={aircraft.drone} />
      <primitive object={gps.ghost} />
      <primitive object={gps.link} />
      <OrbitControls
        ref={controls}
        enableDamping
        dampingFactor={0.08}
        maxPolarAngle={Math.PI * 0.495}
        makeDefault
      />
    </>
  );
}

export function Viewport(props: ViewportProps) {
  // Playing means animate; anything else means draw once and stop. A paused
  // replay, or a world nobody is touching, should cost nothing at all -- this
  // is the difference between a fan that spins forever and one that does not.
  useSyncExternalStore(
    props.playback.subscribe,
    props.playback.getSnapshot,
    props.playback.getSnapshot,
  );

  return (
    <div className={props.className} style={props.picking ? { cursor: "crosshair" } : undefined}>
      <Canvas
        shadows={{ type: THREE.PCFShadowMap }}
        frameloop={props.playback.playing ? "always" : "demand"}
        gl={{ antialias: true }}
        camera={{ fov: 55, near: 0.5, far: 6000, position: [120, 110, 160] }}
        onCreated={({ scene }) => {
          scene.background = new THREE.Color(COLORS.background);
        }}
        dpr={[1, 1.75]}
      >
        <Scene {...props} />
      </Canvas>
    </div>
  );
}
