// src/fx/ThreeLogo.tsx
// Three.js hexagonal prism logo — 96x96 canvas, top-right header.
// Lazy-loaded after first paint; app works without it.

import { useEffect, useRef } from 'react'

interface Props {
  className?: string
  color?: string
}

export default function ThreeLogo({ className, color = '#38e8ff' }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef  = useRef<number>(0)
  const mountRef  = useRef(true)

  useEffect(() => {
    mountRef.current = true
    let scene: any, camera: any, renderer: any, prism: any, cube: any

    // Dynamically import three.js (progressive enhancement)
    import('three').then(THREE => {
      if (!mountRef.current || !canvasRef.current) return

      const canvas = canvasRef.current
      const W = canvas.clientWidth  || 96
      const H = canvas.clientHeight || 96

      renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.setSize(W, H, false)

      scene  = new THREE.Scene()
      camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100)
      camera.position.set(0, 0, 2.2)

      // Hexagonal prism
      const hexGeo = new THREE.CylinderGeometry(0.8, 0.8, 0.65, 6, 1, false)
      const mat    = new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 1.0 })
      prism = new THREE.Mesh(hexGeo, mat)
      prism.rotation.x = 0.4
      scene.add(prism)

      // Inner rotating cube
      const boxGeo = new THREE.BoxGeometry(0.55, 0.55, 0.55)
      const matB   = new THREE.MeshBasicMaterial({ color: 0x5dffb0, wireframe: true, transparent: true, opacity: 1.0 })
      cube = new THREE.Mesh(boxGeo, matB)
      scene.add(cube)

      // Ambient point light hint via emissive (BasicMaterial ignores lights — just colors)

      const tick = () => {
        if (!mountRef.current) return
        frameRef.current = requestAnimationFrame(tick)
        prism.rotation.y += 0.012
        cube.rotation.x  += 0.018
        cube.rotation.y  += 0.010
        renderer.render(scene, camera)
      }
      tick()
    }).catch(() => {
      // Three.js unavailable — canvas stays blank (acceptable)
    })

    return () => {
      mountRef.current = false
      cancelAnimationFrame(frameRef.current)
      renderer?.dispose()
    }
  }, [color])

  return <canvas ref={canvasRef} className={className} style={{ width: 96, height: 96 }} />
}
